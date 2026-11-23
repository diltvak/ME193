# MicroPython for Raspberry Pi Pico 2 W: sample an MPU6050 at ~100 Hz and
# stream small UDP batches to a PC. Save to the Pico as main.py.
#
# Two-leg version: flash this same file onto BOTH Picos. Set LEG below so the
# serial printouts say which one you are looking at. The PC tells the two
# streams apart by IP address, so the packet format is unchanged.
# The onboard LED is on while a PC is connected and streaming.
#
# Packet: <seq u32><n u8> followed by n records of <ticks_us u32><14 raw MPU bytes>
# The 14 raw bytes are big-endian int16: ax ay az temp gx gy gz.

import network
import socket
import struct
import time
from machine import I2C, Pin, Timer

LEG = "right"                # "left" or "right": only used in printouts

WIFI_SSID = "Tufts_Robot"
WIFI_PASSWORD = "Jumbos1852!"
PORT = 5005                 # Pico listens here; the PC's hello packet tells it where to stream
HELLO_TIMEOUT_MS = 5000     # stop streaming if the PC stops sending hellos

SAMPLE_HZ = 100
BATCH_SAMPLES = 5           # send once this many samples are waiting (5 -> ~20 packets/s)
MAX_PER_PACKET = 20         # cap so a packet stays well under the MTU

MPU_ADDR = 0x68
RECORD = 18                 # 4 byte timestamp + 14 byte sensor burst
RING_SLOTS = 200            # ~2 s of buffer for Wi-Fi stalls

ring = bytearray(RING_SLOTS * RECORD)
ring_mv = memoryview(ring)
pkt = bytearray(5 + MAX_PER_PACKET * RECORD)
pkt_mv = memoryview(pkt)

head = 0                    # written only by the timer callback
tail = 0                    # written only by the main loop
dropped = 0
i2c_errors = 0

i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400_000)
led = Pin("LED", Pin.OUT)


def find_mpu():
    # Scan the I2C bus until an MPU6050 answers. EIO on the first read means nothing is
    # answering at that address: almost always wiring (SDA/SCL swapped or loose, no power).
    global MPU_ADDR
    while True:
        found = i2c.scan()
        for addr in (0x68, 0x69):              # 0x69 if the AD0 pin is pulled high
            if addr in found:
                MPU_ADDR = addr
                print("[%s leg] MPU6050 found at %s" % (LEG, hex(addr)))
                return
        print("[%s leg] No MPU6050 on I2C (devices found: %s)." % (LEG, [hex(a) for a in found]))
        print("  Check: SDA -> GP0 (pin 1), SCL -> GP1 (pin 2), VCC -> 3V3 (pin 36), GND -> GND.")
        print("  Retrying in 2 s...")
        for _ in range(10):                    # fast blink = sensor problem
            led.toggle()
            time.sleep_ms(200)
        led.off()


def mpu_init():
    find_mpu()
    who = i2c.readfrom_mem(MPU_ADDR, 0x75, 1)[0]
    if who != 0x68:
        print("Unexpected WHO_AM_I:", hex(who))
    i2c.writeto_mem(MPU_ADDR, 0x6B, b"\x01")   # wake, PLL clock
    time.sleep_ms(100)
    i2c.writeto_mem(MPU_ADDR, 0x1A, b"\x03")   # DLPF ~44 Hz, avoids aliasing at 100 Hz
    i2c.writeto_mem(MPU_ADDR, 0x19, bytes([1000 // SAMPLE_HZ - 1]))
    i2c.writeto_mem(MPU_ADDR, 0x1B, b"\x08")   # gyro +/-500 dps
    i2c.writeto_mem(MPU_ADDR, 0x1C, b"\x08")   # accel +/-4 g


def sample(_):
    # Runs from the timer; only touches preallocated buffers.
    global head, dropped, i2c_errors
    nxt = (head + 1) % RING_SLOTS
    if nxt == tail:
        dropped += 1
        return
    off = head * RECORD
    try:
        struct.pack_into("<I", ring, off, time.ticks_us())
        i2c.readfrom_mem_into(MPU_ADDR, 0x3B, ring_mv[off + 4:off + RECORD])
    except OSError:
        i2c_errors += 1
        return
    head = nxt


def wifi_connect(wlan):
    wlan.active(True)
    wlan.config(pm=network.WLAN.PM_NONE)       # power save adds latency
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    t0 = time.ticks_ms()
    while not wlan.isconnected():
        if time.ticks_diff(time.ticks_ms(), t0) > 20000:
            print("Wi-Fi timeout, retrying")
            wlan.connect(WIFI_SSID, WIFI_PASSWORD)
            t0 = time.ticks_ms()
        time.sleep_ms(200)
    print("Connected:", wlan.ifconfig())
    print("[%s leg] Pico IP address: %s" % (LEG, wlan.ifconfig()[0]))
    mac = wlan.config("mac")
    print("[%s leg] Pico MAC address: %s" % (LEG, ":".join("%02x" % b for b in mac)))


def main():
    global tail
    led.off()
    mpu_init()
    wlan = network.WLAN(network.STA_IF)
    wifi_connect(wlan)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.setblocking(False)                    # a Wi-Fi hiccup must not stall the loop

    Timer(freq=SAMPLE_HZ, mode=Timer.PERIODIC, callback=sample)

    seq = 0
    addr = None                                # PC address, learned from its hello packets
    last_hello = 0
    last_check = time.ticks_ms()
    print("[%s leg] Waiting for PC to connect to %s:%d" % (LEG, wlan.ifconfig()[0], PORT))
    while True:
        try:
            _, sender = sock.recvfrom(16)
            if addr is None:
                print("PC connected:", sender[0])
                led.on()
                tail = head                    # discard stale samples
            addr = sender
            last_hello = time.ticks_ms()
        except OSError:
            pass                               # no hello waiting

        if addr is not None and time.ticks_diff(time.ticks_ms(), last_hello) > HELLO_TIMEOUT_MS:
            print("PC timed out, waiting for new connection")
            addr = None
            led.off()

        if addr is None:
            tail = head                        # not streaming: keep the ring empty
            time.sleep_ms(20)
            continue

        avail = (head - tail) % RING_SLOTS
        if avail >= BATCH_SAMPLES:
            n = min(avail, MAX_PER_PACKET)
            nbytes = n * RECORD
            struct.pack_into("<IB", pkt, 0, seq, n)
            start = tail * RECORD
            if tail + n <= RING_SLOTS:
                pkt_mv[5:5 + nbytes] = ring_mv[start:start + nbytes]
            else:
                first = (RING_SLOTS - tail) * RECORD
                pkt_mv[5:5 + first] = ring_mv[start:]
                pkt_mv[5 + first:5 + nbytes] = ring_mv[:nbytes - first]
            try:
                sock.sendto(pkt_mv[:5 + nbytes], addr)
            except OSError:
                pass                           # packet lost; PC sees the seq gap
            seq = (seq + 1) & 0xFFFFFFFF
            tail = (tail + n) % RING_SLOTS
        else:
            time.sleep_ms(2)

        if time.ticks_diff(time.ticks_ms(), last_check) > 2000:
            last_check = time.ticks_ms()
            if not wlan.isconnected():
                wifi_connect(wlan)


main()
