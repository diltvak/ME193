"""Print the LEGO Double Motor's IMU orientation (yaw, pitch, roll) live in the terminal.

Yaw is zeroed at startup, so it reads the rotation since the script began. Pitch and
roll are the tilt relative to level. Raw accelerometer and gyroscope values are shown
too.

Run: python3 imu_test.py  (Ctrl+C to quit)
"""

import time

import legoeducation as le

# ---- Configuration ----
MOTOR_CARD_COLOR = le.LEGO_COLOR_ORANGE
MOTOR_CARD_SERIAL = "1151"
YAW_FACE = le.DEVICE_FACE_TOP   # which side of the Double Motor is the yaw axis
PRINT_INTERVAL_S = 0.05
ANGLE_SCALE = 0.1   # the hub reports angles and gyro rates in tenths of a degree


def main():
    doublemotor = le.DoubleMotor()
    try:
        print("Connecting to Double Motor...")
        doublemotor.connect(card_color=MOTOR_CARD_COLOR, card_serial=MOTOR_CARD_SERIAL)
        if not doublemotor.connected:
            print("Error connecting to Double Motor.")
            return

        doublemotor.imu_set_yaw_face(YAW_FACE)
        doublemotor.imu_reset_yaw_axis(0)
        print("Yaw zeroed. Ctrl+C to quit.\n")

        while True:
            imu = doublemotor.imu_device
            line = (f"Yaw: {imu.yaw * ANGLE_SCALE:7.1f}°  Pitch: {imu.pitch * ANGLE_SCALE:7.1f}°  Roll: {imu.roll * ANGLE_SCALE:7.1f}°"
                    f"  | Accel X/Y/Z: {imu.accelerometerX:6.0f} {imu.accelerometerY:6.0f} "
                    f"{imu.accelerometerZ:6.0f}"
                    f"  | Gyro X/Y/Z (°/s): {imu.gyroscopeX * ANGLE_SCALE:7.1f} {imu.gyroscopeY * ANGLE_SCALE:7.1f} "
                    f"{imu.gyroscopeZ * ANGLE_SCALE:7.1f}")
            print(f"\r{line}", end="", flush=True)
            time.sleep(PRINT_INTERVAL_S)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if doublemotor.connected:
            doublemotor.disconnect()


if __name__ == "__main__":
    main()
