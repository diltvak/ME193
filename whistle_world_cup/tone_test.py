"""Detect the note you're whistling using the laptop microphone.

Shows a live oscilloscope window with the microphone waveform and the ideal
sine wave at the detected pitch overlaid on top of it.

Requires: pip install pyaudio numpy matplotlib
(On macOS, pyaudio needs PortAudio first: brew install portaudio)

Run: python tone_test.py   (Ctrl+C or close the plot window to quit)
"""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

RATE = 44100          # samples per second
CHUNK = 4096          # samples per read (~93 ms); bigger = finer frequency resolution
FFT_SIZE = CHUNK * 4  # zero-padding for a smoother spectrum

# Whistling is almost a pure sine wave, typically between ~500 Hz and ~4 kHz.
MIN_FREQ = 400
MAX_FREQ = 4500

VOLUME_THRESHOLD = 0.01  # RMS level (0-1) below which we treat input as silence
PURITY_THRESHOLD = 0.15  # fraction of band energy in the peak; whistles are "pure"

# Confidence gating against background noise.
CALIBRATION_SECONDS = 1.5  # room noise recorded at startup (stay quiet!)
SNR_THRESHOLD_DB = 15      # peak must be this far above the room's noise floor
STABLE_FRAMES = 3          # consecutive frames (~93 ms each) the pitch must hold
STABLE_CENTS = 50          # max pitch wobble across those frames

SCOPE_MS = 10  # how much of the waveform to show in the plot (milliseconds)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note(freq):
    """Return (note name with octave, cents off from that note)."""
    midi = 69 + 12 * np.log2(freq / 440.0)
    nearest = int(round(midi))
    cents = (midi - nearest) * 100
    name = NOTE_NAMES[nearest % 12] + str(nearest // 12 - 1)
    return name, cents


def band_power(samples, window, band):
    """Power spectrum of one chunk, restricted to the whistling band."""
    spectrum = np.abs(np.fft.rfft(samples * window, n=FFT_SIZE))
    return spectrum[band] ** 2


def calibrate_noise(stream, window, band):
    """Record the room for a moment and return its average power per frequency bin.

    Steady background noise (fans, HVAC, hum, chatter) isn't flat across
    frequencies, so we keep a per-bin profile rather than a single number.
    """
    n_frames = max(1, int(CALIBRATION_SECONDS * RATE / CHUNK))
    profile = np.zeros(band.sum())
    for _ in range(n_frames):
        data = stream.read(CHUNK, exception_on_overflow=False)
        profile += band_power(np.frombuffer(data, dtype=np.float32), window, band)
    return profile / n_frames + 1e-12


def snr_db(power, noise_profile, lo, hi):
    """Signal-to-noise ratio (dB) of the peak region vs. the room's noise there."""
    return 10 * np.log10(power[lo:hi].sum() / noise_profile[lo:hi].sum())


def detect_pitch(samples, window, freqs, band, noise_profile):
    """Return (dominant frequency or None, SNR in dB) for the whistling band.

    A frame only counts as a whistle when it is loud enough, stands well above
    the calibrated noise floor (SNR), and has its energy concentrated in one
    narrow peak (purity) like a sine wave.
    """
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < VOLUME_THRESHOLD:
        return None, -np.inf

    power = band_power(samples, window, band)
    band_spectrum = np.sqrt(power)
    peak = np.argmax(power)
    lo, hi = max(peak - 3, 0), peak + 4

    snr = snr_db(power, noise_profile, lo, hi)
    if snr < SNR_THRESHOLD_DB:
        return None, snr

    # Reject noisy input (speech, claps, etc.) where energy isn't concentrated.
    if power[lo:hi].sum() / power.sum() < PURITY_THRESHOLD:
        return None, snr

    # Parabolic interpolation around the peak for sub-bin accuracy.
    if 0 < peak < len(band_spectrum) - 1:
        a, b, c = np.log(band_spectrum[peak - 1:peak + 2] + 1e-12)
        offset = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        offset = 0.0

    bin_width = freqs[1] - freqs[0]
    return freqs[band][peak] + offset * bin_width, snr


class StabilityGate:
    """Only confirm a pitch once it has held steady for several frames in a row.

    Random noise can occasionally pass the SNR/purity checks for a single frame,
    but it almost never produces the same pitch repeatedly; a whistle does.
    """

    def __init__(self, frames=STABLE_FRAMES, tolerance_cents=STABLE_CENTS):
        self.history = deque(maxlen=frames)
        self.tolerance = tolerance_cents

    def update(self, freq):
        """Feed one frame's pitch (or None); return the confirmed pitch or None."""
        if freq is None:
            self.history.clear()
            return None
        self.history.append(freq)
        if len(self.history) < self.history.maxlen:
            return None
        spread = 1200 * np.log2(max(self.history) / min(self.history))
        if spread > self.tolerance:
            return None
        return float(np.median(self.history))


def trigger_index(samples, max_start):
    """First rising zero crossing, so the plotted wave holds still like a scope."""
    crossings = np.where((samples[:-1] < 0) & (samples[1:] >= 0))[0]
    crossings = crossings[crossings < max_start]
    return int(crossings[0]) + 1 if len(crossings) else 0


def fit_sine(samples, t, freq):
    """Least-squares fit of A*sin(2*pi*f*t) + B*cos(2*pi*f*t) at a known freq."""
    w = 2 * np.pi * freq * t
    basis = np.column_stack([np.sin(w), np.cos(w)])
    coeffs, *_ = np.linalg.lstsq(basis, samples, rcond=None)
    return basis @ coeffs


def setup_plot(n):
    """Create the scope window; returns (fig, raw line, sine line, title)."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4))
    t_ms = np.arange(n) / RATE * 1000
    raw_line, = ax.plot(t_ms, np.zeros(n), color="0.6", lw=1, label="Microphone")
    sine_line, = ax.plot(t_ms, np.full(n, np.nan), color="tab:blue", lw=2,
                         label="Detected sine")
    ax.set_xlim(0, t_ms[-1])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    title = ax.set_title("Listening...")
    fig.tight_layout()
    fig.show()
    return fig, ax, raw_line, sine_line, title


def cents_meter(cents, width=21):
    """Little ASCII tuner bar: '|' marks where you are relative to the note."""
    pos = int(round((cents + 50) / 100 * (width - 1)))
    bar = ["-"] * width
    bar[width // 2] = "+"
    bar[min(max(pos, 0), width - 1)] = "|"
    return "".join(bar)


def main():
    window = np.hanning(CHUNK)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / RATE)
    band = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)

    scope_n = int(RATE * SCOPE_MS / 1000)
    t = np.arange(CHUNK) / RATE
    fig, ax, raw_line, sine_line, title = setup_plot(scope_n)

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)

    print(f"Calibrating: stay quiet for {CALIBRATION_SECONDS:g} s...")
    noise_profile = calibrate_noise(stream, window, band)
    gate = StabilityGate()

    print("Listening... whistle a note (Ctrl+C to stop)\n")
    try:
        while plt.fignum_exists(fig.number):
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)

            raw_freq, snr = detect_pitch(samples, window, freqs, band, noise_profile)
            freq = gate.update(raw_freq)
            start = trigger_index(samples, CHUNK - scope_n)
            view = slice(start, start + scope_n)
            raw_line.set_ydata(samples[view])

            if freq is None:
                line = f"  ...  (listening)  SNR {snr:5.1f} dB" if np.isfinite(snr) else "  ...  (listening)"
                sine_line.set_ydata(np.full(scope_n, np.nan))
                title.set_text("Listening...")
            else:
                note, cents = freq_to_note(freq)
                line = (f"{note:>4}  {freq:7.1f} Hz  {cents:+5.0f} cents  "
                        f"[{cents_meter(cents)}]  SNR {snr:4.1f} dB")
                sine_line.set_ydata(fit_sine(samples, t, freq)[view])
                title.set_text(f"{note}  {freq:.1f} Hz  ({cents:+.0f} cents, SNR {snr:.0f} dB)")
            print(f"\r{line:<75}", end="", flush=True)

            # Auto-scale so quiet whistles are still visible.
            peak = max(np.abs(samples[view]).max(), 0.02)
            ax.set_ylim(-1.2 * peak, 1.2 * peak)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        print("\nPlot closed.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        plt.close("all")
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
