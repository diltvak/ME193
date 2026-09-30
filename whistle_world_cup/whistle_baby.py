"""Recognize the whistled tune in RockabyBaby.mp3 when it comes in on the microphone.

Same idea as whistle_recognition.py: the reference recording is turned into a
pitch contour (semitones per frame), the microphone is tracked the same way, and
the most recent stretch of whistling is compared to the reference with dynamic
time warping (DTW), in any key and at somewhat different tempos.

Changes to make it hold up in a real room:
  * Pitch tracking keeps a per-frequency noise floor (calibrated at startup,
    then adapted continuously), so steady noise -- fans, hum, whine from a
    laptop -- is subtracted out instead of drowning the whistle.
  * A frame counts as whistle based on how far its peak stands out (SNR against
    the noise floor, prominence over the rest of the band, and purity measured
    over a window as wide as a slightly wobbly whistle), not on overall volume,
    so a loud clap or talking doesn't mute detection for seconds afterwards.
  * Frames overlap (new frame every ~23 ms), which doubles time resolution at
    the same FFT size.
  * The contour is cleaned before matching: one-frame blips removed, short
    dropouts inside a note bridged, single-frame pitch glitches median-filtered.
  * The key is estimated from the melody's range (not its median, which moves
    depending on which note you hold longest) and then refined from the DTW
    alignment, so slightly off intervals or a different key still match.
  * Every note of the reference counts about equally (short notes aren't
    drowned out by long ones), and each note must be matched on its own -- a
    missing, swapped or wrong note fails even when the average looks fine.
    That makes false matches rare, so the overall threshold can be forgiving
    about slightly off-key intervals.
  * Matching runs on every new whistled frame instead of only once the whistle
    stops, so the tune is recognized right as its last note starts. DTW is
    vectorized with numpy, keeping each check under ~10 ms.

Requires: pip install pyaudio numpy   (macOS: brew install portaudio first)
Decoding the reference uses macOS's built-in `afconvert`.

Run:  python3 whistle_baby.py              listen on the microphone
      python3 whistle_baby.py --test F     run the detector over an audio file
      python3 whistle_baby.py --verbose    also print the best match cost as you whistle
"""

import argparse
import shutil
import subprocess
import tempfile
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).with_name("RockabyBaby.mp3")

RATE = 44100
HOP = 1024            # a new frame every ~23 ms
WINDOW = 2048         # each frame looks at the last ~46 ms
FFT_SIZE = WINDOW * 4  # zero-padding for finer peak location

# Whistling is almost a pure sine wave, typically between ~500 Hz and ~4 kHz.
MIN_FREQ = 500
MAX_FREQ = 4500

# A frame counts as "whistle" when its strongest peak...
MIN_RMS = 0.0005           # (frame isn't digital silence, 0-1 full scale)
MIN_SNR_DB = 12.0          # ...is this far above the room's noise at that frequency,
MIN_PROMINENCE_DB = 20.0   # ...this far above the band's median level,
PURITY_THRESHOLD = 0.5     # ...and holds this share of the (noise-subtracted) band energy
PURITY_HALF_WIDTH = 60.0   # Hz either side of the peak counted as "the peak"
NOISE_OVERSUBTRACT = 2.0   # noise floor is multiplied by this before subtracting
NOISE_RISE = 0.03          # per-frame adaptation of the noise floor when it goes up
NOISE_FALL = 0.3           # ...and when it goes down (fast, so a bad calibration heals)
CALIBRATION_SECONDS = 1.0  # room noise recorded at startup (stay quiet!)

# Contour cleanup (in frames of HOP samples).
MIN_RUN = 2                # whistled stretches shorter than this are dropped as blips
MAX_FILL = 5               # dropouts up to this long (~115 ms) between whistled frames are bridged

# Matching. Silent frames are kept in both contours, so rhythm counts too.
TRANSPOSE_INVARIANT = True       # True = whistling it in any key counts
TEMPO_STRETCHES = np.linspace(0.7, 1.45, 8)  # window lengths tried, relative to the reference
MAX_STEP_COST = 3.0              # semitones; caps the penalty for a wild frame
GAP_COST = 1.0                   # penalty for whistle vs. silence in the same step
WARP_COST = 0.3                  # extra penalty for stretching one side (non-diagonal step)
WARP_BAND = 0.2                  # DTW path must stay within this fraction of the diagonal
NOTE_SPLIT = 1.0                 # semitones; a bigger jump starts a new note
MAX_NOTE_WEIGHT = 3.0            # limit on how much more a short note counts than a long one
REST_WEIGHT = 0.25               # weight of a silent reference frame (a note frame averages 1)
MAX_NOTE_COST = 1.25             # semitones; a reference note matched worse than this on average...
NOTE_PENALTY = 1.0               # ...adds this much per semitone beyond it to the cost
MATCH_THRESHOLD = 0.8            # avg DTW cost (semitones) at or below = match
MIN_VOICED_RATIO = 0.6           # window must have at least this share of the
                                 # reference's whistled frames
COOLDOWN_S = 2.0                 # ignore new matches this long after one fires


# ---------------------------------------------------------------- audio input

def load_audio(path):
    """Decode any file afconvert understands to mono float32 at RATE."""
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        src = path
        # Some of these .mp3 files are really AAC/M4A; afconvert trusts the
        # extension, so give it the right one.
        with open(path, "rb") as f:
            if f.read(12)[4:8] == b"ftyp":
                src = Path(tmp) / "in.m4a"
                shutil.copy(path, src)
        out = Path(tmp) / "out.wav"
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1",
                        str(src), str(out)], check=True, capture_output=True)
        with wave.open(str(out)) as w:
            data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768


def frames_of(audio):
    """Overlapping WINDOW-long frames, one every HOP samples."""
    for i in range(0, len(audio) - WINDOW + 1, HOP):
        yield audio[i:i + WINDOW]


# ------------------------------------------------------------ pitch tracking

_window = np.hanning(WINDOW)
_freqs = np.fft.rfftfreq(FFT_SIZE, 1 / RATE)
_band = (_freqs >= MIN_FREQ) & (_freqs <= MAX_FREQ)
_band_freqs = _freqs[_band]
_bin_hz = _freqs[1] - _freqs[0]
_half = int(round(PURITY_HALF_WIDTH / _bin_hz))


def band_power(samples):
    return np.abs(np.fft.rfft(samples * _window, n=FFT_SIZE))[_band] ** 2


class PitchTracker:
    """Turns audio frames into whistle pitches (MIDI number, or None)."""

    def __init__(self):
        self.noise = None  # per-bin noise power in the whistle band

    def calibrate(self, frames):
        self.noise = np.mean([band_power(f) for f in frames], axis=0)

    def process(self, samples):
        power = band_power(samples)
        pitch = self._detect(samples, power)
        if pitch is None:
            self._update_noise(power)
        return pitch

    def _detect(self, samples, power):
        if np.sqrt(np.mean(samples ** 2)) < MIN_RMS:
            return None
        noise = self.noise if self.noise is not None else np.zeros_like(power)
        clean = np.maximum(power - NOISE_OVERSUBTRACT * noise, 0.0)
        peak = int(np.argmax(clean))
        peak_power = power[peak]
        if peak_power < 10 ** (MIN_SNR_DB / 10) * noise[peak]:
            return None
        if peak_power < 10 ** (MIN_PROMINENCE_DB / 10) * np.median(power):
            return None
        total = clean.sum()
        if total <= 0 or clean[max(peak - _half, 0):peak + _half + 1].sum() < PURITY_THRESHOLD * total:
            return None

        # Parabolic interpolation around the peak for sub-bin accuracy.
        offset = 0.0
        if 0 < peak < len(power) - 1:
            a, b, c = 0.5 * np.log(power[peak - 1:peak + 2] + 1e-20)
            denom = a - 2 * b + c
            if denom < 0:
                offset = 0.5 * (a - c) / denom
        freq = _band_freqs[peak] + offset * _bin_hz
        return 69 + 12 * np.log2(freq / 440.0)

    def _update_noise(self, power):
        if self.noise is None:
            self.noise = power.copy()
            return
        # Follow drops quickly; rise slowly, and never by more than 4x per frame,
        # so the tail of a whistle can't inflate the floor.
        target = np.minimum(power, 4 * self.noise)
        rate = np.where(power < self.noise, NOISE_FALL, NOISE_RISE)
        self.noise += rate * (target - self.noise)


def pitch_track(audio):
    """Per-frame pitch (MIDI or None) of a whole recording."""
    tracker = PitchTracker()
    return [tracker.process(frame) for frame in frames_of(audio)]


def build_template(path):
    """Reference contour from first to last whistled frame (NaN = silence)."""
    track = clean_contour(to_array(pitch_track(load_audio(path))))
    voiced = np.flatnonzero(~np.isnan(track))
    if len(voiced) == 0:
        raise SystemExit(f"No whistle found in {path}")
    return track[voiced[0]:voiced[-1] + 1]


# ------------------------------------------------------------------ matching

def to_array(pitches):
    return np.array([np.nan if p is None else p for p in pitches], dtype=float)


def _runs(mask):
    """(start, end) index pairs of the True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def clean_contour(seq):
    """Drop blips, bridge short dropouts, and median-filter single-frame glitches."""
    seq = seq.copy()
    for start, end in _runs(~np.isnan(seq)):
        if end - start < MIN_RUN:
            seq[start:end] = np.nan
    for start, end in _runs(np.isnan(seq)):
        if 0 < start and end < len(seq) and end - start <= MAX_FILL:
            mid = (start + end) // 2  # each side of the gap holds its neighbour's pitch
            seq[start:mid] = seq[start - 1]
            seq[mid:end] = seq[end]
    # Median of 3; at the edge of a whistled stretch the missing neighbour is
    # replaced by the one on the other side, so onset/offset glitches go too.
    padded = np.concatenate(([np.nan], seq, [np.nan]))
    prev, nxt = padded[:-2], padded[2:]
    prev, nxt = np.where(np.isnan(prev), nxt, prev), np.where(np.isnan(nxt), prev, nxt)
    both = ~np.isnan(seq) & ~np.isnan(prev)
    seq[both] = np.median(np.stack((prev, seq, nxt))[:, both], axis=0)
    return seq


def normalize(seq):
    """Shift a contour so the middle of its pitch range is 0 (if key doesn't matter).

    The midpoint of lowest and highest note doesn't depend on how long each
    note is held, unlike the median."""
    if not TRANSPOSE_INVARIANT:
        return seq
    voiced = seq[~np.isnan(seq)]
    return seq - 0.5 * (voiced.min() + voiced.max())


def note_ids(seq):
    """Number the notes of a contour (-1 for rests). A new note starts wherever
    the pitch jumps by more than NOTE_SPLIT semitones."""
    silent = np.isnan(seq)
    change = np.abs(np.diff(seq)) > NOTE_SPLIT
    change |= silent[1:] != silent[:-1]
    ids = np.concatenate(([0], np.cumsum(change)))
    ids[silent] = -1
    return ids


def note_weights(notes):
    """Per-frame weight that makes every note count about the same, so a short
    note -- like the tune's last one -- can't just be skipped. Rests count
    little, since whether you pause between notes is mostly style."""
    voiced = notes >= 0
    lengths = np.bincount(notes[voiced])[notes[voiced]]
    weights = np.full(len(notes), REST_WEIGHT)
    weights[voiced] = np.clip(lengths.mean() / lengths, 1 / MAX_NOTE_WEIGHT, MAX_NOTE_WEIGHT)
    return weights / weights.mean()


def local_costs(a, b):
    """Frame-vs-frame distance between two contours (NaN = silence)."""
    local = np.minimum(np.abs(a[:, None] - b[None, :]), MAX_STEP_COST)
    silent_a, silent_b = np.isnan(a)[:, None], np.isnan(b)[None, :]
    local[silent_a | silent_b] = GAP_COST
    local[silent_a & silent_b] = 0.0
    return local


def dtw(a, b, notes, weights):
    """Score how well contour b matches template a, plus the median pitch
    difference (b - a) along the best alignment.

    The score is the average per-frame DTW distance (a's frames weighted by
    `weights`), plus a penalty for every note of a (numbered by `notes`) that
    is off by more than MAX_NOTE_COST on average -- so a missing, extra or
    swapped note fails even when the rest of the tune fits.

    Uses the symmetric step pattern (1,1), (1,2), (2,1), which limits local
    tempo changes to 2x and -- because each row only depends on the two rows
    above -- lets every row be computed at once with numpy."""
    n, m = len(a), len(b)
    raw = local_costs(a, b)
    local = raw * weights[:, None]
    i_idx, j_idx = np.arange(n)[:, None], np.arange(m)[None, :]
    local[np.abs(i_idx / n - j_idx / m) > WARP_BAND] = np.inf
    left = np.concatenate((np.full((n, 1), np.inf), local[:, :-1]), axis=1)  # local[i, j-1]

    acc = np.full((n + 2, m + 2), np.inf)  # acc[i+2, j+2] = best cost ending at (i, j)
    acc[1, 1] = 0.0
    choice = np.zeros((n, m), dtype=np.int8)
    for i in range(n):
        r = i + 2
        up = local[i - 1] if i > 0 else np.full(m, np.inf)
        options = np.stack((acc[r - 1, 1:-1] + 2 * local[i],                           # (i-1, j-1)
                            acc[r - 1, :-2] + 2 * left[i] + local[i] + WARP_COST,      # (i-1, j-2)
                            acc[r - 2, 1:-1] + 2 * up + local[i] + WARP_COST))         # (i-2, j-1)
        choice[i] = np.argmin(options, axis=0)
        acc[r, 2:] = options[choice[i], np.arange(m)]

    total = acc[n + 1, m + 1]
    if not np.isfinite(total):
        return np.inf, 0.0

    # Walk the best path back to collect aligned frame pairs.
    pairs, i, j = [], n - 1, m - 1
    while i >= 0 and j >= 0:
        pairs.append((i, j))
        step = choice[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            pairs.append((i, j - 1))
            i, j = i - 1, j - 2
        else:
            pairs.append((i - 1, j))
            i, j = i - 2, j - 1
    ia, jb = np.array(pairs).T
    diffs = b[jb] - a[ia]
    diffs = diffs[~np.isnan(diffs)]
    shift = float(np.median(diffs)) if len(diffs) else 0.0

    # For the per-note check, a note met by silence counts as fully missed.
    on_note = notes[ia] >= 0
    frame_cost = np.where(np.isnan(b[jb]), MAX_STEP_COST, raw[ia, jb])[on_note]
    per_note = (np.bincount(notes[ia][on_note], frame_cost)
                / np.maximum(np.bincount(notes[ia][on_note]), 1))
    miss = np.maximum(per_note - MAX_NOTE_COST, 0.0).sum()
    return total / (n + m) + NOTE_PENALTY * miss, shift


def match_cost(template, notes, weights, window):
    """DTW score of a (normalized) live window against the template, with the
    key refined once from the alignment."""
    cost, shift = dtw(template, window, notes, weights)
    if TRANSPOSE_INVARIANT and np.isfinite(cost) and abs(shift) > 0.2:
        cost = min(cost, dtw(template, window - shift, notes, weights)[0])
    return cost


class TuneMatcher:
    """Feed one frame's pitch at a time; reports when the tune has been heard."""

    def __init__(self, template):
        self.template = normalize(template)
        self.notes = note_ids(template)
        self.weights = note_weights(self.notes)
        self.template_voiced = np.count_nonzero(~np.isnan(template))
        self.history = deque(maxlen=int(len(template) * max(TEMPO_STRETCHES)) + MAX_FILL + 4)
        self.cooldown_until = -np.inf
        self.last_best = None

    def update(self, pitch, now):
        """Return the match cost if the tune was just recognized, else None."""
        self.history.append(pitch)
        # Check on every whistled frame, so the tune is caught during its last note.
        if pitch is None or now < self.cooldown_until:
            return None

        best = self.best_cost()
        self.last_best = best
        if best is not None and best <= MATCH_THRESHOLD:
            self.cooldown_until = now + COOLDOWN_S
            self.history.clear()
            return best
        return None

    def best_cost(self):
        """Lowest DTW cost over the tempo stretches, or None if not enough whistle."""
        frames = clean_contour(to_array(self.history))
        voiced = np.flatnonzero(~np.isnan(frames))
        if len(voiced) == 0:
            return None
        frames = frames[:voiced[-1] + 1]  # cleanup may have dropped a trailing blip
        best, tried = None, set()
        for stretch in TEMPO_STRETCHES:
            window = frames[-int(round(len(self.template) * stretch)):]
            is_voiced = ~np.isnan(window)
            if np.count_nonzero(is_voiced) < MIN_VOICED_RATIO * self.template_voiced * stretch:
                continue
            # Start the window at its first whistled frame, like the template.
            start = np.flatnonzero(is_voiced)[0]
            if len(window) - start in tried:
                continue
            tried.add(len(window) - start)
            cost = match_cost(self.template, self.notes, self.weights, normalize(window[start:]))
            best = cost if best is None else min(best, cost)
        return best


# --------------------------------------------------------------------- modes

def announce(cost, when):
    print(f"\n*** Rock-a-bye Baby whistle recognized at {when}  (cost {cost:.2f}) ***")


def run_file(path, matcher):
    """Offline check: stream a recording through the matcher frame by frame."""
    audio = load_audio(path)
    hits = 0
    for k, pitch in enumerate(pitch_track(audio)):
        t = (k * HOP + WINDOW) / RATE
        cost = matcher.update(pitch, t)
        if cost is not None:
            hits += 1
            announce(cost, f"{t:.2f} s")
    print(f"{path}: {hits} match(es)")
    return hits


def run_microphone(matcher, verbose=False):
    import pyaudio

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                     input=True, frames_per_buffer=HOP)
    buffer = np.zeros(WINDOW, dtype=np.float32)

    def read():
        """Advance by one HOP and return the latest WINDOW samples."""
        nonlocal buffer
        new = np.frombuffer(stream.read(HOP, exception_on_overflow=False), dtype=np.float32)
        buffer = np.concatenate((buffer[HOP:], new))
        return buffer

    tracker = PitchTracker()
    print(f"Calibrating: stay quiet for {CALIBRATION_SECONDS:g} s...")
    for _ in range(WINDOW // HOP):  # fill the buffer first
        read()
    n = max(1, int(CALIBRATION_SECONDS * RATE / HOP))
    tracker.calibrate([read().copy() for _ in range(n)])

    print("Listening for Rock-a-bye Baby (Ctrl+C to stop)\n")
    try:
        while True:
            pitch = tracker.process(read())
            cost = matcher.update(pitch, time.monotonic())
            status = f"whistle {pitch:5.1f} (MIDI)" if pitch is not None else "..."
            if verbose and matcher.last_best is not None:
                status += f"   best cost {matcher.last_best:.2f}"
            print(f"\r{status:<50}", end="", flush=True)
            if cost is not None:
                announce(cost, time.strftime("%H:%M:%S"))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def main():
    parser = argparse.ArgumentParser(description="Recognize Rock-a-bye Baby whistled on the microphone.")
    parser.add_argument("--test", nargs="+", metavar="FILE",
                        help="run the detector over audio files instead of the mic")
    parser.add_argument("--verbose", action="store_true",
                        help="show the best match cost while whistling (for tuning MATCH_THRESHOLD)")
    args = parser.parse_args()

    template = build_template(REFERENCE)
    print(f"Reference: {np.count_nonzero(~np.isnan(template))} whistled frames "
          f"over {len(template) * HOP / RATE:.2f} s")

    if args.test:
        for path in args.test:
            run_file(path, TuneMatcher(template))
    else:
        run_microphone(TuneMatcher(template), args.verbose)


if __name__ == "__main__":
    main()
