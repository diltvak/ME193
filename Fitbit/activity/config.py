"""Single place for every tunable in the pipeline.

Edit the defaults here, or override from the command line / a saved config.json.
Training saves the Config it used next to the model so live inference always matches.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT.parent            # labeled CSVs live next to pico_main.py
DEFAULT_MODEL_DIR = ROOT / "models"

ACTIVITIES = ["standing", "walking", "running", "biking"]
MOVING = ["walking", "running", "biking"]

# CSV filename prefix -> class label (your cycling file is called "cycling_data_1.csv")
FILE_LABELS = {"standing": "standing", "walking": "walking", "running": "running",
               "cycling": "biking", "biking": "biking"}


@dataclass
class Config:
    # ---- sampling / windowing -------------------------------------------------
    fs: float = 100.0              # Hz, uniform rate everything is resampled to
    window_s: float = 2.5          # window length (2-3 s)
    overlap: float = 0.5           # fraction of overlap between consecutive windows
    max_gap_s: float = 0.3         # gaps <= this are interpolated; longer ones break the stream
    buffer_s: float = 20.0         # rolling buffer length for live inference

    # ---- stage 1: standing-still rule -----------------------------------------
    stand_acc_std_g: float = 0.05      # std of |acc| below this ...
    stand_gyro_std_dps: float = 8.0    # ... and std of |gyro| below this => standing

    # ---- feature extraction ---------------------------------------------------
    f_min_hz: float = 0.5              # dominant-frequency search band
    f_max_hz: float = 5.0
    f_total_max_hz: float = 15.0       # spectrum above this is ignored when normalising power
    fund_halfband_hz: float = 0.4      # +/- width of the "fundamental" / harmonic bands
    nfft: int = 1024                   # zero-padded FFT size
    peak_prominence_g: float = 0.5     # a "sharp peak" in |acc| must stand out by this much
    peak_min_dist_s: float = 0.25

    # ---- fallback rule-based decision tree (used when no trained model exists) -
    # Defaults were read off your recordings (|acc| std: walk ~0.5, run ~1.2, bike ~0.36;
    # gyro fundamental fraction: bike ~0.93, walk/run ~0.5).
    tree_run_acc_std_g: float = 0.85       # |acc| std above this => running
    tree_run_peak_g: float = 4.5           # ... or |acc| peak above this
    tree_bike_gyro_fund_frac: float = 0.70 # gyro power concentrated at fundamental => smooth/sinusoidal

    # ---- real-time ------------------------------------------------------------
    live_hop_s: float = 0.5            # classify this often live (training still uses `overlap`)
    vote_windows: int = 3              # majority vote over the last N window predictions (~1.5 s)
    quick_stand_s: float = 1.0         # if the last this-many seconds are still => standing now
    min_confidence: float = 0.5       # forest votes below this are "uncertain" and not counted
    stale_s: float = 2.0               # no packets for this long => "disconnected"

    # ---- transport (matches pico_main.py / pc_receiver.py) -------------------
    pico_ip: str = "10.247.137.46"
    pico_port: int = 5005
    local_port: int = 5005
    hello_period_s: float = 1.0
    accel_lsb_per_g: float = 8192.0    # +/-4 g
    gyro_lsb_per_dps: float = 65.5     # +/-500 dps

    # ---- derived --------------------------------------------------------------
    @property
    def window_n(self) -> int:
        return int(round(self.window_s * self.fs))

    @property
    def hop_n(self) -> int:
        return max(1, int(round(self.window_n * (1.0 - self.overlap))))

    @property
    def hop_s(self) -> float:
        return self.hop_n / self.fs

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "Config":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})
