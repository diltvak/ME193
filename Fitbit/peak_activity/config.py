"""Every tunable for the single-axis peak classifier.

Field names for sampling/transport match activity.config.Config so the receiver in
activity.data_io (UdpSource, ReplaySource, resample_uniform) works with this class unchanged.
`calibrate.py` writes the fitted axis + thresholds to models/thresholds.json; `live.py` loads them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT.parent             # labeled CSVs live next to pico_main.py
DEFAULT_MODEL_DIR = ROOT / "models"

ACTIVITIES = ["standing", "biking", "walking", "running"]   # ordered by peak size, low -> high
AXES = ["x", "y", "z"]


@dataclass
class PeakConfig:
    # ---- the one signal used --------------------------------------------------
    axis: int = 0                      # 0=ax, 1=ay, 2=az. x separated the classes best on the recordings
    peak_pct: float = 95.0             # "peak" = this percentile of |a - median(a)| in the window

    # ---- thresholds on the peak (g); fitted by calibrate.py -------------------
    t_stand_bike: float = 0.20         # peak below this => standing
    t_bike_walk: float = 0.78          # below this => biking
    t_walk_run: float = 2.10           # below this => walking, above => running

    # ---- sampling / windowing -------------------------------------------------
    fs: float = 100.0
    window_s: float = 2.5
    overlap: float = 0.5               # used for calibration windows
    max_gap_s: float = 0.3
    buffer_s: float = 10.0

    # ---- real-time ------------------------------------------------------------
    live_hop_s: float = 0.5
    vote_windows: int = 3              # majority vote over the last N windows
    quick_stand_s: float = 1.0         # last this-many seconds still => standing right away
    stale_s: float = 2.0

    # ---- transport (matches pico_main.py / pc_receiver.py) -------------------
    pico_ip: str = "10.247.137.46"
    pico_port: int = 5005
    local_port: int = 5005
    hello_period_s: float = 1.0
    accel_lsb_per_g: float = 8192.0
    gyro_lsb_per_dps: float = 65.5

    @property
    def window_n(self) -> int:
        return int(round(self.window_s * self.fs))

    @property
    def hop_n(self) -> int:
        return max(1, int(round(self.window_n * (1.0 - self.overlap))))

    @property
    def axis_name(self) -> str:
        return AXES[self.axis]

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "PeakConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def load_or_default(cls, model_dir=DEFAULT_MODEL_DIR) -> "PeakConfig":
        p = Path(model_dir) / "thresholds.json"
        return cls.load(p) if p.exists() else cls()
