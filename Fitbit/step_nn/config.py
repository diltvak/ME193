"""Every tunable for the neural-net step counter / distance estimator.

Field names for sampling and transport match activity.config.Config, so the receiver in
activity.data_io (UdpSource, ReplaySource, resample_uniform) works with this class unchanged.
`train.py` saves the Config it used next to the weights; `live.py` loads it so inference always
matches training.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT.parent             # CSVs live next to pico_main.py
DEFAULT_MODEL_DIR = ROOT / "models"
DEFAULT_RESULTS_DIR = ROOT / "results"

FEET_TO_M = 0.3048
TRIAL_DISTANCE_M = 64 * FEET_TO_M          # every trial_<n>.csv covers ~64 ft
N_TRIALS = 10

# Recordings where you counted the steps yourself (both legs). Used only as ground truth.
TRUE_STEPS = {"33_steps": 33, "29_steps": 29, "23_steps": 23}

# Recordings with no walking in them: used as negative examples (the net must output "no step").
NEGATIVE_FILES = ["standing_data_1", "cycling_data_1"]
# Extra walking with no known distance or count: pseudo-labeled and used for step detection only.
EXTRA_WALK_FILES = ["walking_data_1"]


@dataclass
class StepConfig:
    # ---- sampling -------------------------------------------------------------
    fs: float = 100.0              # Hz, uniform rate everything is resampled to
    max_gap_s: float = 0.3         # (used by activity.data_io.resample_uniform)
    gap_flag_s: float = 0.1        # raw sample spacing above this counts as a dropped-packet gap
    buffer_s: float = 8.0          # rolling raw buffer for live inference

    # ---- trimming (data processing allowed by the assignment) -----------------
    trim_win_s: float = 0.5        # rolling window for "is the leg moving?"
    trim_gyro_std_dps: float = 40.0
    trim_pad_s: float = 0.75       # keep this much quiet data around the walking bout

    # ---- pseudo-labelling of strides (training only; the net is what runs live) -
    label_lp_hz: float = 6.0       # low-pass on gz before peak picking
    label_prom_frac: float = 0.5   # peak prominence >= this x 95th percentile of gz
    label_min_prom_dps: float = 120.0   # absolute floor so standing noise never yields peaks
    label_merge_frac: float = 0.55 # peaks closer than this x median period are one stride
    label_fill_frac: float = 1.35  # a gap-hidden interval longer than this x median hides strides

    # ---- network --------------------------------------------------------------
    hidden: int = 24
    kernel: int = 5
    dilations: tuple = (1, 2, 4, 8, 16)   # receptive field = +/-0.62 s at 100 Hz
    dropout: float = 0.1
    len_center_m: float = 1.3      # stride-length head predicts an offset from this
    len_scale_m: float = 0.5

    # ---- training -------------------------------------------------------------
    crop_s: float = 6.0
    batch: int = 32
    iters: int = 700
    lr: float = 3e-3
    weight_decay: float = 1e-3
    peak_sigma_s: float = 0.05     # width of the Gaussian "step happened here" target
    len_loss_weight: float = 1.0
    negative_share: float = 0.25   # fraction of training crops taken from non-walking files
    warp_range: tuple = (0.85, 1.2)   # time-stretch augmentation (changes cadence)
    warp_prob: float = 0.6
    amp_range: tuple = (0.95, 1.05)
    noise_acc_g: float = 0.03
    noise_gyro_dps: float = 3.0
    drop_prob: float = 0.5         # simulate a dropped-packet gap in a crop
    ridge_alpha: float = 1.0

    # ---- inference ------------------------------------------------------------
    hop_s: float = 0.25            # run the net this often live
    infer_window_s: float = 6.0
    lookahead_s: float = 0.65      # a peak is final once this much later data exists (= receptive field)
    peak_thresh: float = 0.5
    maybe_thresh: float = 0.2      # sub-threshold peaks above this are shown as "possible missed stride"
    weak_conf: float = 0.7         # counted strides below this peak probability are flagged as low confidence
    refractory_s: float = 0.4      # two strides of one leg are never closer than this
    gap_reset_s: float = 0.6       # a stream gap longer than this restarts the buffer
    stale_s: float = 2.0
    default_period_s: float = 1.1  # stride period assumed for the very first stride
    first_step_on_sensor_leg: bool = True   # steps = 2*strides - 1 (see README); False -> 2*strides
    len_source: str = "nn"         # "nn" or "ridge"; train.py/evaluate.py pick the one with lower CV error

    # ---- transport (matches pico_main.py / pc_receiver.py) -------------------
    pico_ip: str = "10.0.0.251"
    #pico_ip: str = "10.247.137.46"
    pico_port: int = 5005
    local_port: int = 5005
    hello_period_s: float = 1.0
    accel_lsb_per_g: float = 8192.0    # +/-4 g
    gyro_lsb_per_dps: float = 65.5     # +/-500 dps

    # ---- derived --------------------------------------------------------------
    @property
    def crop_n(self) -> int:
        return int(round(self.crop_s * self.fs))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "StepConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        out = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in known}
        return cls(**out)

    @classmethod
    def load_or_default(cls, model_dir=DEFAULT_MODEL_DIR) -> "StepConfig":
        p = Path(model_dir) / "config.json"
        return cls.load(p) if p.exists() else cls()
