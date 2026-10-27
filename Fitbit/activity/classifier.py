"""Hybrid classifier: rule-based "standing" detector + random forest (or rule tree) for moving."""
from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import joblib

from .config import Config, DEFAULT_MODEL_DIR
from .features import extract_features, to_vector


def is_standing(feats: dict, cfg: Config) -> bool:
    """Stage 1: low variance of both |acc| and |gyro| => standing still."""
    return feats["a_std"] < cfg.stand_acc_std_g and feats["g_std"] < cfg.stand_gyro_std_dps


def rule_based_moving(feats: dict, cfg: Config) -> str:
    """Fallback decision tree for moving windows (no training needed; tune thresholds in Config).

    1. Violent impacts / large acceleration variance  -> running
    2. Gyro dominated by one smooth fundamental      -> biking (near-sinusoidal rotation)
    3. Otherwise                                     -> walking
    """
    if feats["a_std"] > cfg.tree_run_acc_std_g or feats["a_peak"] > cfg.tree_run_peak_g:
        return "running"
    if feats["g1_fund_frac"] > cfg.tree_bike_gyro_fund_frac:
        return "biking"
    return "walking"


class HybridClassifier:
    """Stage 1 rule -> stage 2 random forest. Falls back to the rule tree when model is None."""

    def __init__(self, cfg: Config, model=None, feature_names=None):
        self.cfg, self.model, self.feature_names = cfg, model, feature_names

    @property
    def uses_model(self) -> bool:
        return self.model is not None

    def predict_features(self, feats: dict) -> tuple:
        """(label, confidence). Confidence is the forest's vote share (1.0 for rules)."""
        if is_standing(feats, self.cfg):
            return "standing", 1.0
        if self.model is None:
            return rule_based_moving(feats, self.cfg), 1.0
        x = to_vector(feats, self.feature_names).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        k = int(proba.argmax())
        return str(self.model.classes_[k]), float(proba[k])

    def predict_window(self, win) -> tuple:
        """(label, confidence, feature dict) for one (window_n, 6) array."""
        feats = extract_features(win, self.cfg)
        label, conf = self.predict_features(feats)
        return label, conf, feats

    @classmethod
    def load(cls, model_dir=DEFAULT_MODEL_DIR, force_rules=False):
        """Load config + model from model_dir; use the rule tree if no model is there."""
        model_dir = Path(model_dir)
        cfg_path, model_path = model_dir / "config.json", model_dir / "model.joblib"
        cfg = Config.load(cfg_path) if cfg_path.exists() else Config()
        if force_rules or not model_path.exists():
            if not force_rules:
                print(f"[classifier] no model at {model_path}; using rule-based fallback tree")
            return cls(cfg)
        blob = joblib.load(model_path)
        return cls(cfg, blob["model"], blob["feature_names"])


class MajorityVote:
    """Smooth predictions: most common label over the last n windows (ties -> most recent)."""

    def __init__(self, n: int):
        self.buf = deque(maxlen=n)

    def update(self, label: str) -> str:
        self.buf.append(label)
        return self.current()

    def current(self):
        """Majority label of the history without adding to it (None if empty)."""
        if not self.buf:
            return None
        counts = Counter(self.buf)
        best = max(counts.values())
        for lab in reversed(self.buf):
            if counts[lab] == best:
                return lab

    def reset(self, label=None):
        """Clear history; with a label, fill it so the smoothed output switches immediately."""
        self.buf.clear()
        if label is not None:
            self.buf.extend([label] * self.buf.maxlen)
