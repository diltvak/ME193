"""Loading the recorded walks (recordings/<trial>/left.csv, right.csv, labels.csv)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as C

POINTS = ["start", "wp1", "wp2", "wp3", "wp4", "end"]


@dataclass
class Walk:
    name: str
    legs: dict = field(default_factory=dict)   # leg -> (t (n,), acc (n,3), gyro (n,3))
    labels: dict = field(default_factory=dict) # point -> t_s


def load_leg(path: Path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1, invalid_raise=False)
    if d.ndim != 2 or len(d) == 0:
        return np.empty(0), np.empty((0, 3)), np.empty((0, 3))
    d = d[np.isfinite(d[:, [0, 2, 3, 4, 5, 6, 7]]).all(axis=1)]
    order = np.argsort(d[:, 0], kind="stable")
    d = d[order]
    return d[:, 0], d[:, 2:5], d[:, 5:8]


def load_walk(folder: Path) -> Walk:
    folder = Path(folder)
    w = Walk(name=folder.name)
    for leg in C.LEGS:
        w.legs[leg] = load_leg(folder / f"{leg}.csv")
    lab = np.genfromtxt(folder / "labels.csv", delimiter=",", skip_header=1, dtype=None, encoding=None)
    for row in np.atleast_1d(lab):
        w.labels[str(row[1])] = float(row[0])
    return w


def training_folders():
    """Every recording that has a complete set of route labels."""
    out = []
    for f in sorted(C.RECORDINGS.glob("*")):
        if (f / "labels.csv").exists() and (f / "left.csv").exists():
            try:
                w = load_walk(f)
            except Exception:
                continue
            if all(p in w.labels for p in POINTS):
                out.append(f)
    return out
