"""Uncertainty bands for the live display, taken from the evaluate.py results.

Distance: the leave-one-trial-out relative error (estimate - truth) / truth of the net's distance was
mean -0.7%, std 7.9%, 95% prediction interval about -19% to +18% for a ~19.5 m walk. Inverting that gives
the plausible true distance for any estimate. Errors are mostly systematic (a whole walk's stride length is
off), so the interval scales with distance. It was measured on ~19.5 m walks; treat it as approximate elsewhere.

Steps: the worst held-out error on the hand-counted walks was 2 steps (the missed stopping step), mean -0.67.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_RESULTS_DIR

# fallbacks = values measured in results/results.json (used if that file is missing)
_DEFAULT = {"std_pct": 7.87, "bias_pct": -0.68, "pi95_pct": (-19.35, 17.99), "step_max_err": 2, "step_bias": -0.67}


def load(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict:
    u = dict(_DEFAULT)
    try:
        r = json.loads((Path(results_dir) / "results.json").read_text())
        nn = r["loto_summary"]["nn"]
        u.update(std_pct=nn["std_pct"], bias_pct=nn["bias_pct"], pi95_pct=tuple(nn["pred_interval95_pct"]))
        errs = [abs(h["error"]) for h in r["heldout"]]
        u.update(step_max_err=max(errs), step_bias=sum(h["error"] for h in r["heldout"]) / len(errs))
    except (OSError, KeyError, ValueError):
        pass
    return u


def distance_interval(est_m: float, u: dict, which="pi95_pct"):
    """(low, high) plausible true distance. which: 'pi95_pct' (95%) or 'std' (+/-1 sigma, about 68%)."""
    if which == "std":
        lo, hi = -u["std_pct"] + u["bias_pct"], u["std_pct"] + u["bias_pct"]
    else:
        lo, hi = u[which]
    # error e = (est - true)/true  ->  true = est / (1 + e); large e gives the low end
    return est_m / (1 + hi / 100), est_m / (1 + lo / 100)
