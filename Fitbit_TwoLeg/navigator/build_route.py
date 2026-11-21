"""Build the route map (route.json) from the labelled training walks.

    python3 -m navigator.build_route                          # all walks in recordings/
    python3 -m navigator.build_route --trainer-height 1.80     # your height, for the meter conversion

For each walk: count steps (both legs, Wi-Fi gaps filled at the walking cadence around them), find the
cumulative step count at every waypoint press, and find the turn made at each waypoint. The map stores,
per waypoint, the median distance from the start and the median turn angle. Distances are converted
from the trainer's steps to meters with the trainer's step length (0.415 x height), so another person
is guided by meters and their own step length, never by the trainer's step count.

If you measure the route with a tape measure, edit "dist_m" in route.json; nothing else needs changing.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import config as C
from .data import POINTS, load_walk, training_folders
from .fusion import Fusion


def process_walk(walk, start_t=0.0):
    fu = Fusion()
    for leg in C.LEGS:
        fu.feed(leg, *walk.legs[leg])
    end = max(float(walk.legs[l][0].max()) for l in C.LEGS if len(walk.legs[l][0])) + 1.0
    fu.start(start_t)
    fu.update(end + C.LATENCY)
    return fu


def filled_steps(ticks):
    """Steps per tick, with gap ticks filled by the local cadence (walking on both sides of the gap)."""
    s = np.array([np.nan if t.steps is None else t.steps for t in ticks])
    out = s.copy()
    w = int(C.CADENCE_WINDOW / C.TICK)
    gap = np.isnan(s)
    i = 0
    while i < len(s):
        if not gap[i]:
            i += 1
            continue
        j = i
        while j < len(s) and gap[j]:
            j += 1
        before = s[max(0, i - w):i]
        after = s[j:j + w]
        both = np.r_[before[~np.isnan(before)], after[~np.isnan(after)]]
        rate = float(np.mean(both)) if len(both) else 0.0
        # only coast if walking on both sides; a stop next to the gap means probably standing
        walking = len(before) and len(after) and np.nanmean(before) > 0.2 and np.nanmean(after) > 0.2
        out[i:j] = rate if walking else 0.0
        i = j
    return out


def walk_summary(walk):
    fu = process_walk(walk)
    te = np.array([t.t1 for t in fu.ticks])
    cum = np.r_[0.0, np.cumsum(filled_steps(fu.ticks))]
    tt = np.r_[fu.ticks[0].t0, te]
    at = lambda t: float(np.interp(t, tt, cum))
    s0 = at(walk.labels["start"])
    dist = {p: at(walk.labels[p]) - s0 for p in POINTS}
    # turn at waypoint p: a turn starting after p was pressed (minus a little) and before the next press
    turns, observed = {}, {}
    for k, p in enumerate(POINTS[:-1]):
        lo = walk.labels[p] + C.TURN_ASSOC_S[0]
        hi = walk.labels[POINTS[k + 1]] + C.TURN_ASSOC_S[0]
        # did we have heading data around this waypoint? (a Wi-Fi gap can hide the turn)
        near = [t for t in fu.ticks if lo <= t.t0 < min(hi, walk.labels[p] + C.TURN_ASSOC_S[1])]
        observed[p] = bool(near) and sum(t.dhead is not None for t in near) / len(near) >= 0.7
        cand = [t for t in fu.turns if lo <= t.t_start + 0.5 * (t.t - t.t_start) < hi]
        if cand:
            big = max(cand, key=lambda t: abs(t.angle))
            turns[p] = {"angle": big.angle, "steps": at(0.5 * (big.t_start + big.t)) - s0}
            observed[p] = True
    gaps = sum(C.TICK for t in fu.ticks if t.steps is None)
    return {"name": walk.name, "dist": dist, "turns": turns, "observed": observed, "gap_s": gaps,
            "dur": walk.labels["end"] - walk.labels["start"]}


def build(folders, trainer_height=C.DEFAULT_TRAINER_HEIGHT, verbose=True):
    sums = [walk_summary(load_walk(f)) for f in folders]
    step_len = C.STEP_LENGTH_PER_HEIGHT * trainer_height
    nodes = []
    for p in POINTS:
        d = np.array([s["dist"][p] for s in sums])
        angs = np.array([s["turns"][p]["angle"] for s in sums if p in s["turns"]])
        n_obs = sum(s["observed"].get(p, False) for s in sums)
        turn = 0.0
        turn_d = float(np.median(d))
        # keep a turn seen in at least half of the walks that had data there, all in the same direction
        # (a Wi-Fi dead zone can hide it in the others)
        if len(angs) >= 1 and len(angs) >= 0.5 * n_obs and (np.all(angs > 0) or np.all(angs < 0)):
            turn = float(np.median(angs))
            # turns are where you turn, which is a step or two past the button press
            tsteps = np.array([s["turns"][p]["steps"] for s in sums if p in s["turns"]])
            turn_d = float(np.median(tsteps))
        nodes.append({
            "name": p,
            "dist_steps": round(float(np.median(d)), 1),
            "dist_steps_sd": round(float(np.std(d)), 1),
            "dist_m": round(turn_d * step_len, 2),            # where the turn is made (or the press)
            "press_m": round(float(np.median(d)) * step_len, 2),  # where the waypoint button was pressed
            "turn_deg": round(turn, 0),
            "turn_seen": f"{len(angs)}/{n_obs}",
        })
    route = {
        "trainer_height_m": trainer_height,
        "trainer_step_m": round(step_len, 3),
        "walks": [s["name"] for s in sums],
        "nodes": nodes,
        "note": "dist_m is the distance from the start. Replace it with tape-measured values if you have them.",
    }
    if verbose:
        print(f"{len(sums)} walks, trainer step length {step_len:.2f} m")
        for s in sums:
            print(f"  {s['name']:<28} gap {s['gap_s']:4.1f}s  steps at points "
                  + " ".join(f"{s['dist'][p]:5.0f}" for p in POINTS)
                  + "  turns " + " ".join(f"{p}:{v['angle']:+.0f}" for p, v in s["turns"].items()))
        print("Route:")
        for n in nodes:
            turn = "straight" if n["turn_deg"] == 0 else ("LEFT" if n["turn_deg"] > 0 else "RIGHT") + f" {abs(n['turn_deg']):.0f} deg"
            print(f"  {n['name']:<5} {n['dist_m']:6.1f} m  ({n['dist_steps']:.0f} +/- {n['dist_steps_sd']:.0f} steps)  "
                  f"then {turn}  (turn seen in {n['turn_seen']} walks with data there)")
    return route


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainer-height", type=float, default=C.DEFAULT_TRAINER_HEIGHT,
                    help="height (m) of the person who recorded the training walks")
    ap.add_argument("--out", default=str(C.ROUTE_FILE))
    args = ap.parse_args()
    folders = training_folders()
    if not folders:
        raise SystemExit(f"No labelled walks in {C.RECORDINGS}")
    route = build(folders, args.trainer_height)
    try:                                       # keep hand-added notes (e.g. "go down the stairs")
        with open(args.out) as f:
            old = {n["name"]: n for n in json.load(f)["nodes"]}
        for n in route["nodes"]:
            if old.get(n["name"], {}).get("note"):
                n["note"] = old[n["name"]]["note"]
    except (OSError, ValueError, KeyError):
        pass
    with open(args.out, "w") as f:
        json.dump(route, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
