"""How well does the navigator track the route? Leave-one-walk-out, plus simulated "other people".

    python3 -m navigator.evaluate                 # -> navigator/results/results.json + summary printout
    python3 -m navigator.evaluate --plot          # also navigator/results/<condition>_<walk>.png

Each walk is held out in turn. The route map is built from the other walks, then the held-out walk is
replayed through the navigator exactly as it would run live (tick by tick, with latency).
Ground truth: at each waypoint button press, the walker was at that waypoint's mapped distance.

With data from only one person, a different walker is simulated by changing what differs between people:
  stride-20 / stride+20   the height-based step length is 20% too short / too long
                          (= a walker whose steps are 20% longer / shorter than their height predicts)
  other-person            random sensor rotation on each leg, legs swapped, walking 15-20% faster or slower
                          (time-warped), 10% extra packet loss plus a 6 s Wi-Fi dead zone, and a random
                          +/-15% step-length error. Repeated with 3 random seeds.
These test robustness to those factors only. Real people also differ in gait shape, which this cannot test.
"""
from __future__ import annotations

import argparse
import json
import zlib

import numpy as np

from . import config as C
from .build_route import build
from .core import Navigator
from .data import POINTS, load_walk, training_folders


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def augment(walk, rng, rotate=False, swap=False, warp=1.0, loss=0.0, dead_zone=0.0):
    """Return (legs dict, labels dict) with the requested changes applied."""
    legs, labels = {}, {k: v * warp for k, v in walk.labels.items()}
    for leg in C.LEGS:
        t, a, g = walk.legs[leg]
        t, a, g = t * warp, a.copy(), g / warp
        if rotate:
            R = random_rotation(rng)
            a, g = a @ R.T, g @ R.T
        keep = np.ones(len(t), bool)
        if loss > 0:                                   # drop whole 5-sample packets
            pk = np.arange(len(t)) // 5
            keep &= rng.random(pk.max() + 1)[pk] >= loss
        name = {"left": "right", "right": "left"}[leg] if swap else leg
        legs[name] = (t[keep], a[keep], g[keep])
    if dead_zone > 0:                                  # both legs lose data at the same moment
        lo = rng.uniform(labels["start"] + 10, labels["end"] - 10 - dead_zone)
        for leg in legs:
            t, a, g = legs[leg]
            k = (t < lo) | (t > lo + dead_zone)
            legs[leg] = (t[k], a[k], g[k])
    return legs, labels


def run_walk(legs, labels, route, user_height, seed=0):
    nav = Navigator(route, user_height, seed=seed)
    for leg in C.LEGS:
        nav.feed(leg, *legs[leg])
    t0 = labels["start"]
    t_end = max(float(legs[l][0].max()) for l in C.LEGS if len(legs[l][0]))
    nav.start(t0)
    trace = []
    now = t0
    while now < t_end + C.LATENCY:
        now += C.TICK
        g = nav.update(now)
        trace.append((now - C.LATENCY, g))
    return nav, trace


def score(nav, trace, labels, route):
    tt = np.array([t for t, _ in trace])
    s = np.array([g.s for _, g in trace])
    sd = np.array([g.s_sd for _, g in trace])
    d = {n["name"]: n.get("press_m", n["dist_m"]) for n in route["nodes"]}
    wp_err = {}
    for p in POINTS[1:]:
        i = int(np.clip(np.searchsorted(tt, labels[p]), 0, len(tt) - 1))
        wp_err[p] = float(s[i] - d[p])
    arrived = [t for t, g in trace if g.state == "arrived"]
    arr_err = (arrived[0] - labels["end"]) if arrived else None
    states = [g.state for _, g in trace]
    g_last = trace[-1][1]
    return {
        "wp_err_m": wp_err,
        "arrive_err_s": arr_err,
        "turns_taken": g_last.turns_taken,
        "n_turns": nav.tracker.n_turns,
        "false_wrong_turn": sum(1 for _, m in nav.messages if "not on the route" in m),
        "passed_turn_alerts": int(sum(1 for a, b in zip(states, states[1:]) if b == "passed_turn" and a != b)),
        "step_len_est": g_last.step_len,
        "mean_sd_m": float(np.mean(sd)),
    }


def plot(trace, labels, route, nav, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tt = np.array([t for t, _ in trace])
    s = np.array([g.s for _, g in trace])
    sd = np.array([g.s_sd for _, g in trace])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(tt, s - 2 * sd, s + 2 * sd, alpha=0.25, label="estimate +/- 2 sd")
    ax.plot(tt, s, label="estimated distance along route")
    d = {n["name"]: n.get("press_m", n["dist_m"]) for n in route["nodes"]}
    ax.plot([labels[p] for p in POINTS], [d[p] for p in POINTS], "ko", label="true (waypoint presses)")
    for ev in nav.fusion.turns:
        ax.axvline(ev.t, color="C3" if ev.angle < 0 else "C2", lw=0.8, alpha=0.6)
    for tk in nav.fusion.ticks:
        if tk.steps is None:
            ax.axvspan(tk.t0, tk.t1, color="0.85", lw=0)
    ax.set_xlabel("time (s)  [grey = no data, red/green line = right/left turn detected]")
    ax.set_ylabel("m from start")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


CONDITIONS = {
    "same-person": dict(height=1.0, seeds=[0]),
    "stride-20": dict(height=0.8, seeds=[0]),
    "stride+20": dict(height=1.2, seeds=[0]),
    "other-person": dict(height=None, seeds=[1, 2, 3], rotate=True, swap=True, loss=0.10, dead_zone=6.0),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--trainer-height", type=float, default=C.DEFAULT_TRAINER_HEIGHT)
    ap.add_argument("--only", choices=list(CONDITIONS))
    args = ap.parse_args()
    C.RESULTS.mkdir(exist_ok=True)
    folders = training_folders()
    walks = [load_walk(f) for f in folders]
    routes = {}
    for i, w in enumerate(walks):
        routes[w.name] = build([f for j, f in enumerate(folders) if j != i], args.trainer_height, verbose=False)

    results = {}
    for cname, cond in CONDITIONS.items():
        if args.only and cname != args.only:
            continue
        rows = []
        for w in walks:
            route = routes[w.name]
            for seed in cond["seeds"]:
                rng = np.random.default_rng(seed * 1000 + zlib.crc32(w.name.encode()) % 1000)
                if cname == "other-person":
                    warp = rng.choice([0.82, 0.85, 1.15, 1.2])
                    legs, labels = augment(w, rng, rotate=True, swap=True, warp=warp,
                                           loss=cond["loss"], dead_zone=cond["dead_zone"])
                    hf = float(np.exp(rng.normal(0, 0.15)))
                else:
                    legs, labels = augment(w, rng)
                    hf, warp = cond["height"], 1.0
                nav, trace = run_walk(legs, labels, route, args.trainer_height * hf, seed=seed)
                r = score(nav, trace, labels, route)
                r.update(walk=w.name, seed=seed, height_factor=hf, warp=float(warp))
                rows.append(r)
                if args.plot and seed == cond["seeds"][0]:
                    plot(trace, labels, route, nav, f"{cname}: {w.name}", C.RESULTS / f"{cname}_{w.name}.png")
        results[cname] = rows
        summarize(cname, rows)
    with open(C.RESULTS / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved {C.RESULTS / 'results.json'}")


def summarize(cname, rows):
    print(f"\n=== {cname}  ({len(rows)} runs)")
    print("  |position error| at each waypoint press (m): median [90th pct]")
    for p in POINTS[1:]:
        e = np.abs([r["wp_err_m"][p] for r in rows])
        print(f"    {p:<5} {np.median(e):5.1f}  [{np.percentile(e, 90):5.1f}]")
    arr = [r["arrive_err_s"] for r in rows]
    ok = [a for a in arr if a is not None]
    print(f"  arrival announced: {len(ok)}/{len(rows)} runs, "
          + (f"median {np.median(ok):+.1f} s vs. the End press (range {min(ok):+.1f} .. {max(ok):+.1f})" if ok else ""))
    print(f"  all turns taken at the end: {sum(r['turns_taken'] == r['n_turns'] for r in rows)}/{len(rows)}")
    print(f"  false 'wrong turn' alerts: {sum(r['false_wrong_turn'] for r in rows)}   "
          f"'passed turn' alerts: {sum(r['passed_turn_alerts'] for r in rows)}")


if __name__ == "__main__":
    main()
