"""Particle filter: where along the route is the walker, and how long are their steps?

Each particle is one hypothesis: (s = meters from the start, L = this walker's step length,
k = how many of the route's turns have been taken). It is the HMM from the plan with the
"progress along the route" state made continuous and the unknown step length added as a state,
which is what lets it adapt to a new person.

  steps      every tick moves each particle forward by L x steps (with noise). During a Wi-Fi gap the
             walker is assumed to keep roughly their recent cadence, with wide uncertainty.
  turns      a detected turn is either the next turn on the map (likely if the angle matches and the
             particle is near that turn) or clutter. Matching snaps the particle to the turn's position.
             Particles whose step length predicted the turn's position well gain weight, so after a turn
             or two the step length (and with it every distance) is calibrated to this person.
  waiting    a particle cannot run more than OVERSHOOT_M past a turn it has not taken. Either the turn
             is still ahead of the walker (they are shorter-stepped than assumed) or the turn was missed
             in a data gap (small chance per tick, large if a gap happened recently).
  off route  a turn that matches no particle's next turn is flagged as a wrong turn. Position is frozen
             until the walker turns back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass
class Guidance:
    state: str               # "walking" | "turn_now" | "arrived" | "off_route" | "passed_turn"
    text: str                # instruction for the walker
    s: float                 # estimated meters from the start
    s_sd: float              # uncertainty (m)
    step_len: float          # estimated step length (m)
    step_len_sd: float
    next_node: int           # index into route nodes of the next waypoint
    to_next_m: float
    to_next_steps: float
    to_end_m: float
    turns_taken: int
    p_arrived: float


def note_of(node):
    """Extra instruction for a waypoint from route.json, e.g. "note": "go down the stairs"."""
    return f" and {node['note']}" if node.get("note") else ""


def turn_word(deg):
    return "LEFT" if deg > 0 else "RIGHT"


class RouteTracker:
    def __init__(self, route: dict, step_len0: float, n=C.N_PARTICLES, seed=0):
        self.route = route
        self.nodes = route["nodes"]
        self.D = self.nodes[-1]["dist_m"]
        self.turn_nodes = [i for i, n_ in enumerate(self.nodes) if n_["turn_deg"] != 0]
        self.turn_d = np.array([self.nodes[i]["dist_m"] for i in self.turn_nodes] + [np.inf])
        self.turn_a = np.array([self.nodes[i]["turn_deg"] for i in self.turn_nodes] + [0.0])
        self.n_turns = len(self.turn_nodes)
        self.turn_d_prev = np.r_[0.0, self.turn_d[:-1]]
        self.rng = np.random.default_rng(seed)
        self.N = n
        self.s = self.rng.normal(0.0, 0.3, n)
        self.L = step_len0 * np.exp(self.rng.normal(0.0, C.STEP_LEN_PRIOR_SD, n))
        self.k = np.zeros(n, dtype=int)
        self.over = np.zeros(n)                       # meters walked while pinned past an untaken turn
        self.unseen = np.zeros(n, dtype=bool)         # the last turn was assumed taken without being detected
        self.w = np.full(n, 1.0 / n)
        self.recent_steps: list[tuple[float, float]] = []   # (t, steps) of ticks with data
        self.gap_until = -1.0
        self.gap_since_turn = False                  # a Wi-Fi gap since the last matched turn (could hide one)
        self.stopped_ticks = 0                       # consecutive ticks with no steps
        self.gap_cadence = None                      # steps per tick assumed during the current gap
        self.off_route = 0.0                         # net heading (deg) away from the route, 0 = on route
        self.off_turn_expected = None
        self.events: list[str] = []                  # human-readable log
        self.last_t = 0.0

    # ------------------------------------------------------------------------------------
    def tick(self, tk):
        """Advance with one fusion.Tick."""
        self.last_t = tk.t1
        if self.off_route:
            return                                   # position frozen while off the route
        if tk.steps is not None:
            self.recent_steps.append((tk.t1, tk.steps))
            self.stopped_ticks = self.stopped_ticks + 1 if tk.steps == 0 else 0
            ds = self.L * tk.steps * (1.0 + self.rng.normal(0.0, C.STEP_NOISE, self.N))
        else:
            if self.gap_cadence is None:             # cadence just before the gap, kept for all of it
                self.gap_cadence = self._cadence()
            self.gap_until = tk.t1 + 3.0
            self.gap_since_turn = True
            self.stopped_ticks = 0
            ds = self.L * self.gap_cadence * self.rng.uniform(*C.GAP_SPEED_RANGE, self.N)
        if tk.steps is not None:
            self.gap_cadence = None
            self.recent_steps = [x for x in self.recent_steps if x[0] > tk.t1 - C.CADENCE_WINDOW]
        self.s += np.maximum(ds, 0.0)
        self._pin_at_turns(tk.t1)
        self.s = np.minimum(self.s, self.D + 10.0)

    def _cadence(self):
        if not self.recent_steps:
            return 0.0
        return float(np.mean([s for _, s in self.recent_steps]))

    def _pin_at_turns(self, t):
        lim = self.turn_d[self.k] + C.OVERSHOOT_M
        over = self.s > lim
        if not over.any():
            self.over[:] = 0
            return
        # a gap since the last matched turn could have hidden this one
        p = C.MISS_TURN_P_GAP if self.gap_since_turn else C.MISS_TURN_P
        miss = over & (self.rng.random(self.N) < p)
        self.k[miss] += 1                            # assume the turn happened unseen
        self.over[miss] = 0
        self.unseen[miss] = True
        pin = over & ~miss
        self.over[pin] += self.s[pin] - lim[pin]
        self.s[pin] = lim[pin]
        self.over[~over] = 0

    # ------------------------------------------------------------------------------------
    def turn(self, ev):
        """Handle a fusion.Turn event. Returns a message if it looks like a wrong turn."""
        if self.off_route:
            self.off_route = (self.off_route + ev.angle + 180.0) % 360.0 - 180.0   # e.g. a U-turn counts
            if abs(self.off_route) < C.TURN_MIN_DEG:
                self.off_route = 0.0
                self.events.append(f"{ev.t:6.1f}s back on route")
                return "Back on the route."
            return None
        sd_a = C.TURN_ANGLE_SD * (1.5 if ev.gap else 1.0)
        out = []
        # skip 0: the next turn; 1: the one after (previous turn missed); -1: the turn this particle
        # assumed it took unseen during a gap, now actually seen
        for skip in (0, 1, -1):
            j = np.clip(self.k + skip, 0, self.n_turns)
            d, a = self.turn_d[j], self.turn_a[j]
            valid = (j < self.n_turns) & ((skip != -1) | self.unseen)
            seg = np.where(valid, d - self.turn_d_prev[j], 0)     # distance walked since the previous turn
            pos_sd = C.TURN_POS_SD_MIN + C.TURN_POS_SD_REL * seg
            la = np.exp(-0.5 * ((ev.angle - a) / sd_a) ** 2) / (sd_a * np.sqrt(2 * np.pi))
            ls = np.exp(-0.5 * ((self.s - np.where(valid, d, 0)) / pos_sd) ** 2) / (pos_sd * np.sqrt(2 * np.pi))
            # once pinned past the turn, position no longer tells us anything: the turn is overdue
            ls = np.where(self.over > 0, np.maximum(ls, 1.0 / (pos_sd * np.sqrt(2 * np.pi)) * 0.6), ls)
            prior = {0: 1.0, 1: 0.5 if self.gap_since_turn else 0.05, -1: 1.0}[skip]
            out.append(np.where(valid, la * ls * prior, 0.0))
        lc = np.full(self.N, C.TURN_CLUTTER / (360.0 * max(self.D, 1.0)))
        lik = np.stack([out[0], out[1], out[2], lc])   # (4, N)
        tot = lik.sum(axis=0)
        match = float(self.w @ (out[0] + out[1] + out[2]))
        clutter = float(self.w @ lc)
        if match / (match + clutter) < C.WRONG_TURN_P and abs(ev.angle) >= 60:
            self.off_route = ev.angle
            nxt = self._next_turn_desc()
            self.events.append(f"{ev.t:6.1f}s wrong turn {turn_word(ev.angle)} {abs(ev.angle):.0f} deg")
            return (f"That {turn_word(ev.angle).lower()} turn is not on the route. "
                    f"Turn back ({turn_word(-ev.angle).lower()}) and continue{nxt}.")
        # per particle: choose a hypothesis in proportion to its likelihood
        u = self.rng.random(self.N) * tot
        c = np.cumsum(lik, axis=0)
        h = (u[None, :] >= c).sum(axis=0)             # 0, 1, 2 (= confirm) or 3 (= clutter)
        since = sum(st for t, st in self.recent_steps if t > ev.t)   # steps taken after the turn ended
        for hyp, skip in ((0, 0), (1, 1), (2, -1)):
            m = h == hyp
            j = self.k[m] + skip
            self.s[m] = self.turn_d[j] + self.L[m] * since + self.rng.normal(0, 0.5, m.sum())
            self.k[m] = j + 1
            self.over[m] = 0
            self.unseen[m] = False
        self.w *= tot
        self.w /= self.w.sum()
        self._resample()
        if match / (match + clutter) > 0.5:
            self.gap_since_turn = False
            self.events.append(f"{ev.t:6.1f}s turn {turn_word(ev.angle)} {abs(ev.angle):.0f} deg matched")
        return None

    def _next_turn_desc(self):
        k = int(np.round(np.average(self.k, weights=self.w)))
        if k >= self.n_turns:
            return " to the end"
        return f" to the {turn_word(self.turn_a[k]).lower()} turn at {self.nodes[self.turn_nodes[k]]['name']}"

    def _resample(self):
        ess = 1.0 / np.sum(self.w ** 2)
        if ess > self.N / 2:
            return
        pos = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), self.N - 1)
        self.s, self.L, self.k, self.over = self.s[idx], self.L[idx], self.k[idx], self.over[idx]
        self.unseen = self.unseen[idx]
        self.L *= np.exp(self.rng.normal(0, 0.01, self.N))      # keep some diversity in step length
        self.w = np.full(self.N, 1.0 / self.N)

    # ------------------------------------------------------------------------------------
    def guidance(self) -> Guidance:
        w = self.w
        k = int(np.argmax(np.bincount(self.k, weights=w, minlength=self.n_turns + 1)))
        # summarise only the particles that agree on how many turns were taken, so the position is
        # never a blend of "before the turn" and "after the turn"
        sel = self.k == k
        ws = w[sel] / w[sel].sum()
        s = float(ws @ self.s[sel])
        s_sd = float(np.sqrt(ws @ (self.s[sel] - s) ** 2))
        L = float(np.average(self.L, weights=w))
        L_sd = float(np.sqrt(np.average((self.L - L) ** 2, weights=w)))
        done = (self.k >= self.n_turns) & (self.s >= self.D - C.ARRIVE_M)
        if self.stopped_ticks * C.TICK >= C.ARRIVE_STOP_S:       # stopped close to the end: arrived
            done |= (self.k >= self.n_turns) & (self.s >= self.D - C.ARRIVE_STOP_M)
        p_arr = float(w @ done)
        to_end = max(self.D - s, 0.0)
        if k < self.n_turns:
            ni = self.turn_nodes[k]
            to_next = self.turn_d[k] - s
        else:
            ni = len(self.nodes) - 1
            to_next = to_end
        node = self.nodes[ni]
        steps = max(to_next, 0.0) / L
        pinned = float(w @ (self.over > 0))
        overdue = float(np.average(self.over, weights=w))

        if self.off_route:
            state, text = "off_route", (f"Off route. Turn back {turn_word(-self.off_route).lower()} "
                                        f"and continue{self._next_turn_desc()}.")
        elif p_arr > 0.6:
            state, text = "arrived", "You have arrived at your destination."
        elif k < self.n_turns and pinned > 0.6 and overdue > C.OVERSHOOT_M:
            state = "passed_turn"
            text = (f"You may have passed the {turn_word(node['turn_deg']).lower()} turn at {node['name']}. "
                    f"Turn {turn_word(node['turn_deg'])} now, or go back about "
                    f"{overdue + C.OVERSHOOT_M:.0f} m if there is no way through.")
        elif k < self.n_turns and to_next < max(2.0, 1.5 * L):
            state, text = "turn_now", f"Turn {turn_word(node['turn_deg'])} now ({node['name']}){note_of(node)}."
        elif k < self.n_turns:
            state = "walking"
            text = (f"Walk straight about {max(to_next, 0):.0f} m (~{steps:.0f} steps), "
                    f"then turn {turn_word(node['turn_deg'])} at {node['name']}{note_of(node)}.")
        else:
            state = "walking"
            text = f"Walk straight about {to_end:.0f} m (~{to_end / L:.0f} steps) to the destination."
        return Guidance(state, text, s, s_sd, L, L_sd, ni, to_next, steps, to_end, k, p_arr)
