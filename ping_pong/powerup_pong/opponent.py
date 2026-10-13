"""The CPU opponent: an animated cartoon character that reacts to the game.

The figure is a small skeleton (head, torso, two-bone arms and legs) posed each frame from the
current mood and drawn with thick rounded limbs. It reacts to what happens in the game, which
Game reports as events (see Game.emit):

  - it shuffles its feet as it tracks the ball, and its eyes follow the ball (or your paddle),
  - it swings its paddle whenever it sends the ball back,
  - it cheers when it scores, holds its head when you do, and flinches at your smashes,
  - it shivers when an ice storm hits and cowers behind a brick wall (each powerup names its own
    reaction and line, see Powerup.opponent_reaction / opponent_line),
  - it sweats in long rallies, and trash-talks in speech bubbles.

Poses are in local units (about 1.2 = full height) with the feet at the origin, x right, y up.
"""

import math
import random

import cv2
import numpy as np

from powerup_pong.config import WHITE
from powerup_pong.geometry import to_px

Z_OPP = 5.3               # distance of the opponent from the player, just behind the table
SCALE = 1.15              # size of the figure, in units of panel-height / z
SWING_S = 0.32            # duration of a paddle swing
MOOD_S = {"cheer": 1.6, "dismay": 1.6, "flinch": 0.55}
SPEAK_GAP_S = 2.2         # minimum time between unprompted speech bubbles

SKIN, HAIR, SHIRT, SHORTS, SHOE = (150, 190, 235), (35, 40, 60), (60, 60, 175), (80, 60, 45), (235, 235, 235)
PADDLE, PADDLE_RIM, BAND = (40, 40, 210), (20, 20, 70), (255, 255, 255)
ICE = (255, 225, 170)

LINES = dict(
    scored=("Ha! Too slow!", "Point CPU!", "Can't catch me!"),
    lost_point=("Argh!", "Lucky shot!", "Hmpf..."),
    smash=("Whoa!", "Ouch!", "Easy there!"),
    edge=("Ha, edge!", "Nice miss-hit!"),
    won=("GG! I win!", "Too easy!"),
    lost=("GG...", "Rematch?"),
)


def lerp(a, b, t):
    return a + (b - a) * t


def mix(color, other, t):
    return tuple(int(lerp(c, o, t)) for c, o in zip(color, other))


def ik(root, target, l1, l2, prefer):
    """Two-bone IK: the joint (elbow/knee) between root and target, bending toward `prefer`.
    Returns (joint, reachable end point)."""
    dx, dy = target[0] - root[0], target[1] - root[1]
    d = min(max(math.hypot(dx, dy), abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    base = math.atan2(dy, dx)
    bend = math.acos(max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
    joints = [(root[0] + l1 * math.cos(base + s * bend), root[1] + l1 * math.sin(base + s * bend)) for s in (1, -1)]
    joint = max(joints, key=lambda j: (j[0] - root[0]) * prefer[0] + (j[1] - root[1]) * prefer[1])
    return joint, (root[0] + d * math.cos(base), root[1] + d * math.sin(base))


class Opponent:
    def __init__(self):
        self.until = {}                  # temporary mood -> time it ends
        self.over = None                 # "cheer" or "dismay" once the match is decided
        self.swing_t = -math.inf
        self.freeze = 0.0                # 0-1, eased toward 1 while an ice storm is on
        self.step = 0.0                  # walk-cycle phase
        self.moving = 0.0                # smoothed speed of the shuffle
        self.prev_x = None
        self.gaze = (0.0, 0.0)           # smoothed look direction
        self.bubble = None               # (text, end time)
        self.last_speak = -math.inf
        self.rally = 0
        self.reaction = None             # persistent reaction from an active powerup

    # ---- Reacting to the game ----

    def say(self, text, now, force=False, hold=1.7):
        if force or now - self.last_speak >= SPEAK_GAP_S:
            self.bubble, self.last_speak = (text, now + hold), now

    def _mood_for(self, mood, now):
        self.until[mood] = now + MOOD_S[mood]

    def _on_event(self, kind, arg, now):
        if kind == "launch":
            self.swing_t = now
        elif kind == "player_return":
            self.rally += 1
            if arg == "smash":
                self._mood_for("flinch", now)
                self.say(random.choice(LINES["smash"]), now)
            elif arg == "edge" and random.random() < 0.5:
                self.say(random.choice(LINES["edge"]), now)
        elif kind == "opp_scored":
            self.rally = 0
            self._mood_for("cheer", now)
            self.say(random.choice(LINES["scored"]), now, force=True)
        elif kind == "opp_missed":
            self.rally = 0
            self._mood_for("dismay", now)
            self.say(random.choice(LINES["lost_point"]), now, force=True)
        elif kind == "powerup":
            if arg.opponent_line:
                self.say(arg.opponent_line, now, force=True, hold=2.2)
            if arg.opponent_reaction == "shocked":
                self._mood_for("flinch", now)
        elif kind == "over":
            self.over = "cheer" if arg == "opp" else "dismay"
            self.say(random.choice(LINES["won" if arg == "opp" else "lost"]), now, force=True, hold=4.0)

    def update(self, game, now):
        for kind, arg in game.drain_events():
            self._on_event(kind, arg, now)
        if game.phase == "serve" and self.over:    # a new match is starting
            self.over = None
        self.reaction = game.powerups.opponent_reaction()
        self.freeze += 0.1 * ((1.0 if self.reaction == "frozen" else 0.0) - self.freeze)
        x = game.opp_x
        speed = 0.0 if self.prev_x is None else abs(x - self.prev_x)
        self.prev_x = x
        self.moving += 0.2 * (min(1.0, speed * 40) - self.moving)
        self.step += speed * 40
        if self.bubble and now > self.bubble[1]:
            self.bubble = None

    def mood(self, now):
        if self.over:
            return self.over
        for mood in ("cheer", "dismay", "flinch"):
            if now < self.until.get(mood, 0):
                return mood
        return self.reaction or "ready"

    # ---- Drawing ----

    def draw(self, img, game, now, look_px):
        """look_px: pixel the character looks at (the ball, or your paddle)."""
        h, w = img.shape[:2]
        mood = self.mood(now)
        f = self.freeze
        u = h / Z_OPP * SCALE
        fx, fy = to_px(game.opp_x, 0, Z_OPP, w, h)
        shiver = 0.012 * f * math.sin(now * 55)
        jump = abs(math.sin(now * 9)) * 0.14 if mood == "cheer" else 0.0
        bob = 0.012 * math.sin(now * 3)
        lean = max(-1, min(1, (game.ball_world(now) or (game.opp_x,))[0] - game.opp_x)) * 0.05
        swing = (now - self.swing_t) / SWING_S

        def P(x, y):
            return int(fx + (x + shiver) * u), int(fy - y * u)

        col = lambda c: mix(c, ICE, 0.4 * f)
        th = max(2, int(0.075 * u))

        def limb(a, b, color, t=th):
            cv2.line(img, P(*a), P(*b), color, t, cv2.LINE_AA)
            for p in (a, b):
                cv2.circle(img, P(*p), t // 2, color, -1, cv2.LINE_AA)

        # -- Skeleton --
        hip_y = 0.5 + bob + jump
        sh_y = 0.92 + bob + jump
        head_dx = head_dy = 0.0
        hands = {"ready": ((-0.34, 0.62 + 0.03 * math.sin(now * 4)), (0.42 + 0.03 * math.sin(now * 4), 0.72)),
                 "cheer": ((-0.4 + 0.05 * math.sin(now * 12), 1.38 + jump), (0.4 - 0.05 * math.sin(now * 12), 1.38 + jump)),
                 "dismay": ((-0.19, 1.05), (0.19, 1.05)),
                 "frozen": ((0.1, 0.78), (-0.1, 0.74)),
                 "shocked": ((-0.3, 1.1), (0.3, 1.1)),
                 "flinch": ((-0.24, 1.08), (0.24, 1.08))}[mood]
        if mood == "dismay":
            sh_y -= 0.07
            head_dy, head_dx = -0.1, 0.03 * math.sin(now * 14)
        elif mood == "flinch":
            sh_y -= 0.05
            head_dy = -0.04
        hl, hr = hands
        if 0 <= swing < 1:      # paddle swing: sweep from high on the right across to low on the left
            e = swing * swing * (3 - 2 * swing)
            hr = (lerp(0.55, -0.15, e), lerp(1.1, 0.5, e) + 0.14 * math.sin(math.pi * e))
            lean -= 0.05 * math.sin(math.pi * e)
        sl, sr = (lean - 0.16, sh_y), (lean + 0.16, sh_y)
        head = (lean * 1.4 + head_dx, sh_y + 0.2 + head_dy)

        # Legs: a crouched stance, with the feet stepping as the figure shuffles.
        hip_l, hip_r = (-0.09 + lean * 0.4, hip_y), (0.09 + lean * 0.4, hip_y)
        stride = 0.07 * math.sin(self.step) * self.moving
        lift = 0.05 * self.moving
        foot_l = (-0.16 + stride, jump + lift * max(0.0, math.cos(self.step)))
        foot_r = (0.16 - stride, jump + lift * max(0.0, -math.cos(self.step)))
        for hip, foot, side in ((hip_l, foot_l, -1), (hip_r, foot_r, 1)):
            knee, foot = ik(hip, foot, 0.28, 0.28, (side, 0.1))
            limb(hip, knee, col(SHORTS), int(th * 1.25))
            limb(knee, foot, col(SKIN), int(th * 1.1))
            cv2.ellipse(img, P(foot[0] + side * 0.03, foot[1] + 0.01), (int(0.1 * u), int(0.04 * u)), 0, 0, 360,
                        col(SHOE), -1, cv2.LINE_AA)

        # Torso, with the shirt label.
        torso = [P(sl[0] - 0.03, sh_y + 0.02), P(sr[0] + 0.03, sh_y + 0.02),
                 P(hip_r[0] + 0.05, hip_y), P(hip_l[0] - 0.05, hip_y)]
        hull = cv2.convexHull(np.array(torso))
        cv2.fillPoly(img, [hull], col(SHIRT), cv2.LINE_AA)
        cv2.polylines(img, [hull], True, col((30, 30, 100)), 1, cv2.LINE_AA)
        tx, ty = P(lean * 0.7 - 0.11, (sh_y + hip_y) / 2 - 0.02)
        cv2.putText(img, "CPU", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.0045 * u, WHITE, 1, cv2.LINE_AA)
        limb((lean, sh_y), (head[0], head[1] - 0.1), col(SKIN), th)   # neck

        # Arms, then the paddle in the right hand (the screen-right one).
        elbow_r = None
        for shoulder, hand, side in ((sl, hl, -1), (sr, hr, 1)):
            elbow, hand = ik(shoulder, hand, 0.3, 0.3, (side, -0.6))
            limb(shoulder, elbow, col(SHIRT), th)
            limb(elbow, hand, col(SKIN), th)
            cv2.circle(img, P(*hand), int(th * 0.7), col(SKIN), -1, cv2.LINE_AA)
            if side == 1:
                elbow_r, hand_r = elbow, hand
        ax, ay = hand_r[0] - elbow_r[0], hand_r[1] - elbow_r[1]
        n = math.hypot(ax, ay) or 1.0
        pc = (hand_r[0] + ax / n * 0.1, hand_r[1] + ay / n * 0.1)
        cv2.line(img, P(*hand_r), P(*pc), (60, 110, 170), max(2, th // 2), cv2.LINE_AA)
        cv2.circle(img, P(*pc), int(0.115 * u), col(PADDLE_RIM), -1, cv2.LINE_AA)
        cv2.circle(img, P(*pc), int(0.095 * u), col(PADDLE), -1, cv2.LINE_AA)

        self._draw_head(img, head, mood, P, u, look_px, now, col, game)
        if f > 0.3:
            self._draw_breath(img, head, P, u, now, f)
        if self.bubble:
            self._draw_bubble(img, self.bubble[0], P(*head), u, game.opp_x)

    def _draw_head(self, img, c, mood, P, u, look_px, now, col, game):
        R = 0.17
        cx, cy = P(*c)
        r = int(R * u)
        # Hair behind, a spiky crest, face, headband.
        cv2.ellipse(img, (cx, cy), (int(r * 1.1), int(r * 1.1)), 0, 180, 360, col(HAIR), -1, cv2.LINE_AA)
        for k in (-1, 0, 1):
            tip = P(c[0] + k * 0.09, c[1] + R * 1.45 - abs(k) * 0.04)
            base = [P(c[0] + k * 0.09 - 0.05, c[1] + R * 0.95), P(c[0] + k * 0.09 + 0.05, c[1] + R * 0.95), tip]
            cv2.fillConvexPoly(img, np.array(base), col(HAIR), cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, col(SKIN), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, col((60, 90, 140)), 1, cv2.LINE_AA)
        by = c[1] + 0.095
        cv2.line(img, P(c[0] - R * 0.97, by), P(c[0] + R * 0.97, by), BAND, max(2, int(0.045 * u)), cv2.LINE_AA)

        # Eyes follow the look target.
        gx, gy = look_px[0] - cx, look_px[1] - cy
        gn = math.hypot(gx, gy) or 1.0
        tgt = (gx / gn * min(1.0, gn / (4 * r)), gy / gn * min(1.0, gn / (4 * r)))
        self.gaze = tuple(lerp(g, t, 0.35) for g, t in zip(self.gaze, tgt))
        wide = 1.35 if mood in ("shocked", "flinch") else 1.0
        er = max(2, int(0.052 * u * wide))
        for side in (-1, 1):
            ex, ey = P(c[0] + side * 0.065, c[1] - 0.005)
            cv2.circle(img, (ex, ey), er, WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, (ex + int(self.gaze[0] * er * 0.5), ey + int(self.gaze[1] * er * 0.5)),
                       max(1, er // 2), (30, 30, 30), -1, cv2.LINE_AA)
            # Brows: tilted in when determined, up and out when scared, drooping when sad.
            tilt, raise_ = {"ready": (0.03, 0.0), "cheer": (-0.01, 0.03), "dismay": (-0.04, 0.0),
                            "frozen": (-0.035, 0.0), "shocked": (-0.02, 0.05), "flinch": (-0.02, 0.05)}[mood]
            if mood == "ready" and self.rally >= 5:
                tilt = 0.05
            inner, outer = c[1] + 0.07 + raise_ - tilt, c[1] + 0.07 + raise_ + tilt
            cv2.line(img, P(c[0] + side * 0.03, inner), P(c[0] + side * 0.1, outer), col(HAIR), max(1, int(0.025 * u)),
                     cv2.LINE_AA)

        # Mouth.
        mx, my = P(c[0], c[1] - 0.085)
        mw = int(0.07 * u)
        dark = (40, 30, 120)
        if mood == "cheer":
            cv2.ellipse(img, (mx, my - mw // 4), (mw, int(mw * 0.8)), 0, 0, 180, dark, -1, cv2.LINE_AA)
        elif mood == "dismay":
            cv2.ellipse(img, (mx, my + mw // 2), (int(mw * 0.8), int(mw * 0.5)), 0, 200, 340, dark, 2, cv2.LINE_AA)
        elif mood == "frozen":      # teeth chattering
            open_ = int(mw * (0.2 + 0.25 * (math.sin(now * 40) > 0)))
            cv2.rectangle(img, (mx - mw // 2, my - open_ // 2), (mx + mw // 2, my + open_ // 2), WHITE, -1)
            cv2.rectangle(img, (mx - mw // 2, my - open_ // 2), (mx + mw // 2, my + open_ // 2), dark, 1)
        elif mood in ("shocked", "flinch"):
            cv2.circle(img, (mx, my), max(2, int(mw * 0.4)), dark, -1, cv2.LINE_AA)
        elif self.rally >= 5:       # concentrating
            cv2.line(img, (mx - mw // 2, my), (mx + mw // 2, my), dark, 2, cv2.LINE_AA)
        else:
            cv2.ellipse(img, (mx, my - mw // 4), (int(mw * 0.8), int(mw * 0.55)), 0, 20, 160, dark, 2, cv2.LINE_AA)

        # Sweat drop when nervous or in a long rally.
        if mood in ("shocked", "flinch") or (self.rally >= 5 and mood == "ready"):
            drop = (now * 1.5) % 1.0
            sx, sy = P(c[0] + R * 1.05, c[1] + 0.05 - drop * 0.2)
            cv2.circle(img, (sx, sy), max(2, int(0.035 * u)), (255, 200, 120), -1, cv2.LINE_AA)
            cv2.fillConvexPoly(img, np.array([(sx - 2, sy), (sx + 2, sy), (sx, sy - int(0.07 * u))]),
                               (255, 200, 120), cv2.LINE_AA)

    def _draw_breath(self, img, head, P, u, now, f):
        """Cold breath puffs drifting up from the mouth."""
        for i in range(3):
            t = ((now * 0.9 + i / 3) % 1.0)
            x, y = P(head[0] + 0.08 + t * 0.2, head[1] - 0.1 + t * 0.2)
            cv2.circle(img, (x, y), max(1, int((0.02 + 0.05 * t) * u)), mix((255, 255, 255), (200, 200, 200), t),
                       1, cv2.LINE_AA)

    def _draw_bubble(self, img, text, head_px, u, opp_x):
        """Speech bubble beside the head, on the side facing the middle of the screen."""
        w = img.shape[1]
        scale = 0.5 * w / 560
        (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        pad = int(6 * w / 560)
        bw, bh = tw + 2 * pad, th_ + 2 * pad
        side = -1 if opp_x > 0.1 else 1
        x0 = head_px[0] + int(0.38 * u) if side > 0 else head_px[0] - int(0.38 * u) - bw
        x0 = max(4, min(w - bw - 4, x0))
        y0 = max(4, head_px[1] - bh // 2 - int(0.05 * u))
        edge = x0 if side > 0 else x0 + bw
        tail = np.array([(head_px[0] + side * int(0.2 * u), head_px[1]), (edge, y0 + bh // 4), (edge, y0 + 3 * bh // 4)])
        cv2.fillConvexPoly(img, tail, WHITE, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), WHITE, -1)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(img, text, (x0 + pad, y0 + pad + th_), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 1, cv2.LINE_AA)
