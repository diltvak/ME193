"""Ping pong against a virtual opponent, played with real arm swings.

The window is split in two. The top half is the game: a 3D-ish table seen from the player's
end, a virtual opponent who returns the ball, and your paddle. The ball flies out of the
screen toward you, and your paddle is drawn where your wrist is. The bottom half is the live
webcam view with your arm tracked, as in swing.py.

Paddle position comes from the wrist landmark (pose detection); the paddle angle follows your
forearm and it glows with the LEGO Double Motor gyro rate. A return counts when a swing
(camera + IMU, see SwingDetector in swing.py) lands while the ball is in the hit window AND
your paddle is on the ball. Without the IMU (or with --no-imu) swings are detected by camera only.

Setup screen keys:  1/2/3 difficulty   P points to win   A swap arm   W warm-up
                    ENTER/SPACE start  Q quit
In a game or warm-up: ESC returns to setup, Q quits.

Run: python3 first_pong.py [--arm right|left] [--difficulty easy|medium|hard] [--points 7]
                           [--width 560] [--no-imu]
"""

import argparse
import math
import random
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from pose_detection import MODEL_PATH, PL, ensure_model
from swing import (FORWARD_THRESHOLD, GREEN, GYRO_THRESHOLD, MIN_VISIBILITY, RED, SPEED_THRESHOLD,
                   WHITE, YELLOW, SwingDetector, connect_imu, draw_arm, gyro_magnitude)

# ---- Difficulty ----
# flight: seconds for the ball to travel from the opponent to you; window: seconds either side of
# arrival in which a swing can return it; reach: how close (in panel widths) the paddle must be to
# the ball; miss_p: chance the opponent fails to return; spread: how wide the opponent aims.
DIFFICULTIES = {
    "easy":   dict(flight=2.0, window=0.40, reach=0.22, miss_p=0.20, spread=0.22,
                   blurb="slow ball, forgiving timing"),
    "medium": dict(flight=1.4, window=0.28, reach=0.17, miss_p=0.12, spread=0.30,
                   blurb="faster ball, tighter timing"),
    "hard":   dict(flight=0.95, window=0.18, reach=0.12, miss_p=0.06, spread=0.36,
                   blurb="fast, wide shots, small window"),
}
PRACTICE = dict(flight=2.2, window=0.45, reach=0.25, miss_p=0.0, spread=0.20, blurb="")
POINT_CHOICES = (3, 5, 7, 11)

AFTER_WINDOW_FRAC = 0.6   # the hit window is shorter after the ball's arrival than before it
SWING_MEMORY_S = 0.25     # a detected swing can still return the ball this long afterward
SPEEDUP, MIN_SPEEDUP = 0.96, 0.65   # each return in a rally shortens the flight, down to this fraction
SERVE_DELAY_S = 1.2
RETURN_S = 0.8            # flight time of your return back to the opponent
RESULT_S = 1.6

# ---- Scene geometry ----
# World: X is sideways (panel widths at z = 1), Y is height above the table (panel heights at
# z = 1), z is distance from the player (the ball is on you at z = 1, behind you below that).
VP = (0.5, 0.28)          # vanishing point, as a fraction of the panel
G_OFF = 0.52              # table surface sits this far below the vanishing point at z = 1
Z_FAR, Z_NEAR, Z_MIN = 5.0, 1.0, 0.15
Z_TABLE_NEAR, Z_NET, TABLE_HALF_W, NET_H = 1.5, 3.25, 0.6, 0.07
BOUNCE_P = 0.6            # fraction of the flight at which the ball bounces on the table
BALL_R = 0.045            # ball radius in panel widths at z = 1

PADDLE_R = 0.095          # paddle radius in panel widths
PADDLE_CENTER = (0.5, 0.55)   # camera-frame point that maps to the middle of the game view
PADDLE_GAIN = 1.6         # wrist travel in the camera -> paddle travel in the game view
PADDLE_SMOOTHING = 0.5

TABLE, TABLE_SIDE = (140, 85, 25), (90, 50, 15)
WOOD, PADDLE_RED, PADDLE_DARK = (60, 110, 170), (40, 40, 210), (20, 20, 70)
BALL_COLOR = (40, 150, 255)
MENU_BG = (45, 30, 25)


def proj(X, Y, z):
    """Normalized (0-1) panel position and perspective scale of a world point."""
    k = 1.0 / z
    return VP[0] + X * k, VP[1] + (G_OFF - Y) * k, k


def put(img, text, x, y, scale, color, thickness=1):
    # Outline by offset copies; a thicker stroke would also widen the glyph spacing.
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_centered(img, text, y, scale, color, thickness=1):
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    put(img, text, (img.shape[1] - width) // 2, y, scale, color, thickness)


class Paddle:
    """Smoothed paddle position (panel fractions) and handle angle taken from the pose."""

    def __init__(self):
        self.pos = None
        self.dir = (0.0, 1.0)     # unit vector from the paddle toward the elbow (handle side)
        self.tracked = False
        self.trail = deque(maxlen=8)

    @property
    def angle(self):
        return math.atan2(self.dir[1], self.dir[0])

    @property
    def xy(self):
        return self.pos if self.tracked else None

    def update(self, landmarks, arm):
        wrist = landmarks[PL[f"{arm.upper()}_WRIST"].value] if landmarks else None
        if wrist is None or wrist.visibility < MIN_VISIBILITY:
            self.tracked = False
            return
        elbow = landmarks[PL[f"{arm.upper()}_ELBOW"].value]
        mx, my = 1 - wrist.x, wrist.y     # mirrored to match the flipped view
        target = tuple(min(0.97, max(0.03, 0.5 + (v - c) * PADDLE_GAIN))
                       for v, c in zip((mx, my), PADDLE_CENTER))
        if self.pos is None:
            self.pos = target
        else:
            self.pos = tuple(p + PADDLE_SMOOTHING * (t - p) for p, t in zip(self.pos, target))
        if elbow.visibility >= MIN_VISIBILITY:
            dx, dy = (wrist.x - elbow.x) * 4, (elbow.y - wrist.y) * 3   # panel is 4:3
            norm = math.hypot(dx, dy) or 1.0
            self.dir = tuple(d + 0.4 * (n / norm - d) for d, n in zip(self.dir, (dx / norm, dy / norm)))
        self.tracked = True
        self.trail.append(self.pos)


class Game:
    """Rally logic: opponent serves, you return with a swing, first to `target` points wins."""

    def __init__(self, cfg, target, practice, now):
        self.cfg, self.target, self.practice = cfg, target, practice
        self.player = self.opp = 0
        self.returns = self.misses = 0      # warm-up counters
        self.rally = self.best = 0
        self.phase = "serve"                # serve -> incoming -> returned -> (serve | result -> serve | over)
        self.phase_t = now
        self.ball = None
        self.opp_x = 0.0
        self.last_swing = -math.inf
        self.msg, self.msg_color = "", WHITE
        self.flash = None                   # (time, panel position) of the last return

    def in_window(self, now):
        if self.phase != "incoming":
            return False
        arrive, window = self.ball["arrive"], self.cfg["window"]
        return arrive - window <= now <= arrive + window * AFTER_WINDOW_FRAC

    def ball_world(self, now):
        """(X, Y, z) of the ball, or None when it is out of play."""
        b = self.ball
        if b is None:
            return None
        if b["mode"] == "in":
            p = (now - b["t0"]) / b["flight"]
            z = Z_FAR + (Z_NEAR - Z_FAR) * p
            X = b["x0"] + (b["x1"] - b["x0"]) * p
            if p < BOUNCE_P:
                Y = 0.25 * (1 - (p / BOUNCE_P) ** 2)
            else:
                Y = b["h1"] * (1 - (1 - (p - BOUNCE_P) / (1 - BOUNCE_P)) ** 2)
            return (X, Y, z) if z >= Z_MIN else None
        q = (now - b["t0"]) / RETURN_S
        z = b["z0"] + (Z_FAR - b["z0"]) * q
        X = b["x0"] + (b["x1"] - b["x0"]) * q
        Y = b["y0"] * (1 - q) + 0.25 * q + 0.2 * math.sin(math.pi * min(q, 1.0))
        return (X, Y, z) if q <= 1.3 else None

    def _launch(self, now):
        speed = 1.0 if self.practice else max(MIN_SPEEDUP, SPEEDUP ** self.rally)
        flight = self.cfg["flight"] * speed
        spread = self.cfg["spread"]
        self.ball = dict(mode="in", t0=now, flight=flight, arrive=now + flight, x0=self.opp_x,
                         x1=random.uniform(-spread, spread), h1=random.uniform(0.15, 0.42))
        self.phase = "incoming"

    def _return(self, now, world, pos):
        X, Y, z = world
        self.ball = dict(mode="out", t0=now, x0=X, y0=Y, z0=z, x1=random.uniform(-0.18, 0.18))
        self.phase = "returned"
        self.last_swing = -math.inf
        self.rally += 1
        self.best = max(self.best, self.rally)
        self.returns += 1
        self.flash = (now, pos)

    def _end_point(self, now, player_scored):
        if player_scored and not self.practice:
            self.player += 1
        elif not player_scored:
            self.misses += 1
            if not self.practice:
                self.opp += 1
        self.rally = 0
        self.msg, self.msg_color = ("POINT!", GREEN) if player_scored else ("MISS", RED)
        self.phase, self.phase_t = "result", now

    def winner(self):
        if self.practice:
            return None
        return "player" if self.player >= self.target else "opp" if self.opp >= self.target else None

    def update(self, now, swung, paddle_xy):
        if swung:
            self.last_swing = now
        # The opponent shuffles toward the ball while it is near their end.
        world = self.ball_world(now)
        self.opp_x += 0.15 * ((world[0] if world and world[2] > 3.5 else 0.0) - self.opp_x)

        if self.phase == "serve" and now - self.phase_t >= SERVE_DELAY_S:
            self._launch(now)
        if self.phase == "incoming":
            if world and paddle_xy and self.in_window(now) and now - self.last_swing <= SWING_MEMORY_S:
                sx, sy, _ = proj(*world)
                if math.hypot(paddle_xy[0] - sx, (paddle_xy[1] - sy) * 0.75) <= self.cfg["reach"]:
                    self._return(now, world, (sx, sy))
            if self.phase == "incoming" and now > self.ball["arrive"] + self.cfg["window"] * AFTER_WINDOW_FRAC:
                self._end_point(now, player_scored=False)
        elif self.phase == "returned" and now - self.ball["t0"] >= RETURN_S:
            if random.random() >= self.cfg["miss_p"]:
                self.opp_x = self.ball["x1"]
                self._launch(now)
            else:
                self._end_point(now, player_scored=True)
        elif self.phase == "result" and now - self.phase_t >= RESULT_S:
            if self.winner():
                self.phase, self.phase_t = "over", now
            else:
                self.phase, self.phase_t, self.ball, self.opp_x = "serve", now, None, 0.0


# ---- Drawing ----

def make_court(pw, ph):
    """Static background: wall, floor, table and net."""
    img = np.empty((ph, pw, 3), np.uint8)
    horizon = int(0.33 * ph)
    for y in range(ph):
        if y < horizon:
            t, a, b = y / horizon, np.array((60, 40, 35)), np.array((95, 70, 55))
        else:
            t, a, b = (y - horizon) / (ph - horizon), np.array((70, 95, 60)), np.array((40, 55, 35))
        img[y] = a + (b - a) * t

    def pt(X, Y, z):
        sx, sy, _ = proj(X, Y, z)
        return int(sx * pw), int(sy * ph)

    far_l, far_r = pt(-TABLE_HALF_W, 0, Z_FAR), pt(TABLE_HALF_W, 0, Z_FAR)
    near_l, near_r = pt(-TABLE_HALF_W, 0, Z_TABLE_NEAR), pt(TABLE_HALF_W, 0, Z_TABLE_NEAR)
    drop = int(0.035 * ph)
    cv2.fillPoly(img, [np.array([near_l, near_r, (near_r[0], near_r[1] + drop), (near_l[0], near_l[1] + drop)])],
                 TABLE_SIDE)
    cv2.fillPoly(img, [np.array([far_l, far_r, near_r, near_l])], TABLE)
    cv2.polylines(img, [np.array([far_l, far_r, near_r, near_l])], True, WHITE, 2, cv2.LINE_AA)
    cv2.line(img, pt(0, 0, Z_FAR), pt(0, 0, Z_TABLE_NEAR), WHITE, 1, cv2.LINE_AA)

    w = TABLE_HALF_W + 0.04
    net = np.array([pt(-w, 0, Z_NET), pt(w, 0, Z_NET), pt(w, NET_H, Z_NET), pt(-w, NET_H, Z_NET)])
    overlay = img.copy()
    cv2.fillPoly(overlay, [net], (215, 215, 215))
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.line(img, tuple(net[3]), tuple(net[2]), WHITE, 2, cv2.LINE_AA)
    return img


def draw_opponent(img, x):
    h, w = img.shape[:2]
    z = 5.6

    def p(X, Y):
        sx, sy, _ = proj(X, Y, z)
        return int(sx * w), int(sy * h)

    k = 1 / z
    cv2.rectangle(img, p(x - 0.3, 0.8), p(x + 0.3, 0.1), (60, 60, 170), -1)
    cv2.circle(img, p(x, 0.97), max(3, int(0.12 * h * k)), (150, 190, 230), -1, cv2.LINE_AA)
    cv2.circle(img, p(x + 0.45, 0.6), max(3, int(0.09 * w * k)), PADDLE_RED, -1, cv2.LINE_AA)


def draw_ball(img, game, now):
    world = game.ball_world(now)
    if world is None:
        return
    h, w = img.shape[:2]
    X, Y, z = world
    sx, sy, k = proj(X, Y, z)
    c, r = (int(sx * w), int(sy * h)), max(2, int(BALL_R * w * k))
    if Y >= 0 and Z_TABLE_NEAR <= z <= Z_FAR:
        gx, gy, _ = proj(X, 0, z)
        cv2.ellipse(img, (int(gx * w), int(gy * h)), (r, max(1, r // 3)), 0, 0, 360, (30, 20, 10), -1, cv2.LINE_AA)
    cv2.circle(img, c, r, BALL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), max(1, r // 4), (200, 230, 255), -1, cv2.LINE_AA)
    if game.in_window(now):
        cv2.circle(img, c, r + max(4, r // 3), GREEN, 2, cv2.LINE_AA)


def draw_paddle(img, paddle, gyro_dps, flash):
    if paddle.pos is None:
        return
    h, w = img.shape[:2]
    layer = img if paddle.tracked else img.copy()
    cx, cy, r = int(paddle.pos[0] * w), int(paddle.pos[1] * h), int(PADDLE_R * w)
    pts = list(paddle.trail)
    for i in range(1, len(pts)):
        a = (int(pts[i - 1][0] * w), int(pts[i - 1][1] * h))
        b = (int(pts[i][0] * w), int(pts[i][1] * h))
        cv2.line(layer, a, b, (210, 200, 255), max(1, int(r * 0.6 * i / len(pts))), cv2.LINE_AA)
    ux, uy = math.cos(paddle.angle), math.sin(paddle.angle)
    cv2.line(layer, (int(cx + ux * r * 0.7), int(cy + uy * r * 0.7)),
             (int(cx + ux * r * 1.8), int(cy + uy * r * 1.8)), WOOD, max(3, int(r * 0.32)), cv2.LINE_AA)
    power = 0.0 if gyro_dps is None else min(1.0, gyro_dps / (GYRO_THRESHOLD * 3))
    if power > 0.05:   # the IMU gyro rate makes the paddle glow
        cv2.circle(layer, (cx, cy), r + int(4 + 12 * power), YELLOW, 1 + int(3 * power), cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), r, PADDLE_DARK, -1, cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), r - 4, PADDLE_RED, -1, cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), r - 4, GREEN if flash else WHITE, 2 if flash else 1, cv2.LINE_AA)
    if layer is not img:
        cv2.addWeighted(layer, 0.4, img, 0.6, 0, img)


def draw_meter(img, label, value, threshold, vmax, x, y, width, s):
    h = max(6, int(9 * s))
    put(img, label, x, y - 4, 0.45 * s, WHITE)
    frac = min(1.0, max(0.0, value) / vmax)
    cv2.rectangle(img, (x, y), (x + width, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(width * frac), y + h), GREEN if value > threshold else (0, 160, 220), -1)
    tx = x + int(width * threshold / vmax)
    cv2.line(img, (tx, y - 2), (tx, y + h + 2), WHITE, 2)


def draw_top(top, court, game, paddle, detector, gyro_dps, now, flash):
    h, w = top.shape[:2]
    s = w / 560
    top[:] = court
    draw_opponent(top, game.opp_x)
    draw_ball(top, game, now)
    if game.flash and now - game.flash[0] < 0.4:
        age = now - game.flash[0]
        pos = (int(game.flash[1][0] * w), int(game.flash[1][1] * h))
        cv2.circle(top, pos, int((20 + 200 * age) * s), YELLOW, 3, cv2.LINE_AA)
        put_centered(top, "NICE!", max(20, pos[1] - int(40 * s)), 0.9 * s, YELLOW, 2)
    draw_paddle(top, paddle, gyro_dps, flash)

    if game.practice:
        put_centered(top, "WARM-UP", int(32 * s), 0.9 * s, YELLOW, 2)
        put_centered(top, f"returns {game.returns}   misses {game.misses}   streak {game.rally}",
                     int(56 * s), 0.55 * s, WHITE, 1)
        put(top, "ENTER / ESC: back to setup", int(10 * s), h - int(10 * s), 0.5 * s, WHITE)
        mw, mx = int(110 * s), int(10 * s)
        draw_meter(top, "wrist speed", detector.speed, SPEED_THRESHOLD, 3.0, mx, h - int(110 * s), mw, s)
        draw_meter(top, "toward camera", detector.forward, FORWARD_THRESHOLD, 2.0, mx, h - int(80 * s), mw, s)
        if gyro_dps is not None:
            draw_meter(top, "IMU gyro", gyro_dps, GYRO_THRESHOLD, 450.0, mx, h - int(50 * s), mw, s)
    else:
        put_centered(top, f"YOU {game.player}  :  {game.opp} CPU", int(34 * s), 0.9 * s, WHITE, 2)
        put_centered(top, f"first to {game.target}", int(56 * s), 0.5 * s, WHITE, 1)
        if game.rally > 1:
            put(top, f"rally {game.rally}", w - int(100 * s), int(30 * s), 0.6 * s, YELLOW)

    if game.phase == "serve":
        put_centered(top, "Get ready...", h // 2 + int(40 * s), 1.2 * s, YELLOW)
    elif game.phase == "result":
        put_centered(top, game.msg, h // 2, 2.0 * s, game.msg_color, 5)
    elif game.phase == "over":
        won = game.winner() == "player"
        put_centered(top, "YOU WIN!" if won else "CPU WINS", h // 2 - int(10 * s), 1.8 * s,
                     GREEN if won else RED, 5)
        put_centered(top, f"{game.player} - {game.opp}   best rally {game.best}", h // 2 + int(30 * s),
                     0.7 * s, WHITE, 2)
        put_centered(top, "ENTER: back to setup", h // 2 + int(65 * s), 0.7 * s, YELLOW, 2)
    if not paddle.tracked:
        put_centered(top, "step back so your arm is visible", h - int(14 * s), 0.55 * s, RED, 1)


def draw_menu(top, difficulty, points, arm, pose_ok, imu_on, gyro_dps):
    h, w = top.shape[:2]
    s = w / 560
    top[:] = MENU_BG
    x = int(40 * s)
    put_centered(top, "PING PONG", int(58 * s), 1.6 * s, YELLOW, 3)
    put(top, "Difficulty", x, int(100 * s), 0.6 * s, YELLOW)
    for i, (name, cfg) in enumerate(DIFFICULTIES.items()):
        y = int((130 + 28 * i) * s)
        selected = name == difficulty
        if selected:
            cv2.rectangle(top, (x - 8, y - int(19 * s)), (w - x, y + int(8 * s)), (100, 65, 25), -1)
        put(top, f"[{i + 1}] {name.title():<7} {cfg['blurb']}", x, y, 0.55 * s, GREEN if selected else WHITE)
    put(top, f"[P] Play to {points} points", x, int(240 * s), 0.55 * s, WHITE)
    put(top, f"[A] Swinging arm: {arm}", x, int(266 * s), 0.55 * s, WHITE)

    put(top, "Pose tracking: " + ("arm found" if pose_ok else "not found - stand back"),
        x, int(310 * s), 0.55 * s, GREEN if pose_ok else RED)
    imu_text = f"IMU: on (gyro {gyro_dps:4.0f} deg/s)" if imu_on else "IMU: off (camera only)"
    put(top, imu_text, x, int(336 * s), 0.55 * s, GREEN if imu_on else YELLOW)
    put_centered(top, "W warm-up    ENTER start    Q quit", h - int(20 * s), 0.6 * s, WHITE, 1)


def draw_bottom_info(panel, detector, gyro_dps, flash):
    s = panel.shape[1] / 560
    put(panel, "YOU (camera)", int(10 * s), int(22 * s), 0.6 * s, WHITE)
    put(panel, f"wrist speed {detector.speed:4.1f}  forward {detector.forward:4.1f}",
        int(10 * s), panel.shape[0] - int(34 * s), 0.5 * s, WHITE)
    put(panel, f"gyro {gyro_dps:4.0f} deg/s" if gyro_dps is not None else "IMU off",
        int(10 * s), panel.shape[0] - int(12 * s), 0.5 * s, WHITE)
    if flash:
        put(panel, "swing!", panel.shape[1] - int(90 * s), int(24 * s), 0.7 * s, GREEN, 2)


def main():
    parser = argparse.ArgumentParser(description="Ping pong played with arm swings.")
    parser.add_argument("--arm", choices=("right", "left"), default="right", help="swinging arm")
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="medium")
    parser.add_argument("--points", type=int, default=7, help="points needed to win")
    parser.add_argument("--width", type=int, default=560, help="width of each half of the window, in pixels")
    parser.add_argument("--no-imu", action="store_true", help="use the camera only")
    args = parser.parse_args()

    pw = args.width
    ph = pw * 3 // 4
    court = make_court(pw, ph)

    ensure_model()
    motor = None if args.no_imu else connect_imu()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (check macOS camera permission for your terminal/IDE).")

    arm, difficulty, points = args.arm, args.difficulty, args.points
    detector = SwingDetector(arm)
    paddle = Paddle()
    mode, game = "menu", None
    top = np.empty((ph, pw, 3), np.uint8)
    flash_until = 0.0
    start = time.monotonic()
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                # Detect on the unflipped frame so left/right labels are right, then mirror for display.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now - start) * 1000))
                bottom = cv2.resize(cv2.flip(frame, 1), (pw, ph))

                gyro_dps = gyro_magnitude(motor) if motor else None
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                swung = False
                if landmarks:
                    swung = detector.update(landmarks, now, gyro_dps)
                    if swung:
                        flash_until = now + 0.5
                    draw_arm(bottom, landmarks, arm, now < flash_until)
                else:
                    detector.prev = None
                paddle.update(landmarks, arm)

                if mode == "menu":
                    draw_menu(top, difficulty, points, arm, paddle.tracked, motor is not None, gyro_dps or 0.0)
                else:
                    game.update(now, swung, paddle.xy)
                    draw_top(top, court, game, paddle, detector, gyro_dps, now, now < flash_until)
                draw_bottom_info(bottom, detector, gyro_dps, now < flash_until)

                canvas = np.vstack([top, bottom])
                cv2.line(canvas, (0, ph), (pw, ph), WHITE, 2)
                cv2.imshow("Ping pong (q to quit)", canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or (key == 27 and mode == "menu"):
                    break
                if key == 27:
                    mode = "menu"
                elif mode == "menu":
                    if key in (ord("1"), ord("2"), ord("3")):
                        difficulty = tuple(DIFFICULTIES)[key - ord("1")]
                    elif key == ord("p"):
                        points = POINT_CHOICES[(POINT_CHOICES.index(points) + 1) % len(POINT_CHOICES)
                                               if points in POINT_CHOICES else 0]
                    elif key == ord("a"):
                        arm = "left" if arm == "right" else "right"
                        detector = SwingDetector(arm)
                    elif key == ord("w"):
                        mode, game = "warmup", Game(PRACTICE, points, True, now)
                    elif key in (13, 10, 32):
                        mode, game = "game", Game(DIFFICULTIES[difficulty], points, False, now)
                elif key in (13, 10, 32) and (mode == "warmup" or game.phase == "over"):
                    mode = "menu"
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if motor and motor.connected:
            motor.disconnect()


if __name__ == "__main__":
    main()
