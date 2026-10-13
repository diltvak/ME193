"""All drawing: the court, ball, paddle, HUD (including powerups), menu and webcam overlays."""

import math

import cv2
import numpy as np

from swing import FORWARD_THRESHOLD, GYRO_THRESHOLD, SPEED_THRESHOLD

from powerup_pong.config import (BALL_COLOR, BALL_R, DIFFICULTIES, GREEN, HIT_STYLE, MENU_BG, MINIFIG_COLORS,
                                 NET_H, PADDLE_DARK, PADDLE_R, PADDLE_RED, RED, TABLE, TABLE_HALF_W, TABLE_SIDE,
                                 WHITE, WOOD, YELLOW, Z_FAR, Z_NEAR, Z_NET, Z_TABLE_NEAR)
from powerup_pong.drawing import put, put_centered
from powerup_pong.geometry import proj, to_px


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
        return to_px(X, Y, z, pw, ph)

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


def draw_powerup_hud(top, manager, now):
    """One row per powerup, top left: a colored token plus its status (ready / active / used)."""
    s = top.shape[1] / 560
    for i, p in enumerate(manager):
        c = (int(20 * s), int((66 + 28 * i) * s))
        r = int(9 * s)
        used = p.state == "used"
        if p.state in ("active", "fading"):
            pulse = 0.5 + 0.5 * math.sin(now * 8)
            cv2.circle(top, c, r + int((3 + 4 * pulse) * s), p.color, 2, cv2.LINE_AA)
        cv2.circle(top, c, r, (110, 110, 110) if used else p.color, -1, cv2.LINE_AA)
        cv2.circle(top, c, r, (0, 0, 0), 1, cv2.LINE_AA)
        color = (150, 150, 150) if used else WHITE if p.state == "ready" else p.color
        text = f"{p.hud_name} {p.hud_detail()}"
        put(top, text, c[0] + int(18 * s), c[1] + int(5 * s), 0.5 * s, color)
        if used:
            width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, 1)[0][0]
            y = c[1]
            cv2.line(top, (c[0] + int(18 * s), y), (c[0] + int(18 * s) + width, y), (150, 150, 150), 1, cv2.LINE_AA)


def draw_banner(top, manager, now):
    banner = manager.active_banner(now)
    if banner:
        title, sub, color, t = banner
        s = top.shape[1] / 560
        put_centered(top, title, int(top.shape[0] * 0.56), 1.3 * s, color, 4)
        if sub:
            put_centered(top, sub, int(top.shape[0] * 0.56) + int(30 * s), 0.6 * s, WHITE, 2)


def draw_top(top, court, game, paddle, detector, opponent, gyro_dps, now, flash):
    h, w = top.shape[:2]
    s = w / 560
    top[:] = court
    # The CPU watches the ball while it is in play, and your paddle otherwise.
    world = game.ball_world(now)
    look = to_px(*world, w, h) if world else (int(paddle.pos[0] * w), int(paddle.pos[1] * h)) if paddle.pos else (w // 2, h)
    opponent.update(game, now)
    opponent.draw(top, game, now, look)
    game.powerups.draw_scene(top, game, now)
    draw_ball(top, game, now)
    if game.flash and now - game.flash[0] < 0.4:
        age = now - game.flash[0]
        pos = (int(game.flash[1][0] * w), int(game.flash[1][1] * h))
        label, color = HIT_STYLE[game.flash[2]]
        cv2.circle(top, pos, int((20 + 200 * age) * s), color, 3, cv2.LINE_AA)
        put_centered(top, label, max(20, pos[1] - int(40 * s)), 0.9 * s, color, 2)
    game.powerups.draw_overlay(top, game, now)
    draw_paddle(top, paddle, gyro_dps, flash)

    if game.practice:
        put_centered(top, "WARM-UP", int(32 * s), 0.9 * s, YELLOW, 2)
        for i, line in enumerate((f"returns {game.returns}", f"misses {game.misses}", f"streak {game.rally}")):
            put(top, line, int(10 * s), int((20 + 16 * i) * s), 0.45 * s, WHITE)
        put(top, "ENTER / ESC: back to setup", int(10 * s), h - int(10 * s), 0.5 * s, WHITE)
        mw, mx = int(110 * s), int(10 * s)
        draw_meter(top, "wrist speed", detector.speed, SPEED_THRESHOLD, 3.0, mx, h - int(110 * s), mw, s)
        draw_meter(top, "toward camera", detector.forward, FORWARD_THRESHOLD, 2.0, mx, h - int(80 * s), mw, s)
        if gyro_dps is not None:
            draw_meter(top, "IMU gyro", gyro_dps, GYRO_THRESHOLD, 450.0, mx, h - int(50 * s), mw, s)
    else:
        put_centered(top, f"YOU {game.player}  :  {game.opp} CPU", int(34 * s), 0.9 * s, WHITE, 2)
        put(top, f"first to {game.target}", int(10 * s), int(22 * s), 0.5 * s, WHITE)
        if game.rally > 1:
            put(top, f"rally {game.rally}", w - int(100 * s), int(30 * s), 0.6 * s, YELLOW)
    draw_powerup_hud(top, game.powerups, now)
    draw_banner(top, game.powerups, now)

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


def draw_menu(top, difficulty, points, arm, pose_ok, imu_on, gyro_dps, haptics_on, powerups_on, minifig_status):
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
    put(top, f"[H] Haptic feedback: {'on' if haptics_on else 'off'}" + ("" if imu_on else " (needs the motor)"),
        x, int(292 * s), 0.55 * s, WHITE)
    model = "model ready" if minifig_status == "ready" else \
        "model loading..." if minifig_status == "loading" else "no model - keys Y / G still work"
    put(top, f"[M] Powerups: {'on' if powerups_on else 'off'}" + (f" ({model})" if powerups_on else ""),
        x, int(318 * s), 0.55 * s, WHITE if powerups_on else (170, 170, 170))

    put(top, "Pose tracking: " + ("arm found" if pose_ok else "not found - stand back"),
        x, int(350 * s), 0.55 * s, GREEN if pose_ok else RED)
    imu_text = f"IMU: on (gyro {gyro_dps:4.0f} deg/s)" if imu_on else "IMU: off (camera only)"
    put(top, imu_text, x, int(374 * s), 0.55 * s, GREEN if imu_on else YELLOW)
    put_centered(top, "F fullscreen    V game-only view    + / - size", h - int(32 * s), 0.45 * s, (200, 200, 200), 1)
    put_centered(top, "W warm-up    ENTER start    Q quit", h - int(12 * s), 0.6 * s, WHITE, 1)


def draw_bottom_info(panel, detector, gyro_dps, flash):
    s = panel.shape[1] / 560
    put(panel, "YOU (camera)", int(10 * s), int(22 * s), 0.6 * s, WHITE)
    put(panel, f"wrist speed {detector.speed:4.1f}  forward {detector.forward:4.1f}",
        int(10 * s), panel.shape[0] - int(34 * s), 0.5 * s, WHITE)
    put(panel, f"gyro {gyro_dps:4.0f} deg/s" if gyro_dps is not None else "IMU off",
        int(10 * s), panel.shape[0] - int(12 * s), 0.5 * s, WHITE)
    if flash:
        put(panel, "swing!", panel.shape[1] - int(90 * s), int(24 * s), 0.7 * s, GREEN, 2)


def draw_minifigs(panel, minifig):
    """YOLO boxes on the (mirrored) webcam panel, plus the model status."""
    h, w = panel.shape[:2]
    s = w / 560
    for name, conf, (x1, y1, x2, y2) in minifig.detections:
        color = MINIFIG_COLORS.get(name, WHITE)
        p1, p2 = (int((1 - x2) * w), int(y1 * h)), (int((1 - x1) * w), int(y2 * h))   # mirror the x axis
        cv2.rectangle(panel, p1, p2, color, 2)
        put(panel, f"{name} {conf:.2f}", p1[0], max(14, p1[1] - 6), 0.5 * s, color)
    if minifig.status not in ("ready", "off"):
        text = "minifig model loading..." if minifig.status == "loading" else "minifig model unavailable"
        put(panel, text, int(10 * s), int(46 * s), 0.5 * s, YELLOW if minifig.status == "loading" else RED)
