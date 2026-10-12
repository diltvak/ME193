"""Rally logic: the opponent serves, you return with a swing, first to `target` points wins.

Powerups plug in through the PowerupManager (see powerups/base.py): Game asks it for a flight-time
multiplier on each launch, lets it intercept an incoming ball, and tells it about every return.
"""

import math
import random

from powerup_pong.config import (AFTER_WINDOW_FRAC, BOUNCE_P, MIN_SPEEDUP, RESULT_S, RETURN_S, SERVE_DELAY_S,
                                 SPEEDUP, SWING_MEMORY_S, WHITE, GREEN, RED, Z_FAR, Z_MIN, Z_NEAR)
from powerup_pong.geometry import proj
from powerup_pong.haptics import classify_hit
from powerup_pong.powerups import PowerupManager


class Game:
    def __init__(self, cfg, target, practice, now, powerups=()):
        self.cfg, self.target, self.practice = cfg, target, practice
        self.powerups = PowerupManager(powerups)
        self.player = self.opp = 0
        self.returns = self.misses = 0      # warm-up counters
        self.rally = self.best = 0
        self.phase = "serve"                # serve -> incoming -> returned -> (serve | result -> serve | over)
        self.phase_t = now
        self.ball = None
        self.opp_x = 0.0
        self.last_swing = -math.inf
        self.msg, self.msg_color = "", WHITE
        self.flash = None                   # (time, panel position, hit type) of the last return
        self.haptic = None                  # hit type waiting to be played on the motors

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

    def slow_ball(self, factor, now):
        """Stretch the remaining flight of a ball already on its way to you."""
        b = self.ball
        if self.phase == "incoming" and b and b["mode"] == "in":
            progress = (now - b["t0"]) / b["flight"]
            b["flight"] *= factor
            b["t0"] = now - progress * b["flight"]
            b["arrive"] = b["t0"] + b["flight"]

    def _launch(self, now):
        speed = 1.0 if self.practice else max(MIN_SPEEDUP, SPEEDUP ** self.rally)
        flight = self.cfg["flight"] * speed * self.powerups.flight_multiplier()
        spread = self.cfg["spread"]
        self.ball = dict(mode="in", t0=now, flight=flight, arrive=now + flight, x0=self.opp_x,
                         x1=random.uniform(-spread, spread), h1=random.uniform(0.15, 0.42))
        self.phase = "incoming"

    def _return(self, now, world, pos, kind):
        X, Y, z = world
        self.haptic = kind
        self.ball = dict(mode="out", t0=now, x0=X, y0=Y, z0=z, x1=random.uniform(-0.18, 0.18))
        self.phase = "returned"
        self.last_swing = -math.inf
        self.rally += 1
        self.best = max(self.best, self.rally)
        self.returns += 1
        self.flash = (now, pos, kind)
        self.powerups.on_return(self, now)

    def _swing_return(self, now, world, pos, power, offset):
        error = now - self.ball["arrive"]
        timing_error = abs(error) / (self.cfg["window"] * (AFTER_WINDOW_FRAC if error > 0 else 1.0))
        self._return(now, world, pos, classify_hit(power, offset, timing_error))

    def auto_return(self, now, world):
        """Return the ball with no swing from the player (used by powerups)."""
        sx, sy, _ = proj(*world)
        self._return(now, world, (sx, sy), "block")

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

    def update(self, now, swung, paddle_xy, power):
        """power is the recent swing strength in deg/s, used to classify a return."""
        if swung:
            self.last_swing = now
        self.powerups.update(self, now)
        # The opponent shuffles toward the ball while it is near their end.
        world = self.ball_world(now)
        self.opp_x += 0.15 * ((world[0] if world and world[2] > 3.5 else 0.0) - self.opp_x)

        if self.phase == "serve" and now - self.phase_t >= SERVE_DELAY_S:
            self._launch(now)
        if self.phase == "incoming":
            if world and self.powerups.intercept(self, now, world):
                pass    # a powerup returned the ball
            elif (world and paddle_xy and not self.powerups.takes_over_returns() and self.in_window(now)
                  and now - self.last_swing <= SWING_MEMORY_S):
                sx, sy, _ = proj(*world)
                offset = math.hypot(paddle_xy[0] - sx, (paddle_xy[1] - sy) * 0.75) / self.cfg["reach"]
                if offset <= 1.0:
                    self._swing_return(now, world, (sx, sy), power, offset)
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
