"""Powerup framework: the Powerup base class and the PowerupManager that drives them.

To add a powerup, subclass Powerup in its own file, override the hooks you need, and register
the class in powerups/__init__.py. Game and the renderer only talk to the manager, so neither
needs to change.

A powerup's lifecycle: ready -> active -> fading -> used. `finish()` moves active -> fading
(winding-down animation), and the powerup becomes used `fade_s` seconds later. Each powerup can
be used once per game; in warm-up mode the manager makes it ready again after a short pause.
"""

from powerup_pong.config import BANNER_S, PRACTICE_RESET_S

TRIGGER_PHASES = ("serve", "incoming", "returned")   # game phases in which a powerup may fire


class Powerup:
    name = "Powerup"          # long name, used in the banner
    hud_name = "POWER"        # short name, used in the HUD
    trigger_class = None      # YOLO class name that activates it
    key = None                # keyboard key that activates it (testing without the minifig)
    color = (255, 255, 255)   # BGR accent color
    banner = ""               # sub-line shown under the activation banner
    fade_s = 0.0              # how long the winding-down animation lasts after finish()

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "ready"
        self.t0 = self.t_end = 0.0   # when it activated / when it finished

    # ---- State ----

    def can_trigger(self, game, now):
        return self.state == "ready" and game.phase in TRIGGER_PHASES

    def activate(self, game, now):
        self.state, self.t0 = "active", now

    def finish(self, now):
        self.state, self.t_end = "fading", now

    def update(self, game, now):
        if self.state == "fading" and now - self.t_end >= self.fade_s:
            self.state = "used"

    # ---- Game hooks (override as needed) ----

    def flight_multiplier(self):
        """Scales the flight time of each ball the opponent launches (>1 is slower)."""
        return 1.0

    def takes_over_returns(self):
        """True if the powerup returns the ball itself, so swings are ignored."""
        return False

    def intercept(self, game, now, world):
        """Called each frame while a ball is incoming. Return True if the powerup handled the ball."""
        return False

    def on_return(self, game, now):
        """Called after every return of the ball, by a swing or by a powerup."""

    # ---- Visuals ----

    def draw_scene(self, img, game, now):
        """Drawn on the court, under the ball."""

    def draw_overlay(self, img, game, now):
        """Drawn over the scene and ball, under the paddle and HUD."""

    def hud_detail(self):
        """Short status text for the HUD row."""
        return {"ready": "ready", "active": "active", "fading": "active", "used": "used"}[self.state]


class PowerupManager:
    """Holds a game's powerups and fans triggers, game events and drawing out to them."""

    def __init__(self, powerups=()):
        self.powerups = list(powerups)
        self.banner = None    # (title, subtitle, color, time) of the latest activation

    def __iter__(self):
        return iter(self.powerups)

    # ---- Triggers ----

    def _fire(self, powerup, game, now):
        if powerup.can_trigger(game, now):
            powerup.activate(game, now)
            self.banner = (powerup.name.upper() + "!", powerup.banner, powerup.color, now)
            return True
        return False

    def handle_detection(self, game, class_name, now):
        """A minifig class appeared on camera. Returns True if it activated a powerup."""
        return any(self._fire(p, game, now) for p in self.powerups if p.trigger_class == class_name)

    def handle_key(self, game, key, now):
        return any(self._fire(p, game, now) for p in self.powerups if p.key and ord(p.key) == key)

    # ---- Game hooks ----

    def update(self, game, now):
        for p in self.powerups:
            p.update(game, now)
            if game.practice and p.state == "used" and now - p.t_end >= p.fade_s + PRACTICE_RESET_S:
                p.reset()

    def flight_multiplier(self):
        mult = 1.0
        for p in self.powerups:
            mult *= p.flight_multiplier()
        return mult

    def takes_over_returns(self):
        return any(p.takes_over_returns() for p in self.powerups)

    def intercept(self, game, now, world):
        return any(p.intercept(game, now, world) for p in self.powerups)

    def on_return(self, game, now):
        for p in self.powerups:
            p.on_return(game, now)

    # ---- Visuals ----

    def draw_scene(self, img, game, now):
        for p in self.powerups:
            p.draw_scene(img, game, now)

    def draw_overlay(self, img, game, now):
        for p in self.powerups:
            p.draw_overlay(img, game, now)

    def active_banner(self, now):
        if self.banner and now - self.banner[3] < BANNER_S:
            return self.banner
        return None
