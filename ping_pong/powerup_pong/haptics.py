"""Hit classification and Double Motor vibration patterns."""

import queue
import random
import threading
import time

from powerup_pong.config import EDGE_OFFSET, EDGE_TIMING, SMASH_DPS


def classify_hit(power, offset, timing_error):
    """Label a return 'smash', 'edge' or 'block'.

    power: swing strength in deg/s; offset: paddle-to-ball distance / reach;
    timing_error: swing-vs-arrival error / hit window. Edge hits (poorly centred or badly timed)
    take priority, then a fast swing is a smash, and anything gentler is a block.
    """
    if offset > EDGE_OFFSET or timing_error > EDGE_TIMING:
        return "edge"
    return "smash" if power >= SMASH_DPS else "block"


def haptic_pattern(kind):
    """Motor steps (left duty, right duty, seconds) for a hit type; duty is -100..100."""
    if kind == "smash":     # sharp, high-frequency pulse: fast full-power reversals on both sides
        return [(100 * (-1) ** i, 100 * (-1) ** i, 0.025) for i in range(8)]
    if kind == "edge":      # off-axis sputter: one side at a time, irregular, with dropouts
        steps = []
        for _ in range(6):
            side = random.choice((0, 1))
            duty = random.choice((-1, 1)) * random.randint(55, 90)
            steps.append((duty if side == 0 else 0, duty if side == 1 else 0, random.uniform(0.015, 0.06)))
            steps.append((0, 0, random.uniform(0.01, 0.05)))
        return steps
    # block: dull, low-frequency, fading thump on both sides
    return [(amp * sign, amp * sign, 0.07) for amp in (45, 35, 25, 15) for sign in (1, -1)]


class Haptics:
    """Plays hit vibrations on the Double Motor from a background thread, so BLE latency
    never stalls the video loop. A no-op when there is no motor."""

    def __init__(self, motor):
        self.motor = motor
        self.enabled = motor is not None
        self.busy = False
        self.queue = queue.Queue()
        if motor is not None:
            threading.Thread(target=self._run, daemon=True).start()

    def play(self, kind):
        if self.enabled:
            self.queue.put(kind)

    def close(self):
        self.queue.put(None)

    def _drive(self, left, right):
        import legoeducation as le
        if left == right:
            self.motor.motor_set_duty_cycle(left, motor=le.MOTOR_BOTH, blocking=False)
        else:
            self.motor.motor_set_duty_cycle(left, motor=le.MOTOR_LEFT, blocking=False)
            self.motor.motor_set_duty_cycle(right, motor=le.MOTOR_RIGHT, blocking=False)

    def _run(self):
        import legoeducation as le
        while (kind := self.queue.get()) is not None:
            self.busy = True
            try:
                for left, right, seconds in haptic_pattern(kind):
                    self._drive(left, right)
                    time.sleep(seconds)
                self.motor.motor_stop(motor=le.MOTOR_BOTH, blocking=False)
            except Exception as err:   # a flaky link shouldn't take the game down
                print(f"Haptics disabled: {err}")
                self.enabled = False
                return
            finally:
                time.sleep(0.1)        # let the vibration die down before the IMU is trusted again
                self.busy = False
