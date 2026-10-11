# Controls

`pd_tracker.py` is a live PD-control demo: a Single Motor acts as a
hand-turned dial (the target), and one side of a Double Motor drives itself
to match that dial's angle. Two on-screen sliders — Kp and Kd — let you
change the controller's gains while it's running, so you can watch the
effect on rise time, overshoot, and oscillation in real time on the plot.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

pip install legoeducation matplotlib
```

`lelib.py` is already copied into this folder (see `useful libraries/` for
the source).

By default `CARD_COLOR`/`CARD_SERIAL` at the top of `pd_tracker.py` are both
`None`, meaning no connection card: the script just connects to the first
advertising Single Motor and the first advertising Double Motor it finds.
If you have more than one of either lying around, set a shared LEGO
connection card so it reconnects to the same pair every time — the same
color/serial is used for both devices (like `main.py`'s color sensor +
controller in `useful libraries/`):

```python
CARD_COLOR = le.LEGO_COLOR_BLUE
CARD_SERIAL = 1234
```

Then run it:

```bash
python3 pd_tracker.py
```

## How it works

This is a one-degree-of-freedom demo: the Single Motor's shaft is the
target angle, and `DOUBLE_MOTOR_SIDE` (one side of the Double Motor,
`le.MOTOR_LEFT` by default) is the follower. The other side of the Double
Motor is left unpowered.

At every control-loop tick:

```
error  = target_position - actual_position
speed  = Kp * error - Kd * d(actual_position)/dt
```

`speed` (clamped to ±100%) is the *only* thing being commanded — there's no
separate "move to position" command involved, so everything you see in the
plot is a direct result of the P and D terms. The loop reads both motors'
cumulative position counters (not the 0–359° absolute position) so the
error stays well-behaved even if the dial is spun through several full
turns.

The D term is deliberately "derivative on measurement" — it damps the
*follower's own* velocity, not the raw error's rate of change. If it
damped the error instead, every time you turned the dial by hand you'd
get a derivative spike from the dial's own motion (target moving), which
pushes the follower harder rather than damping it — raising Kd would make
things more jittery/overshooting instead of less. Damping the follower's
velocity directly means Kd reduces overshoot when it settles onto a
target regardless of how jerkily that target was moved to get there.

The control loop runs in a background thread as fast as the Bluetooth link
will sustain (~50 Hz target); the plot and sliders run on the main thread
and just read the shared history/gain values.

`speed` is sent straight to `motor_run()` as a signed value — confirmed on
hardware that a negative speed simply reverses whichever `direction` was
passed, so there's no need to compute a direction and pass `abs(speed)`.

## What to try

- Start with Kp around 1 and Kd at 0. Turn the dial and watch the Double
  Motor chase it, and see how much it overshoots before settling.
- Raise Kp — the response gets faster, but overshoot and oscillation get
  worse.
- With a Kp that's now oscillating, raise Kd — it damps the oscillation
  out. Too much Kd, and the response gets sluggish/jittery again (the
  derivative term is a raw frame-to-frame slope, so very large Kd will
  amplify sensor noise into jitter).
- Watch the terminal: it prints the achieved loop rate, current error, and
  commanded speed every couple of seconds, useful for seeing when the
  motor is saturating at ±100% speed instead of following the P/D law.

## Notes

- `MAX_SPEED`, `LOOP_HZ`, and the Kp/Kd slider ranges are all constants
  near the top of the file if you want to retune them for a different
  mechanical setup.
- Closing the plot window stops the control loop and the Double Motor.
