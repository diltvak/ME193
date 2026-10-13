# Powerup Pong

Ping pong against a virtual opponent, played with real arm swings (webcam pose + optional LEGO IMU),
with powerups triggered by showing a LEGO minifigure to the camera.

## How to run

Run from the `ping_pong/` directory (the parent of this folder), not from inside `powerup_pong/`:

```bash
cd ping_pong
pip install -r requirements.txt
python -m powerup_pong
```

Options:

```bash
python -m powerup_pong [--arm right|left] [--difficulty easy|medium|hard] [--points 7]
                       [--width 560] [--no-imu] [--no-powerups] [--weights path/to/weights.pt]
                       [--name YourName] [--no-mqtt]
```

- `--no-imu` plays with the camera only (no LEGO Double Motor needed).
- `--no-powerups` starts with powerups off (toggle with `M` on the setup screen).
- `--name YourName` sets the MQTT topic `ME193/Rogers/<YourName>` (or edit `PLAYER_NAME` in `config.py`).
- `pip install paho-mqtt` is needed for the scoreboard; `--no-mqtt` turns it off.
- Grant camera permission to your terminal/IDE on macOS.

Keys: setup screen: `1/2/3` difficulty, `P` points, `A` swap arm, `H` haptics, `M` powerups, `W` warm-up,
`ENTER`/`SPACE` start, `Q` quit. In game: `ESC` back to setup, `Q` quit, and each powerup's key
(`Y`, `G`) fires it from the keyboard for testing without a minifig.

## AprilTags and the MQTT scoreboard

- **Setup screen:** hold a tag36h11 AprilTag up to the camera. IDs **10 / 11 / 12** pick easy / medium / hard
  and ID **13** starts the game (keyboard keys still work). A tag must be seen for 2 consecutive detections
  (`TAG_CONFIRM_FRAMES`; detection runs on every 2nd frame at reduced size, see `TAG_DETECT_WIDTH`/`TAG_EVERY_N`); IDs are in `DIFFICULTY_TAGS` / `START_TAG` in `config.py`. Code: `apriltags.py`.
- **Score:** the record number of continuous hits (best rally this session, as a float such as `7.0`) is published
  retained to `ME193/Rogers/<name>` on `test.mosquitto.org` every time it improves (and `0.0` on startup).
  Warm-up rallies don't count. Code: `scoreboard.py`, built on `mqttlib.py` (copied from `delivery_service`).

## Changing the LEGO cards (number and color)

Each powerup is tied to one detected minifigure class. A "card" is one powerup plus the minifig
class that triggers it. The pieces involved:

| What | Where |
|---|---|
| Class names the model can detect | `YELLOW_CLASS`, `GREEN_CLASS`, `NO_FIG_CLASS` in [config.py](config.py) |
| Color of the detection box on the webcam panel | `MINIFIG_COLORS` in [config.py](config.py) |
| Which class triggers which powerup | `trigger_class` (and test `key`) in each file in [powerups/](powerups/) |
| Which powerups exist in a game | `ALL_POWERUPS` in [powerups/\_\_init\_\_.py](powerups/__init__.py) |

**Important:** the camera can only recognize classes the YOLO model was trained on. `../weights.pt`
currently knows `Yellow MiniFig`, `Green MiniFig` and `No MiniFig`. A new color or a new number of
distinct minifigs needs a retrained model (new `weights.pt`); the steps below only wire the new
class into the game. Check the exact class names with `YOLO("weights.pt").names`.

### Change the color of an existing card
1. In `config.py`, edit the class string (e.g. `YELLOW_CLASS = "Red MiniFig"`) so it matches a class name in your weights exactly (case and spaces included).
2. Optionally update `MINIFIG_COLORS` (BGR color of the box drawn on the camera view).
3. The powerup file already imports that constant, so nothing else changes. Rename the constant too if it would be misleading (then update its imports in `config.py`, `detection.py`-adjacent code, and the powerup file; `grep -rn YELLOW_CLASS .` finds them all).

### Add a card (more than two)
1. In `config.py`, add a class constant, e.g. `BLUE_CLASS = "Blue MiniFig"`, and add it to `MINIFIG_COLORS`.
2. Create `powerups/my_powerup.py`: subclass `Powerup` and set `trigger_class = BLUE_CLASS`, plus `name`, `hud_name`, `key` (unique test key), `color`, `banner`. Override only the hooks you need (see below).
3. Register it in `powerups/__init__.py`: import the class and append it to `ALL_POWERUPS`.

### Remove a card
Delete the class from `ALL_POWERUPS` in `powerups/__init__.py`. (Optionally delete its file and constants.)

### Swap which powerup a color triggers
Swap the `trigger_class` values in the two powerup files (e.g. make `IceStorm` use `YELLOW_CLASS`).

### Detection tuning
`MINIFIG_CONF` (min confidence), `MINIFIG_CONFIRM_FRAMES` (consecutive frames before it counts), and
`MINIFIG_REARM_S` (cooldown before the same class can fire again) in `config.py`.

### Not the minifigs: the motor card
The LEGO Double Motor "card" used for the IMU/haptics is set by `MOTOR_CARD_SERIAL` (and the
`card_color` in `connect_imu()`) in `../swing.py`.

## Code structure

```
ping_pong/
  weights.pt, swing.py, pose_detection.py     shared assets / helpers used by this package
  powerup_pong/
    __main__.py        entry point: calls main()
    main.py            camera loop, menu/game/warm-up modes, wires everything together
    game.py            Game: rally state machine, scoring, hit windows (no drawing, no camera)
    paddle.py          Paddle: wrist landmarks -> smoothed paddle position/angle
    haptics.py         classify_hit() + Haptics: vibration patterns on the motor (background thread)
    apriltags.py       TagReader: tag36h11 IDs choose difficulty / start the game on the setup screen
    scoreboard.py      Scoreboard: publishes the record rally to MQTT (uses mqttlib.py)
    detection.py       MinifigDetector: threaded YOLO inference, debounced "appeared" events
    render.py          all drawing: court, ball, paddle, HUD, menu, banner, minifig boxes
    drawing.py         tiny text helpers (put, put_centered)
    geometry.py        proj()/to_px(): 3D scene -> 2D panel perspective projection
    config.py          every tunable constant (difficulty, geometry, colors, detection)
    powerups/
      __init__.py      registry: ALL_POWERUPS, make_powerups()
      base.py          Powerup base class + PowerupManager
      brick_wall.py    yellow powerup
      ice_storm.py     green powerup
```

### Design principles

- **One-way dependencies.** `config` and `geometry` are leaves. `game`, `paddle`, `haptics` and
  `detection` hold logic and never draw. `render` only draws and reads state. `main` is the only
  module that knows about all of them, so it is the only place the camera/IMU/pose libraries are wired up.
- **Game logic is separated from I/O.** `Game.update(now, swung, paddle_xy, power)` takes plain
  values (a timestamp, whether a swing happened, paddle position, swing power), so it can be tested without a camera.
- **Slow work is off the main thread.** YOLO runs in `MinifigDetector`'s thread on the newest frame
  only; haptic patterns run in `Haptics`' thread. The camera loop never blocks on them.
- **Constants live in `config.py`**, except powerup-specific tuning, which sits at the top of that
  powerup's own file so adding a powerup never means editing `config.py` (apart from its class name).
- **Powerups are plugins.** `Game` and `render` only talk to `PowerupManager`, never to a specific
  powerup, so a new powerup needs no changes to either.

### Data flow per frame

1. `main.py` reads a webcam frame, runs pose landmarks, updates `Paddle`, and reads the IMU gyro for swing power.
2. The frame is submitted to `MinifigDetector`; `poll()` returns newly appeared classes, which go to
   `PowerupManager.handle_detection()` (keyboard keys go to `handle_key()`).
3. `Game.update(...)` advances the rally, asking the manager for hooks (below) and using `haptics.classify_hit` to label hits (block/smash/edge).
4. `render.py` draws the court (via `geometry`), the paddle, powerup scene/overlay layers and HUD on the left panel and the camera view with minifig boxes on the right.

### Powerup lifecycle and hooks

State: `ready -> active -> fading -> used` (in warm-up it resets to `ready` after a pause). Override
any of these in a `Powerup` subclass; defaults do nothing:

| Hook | Purpose |
|---|---|
| `can_trigger(game, now)` / `activate(game, now)` | gate and start the effect |
| `flight_multiplier()` | slow/speed the opponent's ball |
| `takes_over_returns()` / `intercept(game, now, world)` | return the ball automatically |
| `on_return(game, now)` | react to each player return (e.g. count hits) |
| `draw_scene(img, game, now)` / `draw_overlay(...)` | draw in the 3D scene / over the whole panel |
| `hud_detail()` | extra text in the HUD |
| `finish(now)`, `fade_s` | end the effect and play a wind-down animation |
