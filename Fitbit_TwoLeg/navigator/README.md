# Navigator: walking directions from two leg IMUs

Guides a walker from the start point to the destination of the recorded route (Start → wp1 … wp4 → End)
using only the two shin-mounted Picos. It is built to work for **people other than the one who recorded
the training walks**, even though all training data comes from one person.

```
python3 -m navigator.build_route --trainer-height 1.75     # route map from recordings/ -> navigator/route.json
python3 -m navigator.navigate --left 10.247.137.121 --right 10.247.137.102 --height 1.62 --voice
python3 -m navigator.navigate --replay recordings/training3_20261120_123004 --speed 2               # no hardware
python3 -m navigator.navigate --replay recordings/training3_20261120_123004 --other-person --speed 2
python3 -m navigator.evaluate --plot                       # accuracy report -> navigator/results/
```

Run from the `Fitbit_TwoLeg/` folder with the system `python3`. The PyQt6 in `.venv` cannot load its Qt platform
plugin on this Mac; the system install works. `evaluate --plot` also needs `matplotlib`.

**Using it with a new walker:** strap a Pico on each shin, any way up and on either leg. Connect both and enter
the walker's height. Have them stand still at the start for about 2 s (gyro calibration; Start turns green),
then press **Space**. Directions update continuously and are spoken with `--voice` (macOS `say`).

## How it works

| File | Role |
|---|---|
| `signals.py` | Per leg: gyro bias from still periods, gravity tracking (complementary filter), heading rate, swing (step) detection |
| `fusion.py` | Both legs → 0.25 s ticks (steps, heading change) and turn events; marks Wi-Fi gaps |
| `build_route.py` | Route map from the labelled walks: distance (m) and turn angle at each waypoint |
| `tracker.py` | Particle filter over (distance along route, walker's step length, turns taken) → directions |
| `core.py` | `Navigator`: samples in, directions out (shared by the GUI, replay and evaluation) |
| `navigate.py` | Live GUI (or replay): instruction banner, route map with position estimate, event log, voice |
| `evaluate.py` | Leave-one-walk-out evaluation with simulated "other people" |

**The route map** (`route.json`, built from all 7 walks):

| Waypoint | Distance from start | Then |
|---|---|---|
| wp1 | 5 m | turn RIGHT ≈ 90° (seen in 7/7 walks) |
| wp2 | 30 m | turn LEFT ≈ 90° (2/2 walks that had data there; the other 5 walks lost Wi-Fi at this spot) |
| wp3 | 45 m | turn LEFT ≈ 90° (7/7) |
| wp4 | 48 m | turn LEFT ≈ 90° (7/7) |
| End | 59 m | arrive |

Distances are stored in **meters**, never in the trainer's step counts. They come from the trainer's step count
× the trainer's step length (0.415 × height, using the `--trainer-height` you give). If you measure the route
with a tape measure, put those values in `dist_m`: they will be more accurate.

**Extra instructions:** add `"note": "..."` to any waypoint in `route.json` and it is added to that waypoint's
directions, both on screen and spoken. wp2 has `"note": "go down the stairs"`, which gives
"Turn LEFT now (wp2) and go down the stairs." `build_route` keeps notes when it rebuilds the map.

**What makes it work for someone else:**
- **Any sensor orientation.** Steps come from the leg's main rotation axis, found automatically by PCA of the
  gyro. Heading comes from the gyro projected on "up", which is tracked from the accelerometer. Neither depends
  on how the Pico is strapped on or which leg it is on.
- **Any step length.** The walker's height gives a first guess (±15%). Each particle carries its own step
  length. At every recognised turn, the particles whose step length put them at the right place gain weight,
  so the estimate calibrates to the walker within a turn or two.
- **Any pace.** Progress is measured in meters walked, not time. Slow or fast walkers both work.
- **Wi-Fi dead zones** (every training walk loses both legs for 5–15 s around wp2): the filter keeps
  estimating with the walker's cadence from just before the gap. It also allows "a turn happened in the gap",
  and corrects itself if that turn shows up later.
- **Mistakes:** a turn that is not on the route → "That left turn is not on the route. Turn back…".
  Position is frozen until the walker turns back, and a U-turn counts. Walking well past a turn →
  "You may have passed the left turn at wp2…".

## Results (`python3 -m navigator.evaluate`)

Each walk is held out, the route map is built from the other 6, and the held-out walk is replayed exactly as it
would run live. Ground truth is the waypoint button presses. Full output: `results/summary.txt`, plots:
`results/*.png`.

| Condition | Runs | Position error at waypoints (median) | Arrival announced | All turns followed | False alerts |
|---|---|---|---|---|---|
| Same person | 7 | 0.0–2.1 m | 7/7, median 1.6 s before End | 7/7 | 0 |
| Step length guessed 20% short | 7 | 0.9–4.6 m | 7/7 | 7/7 | 0 |
| Step length guessed 20% long | 7 | 0.9–3.3 m | 7/7 | 7/7 | 0 |
| Simulated other person* | 21 | 0.7–3.2 m | 21/21, median 1.8 s before End | 21/21 | 0 |

\* Each run randomly rotates each sensor, swaps the legs, makes the walk 15–20% faster or slower, drops 10% of
packets plus a 6 s dead zone, and adds a random ±15% step-length error.

**Limits.** The 90th-percentile error around wp3/wp4 is 10–20 m. That happens when a Wi-Fi gap hides a turn and
the filter briefly holds the wrong turn count. Arrival was announced up to 22 s early in the worst run. The
simulation tests sensor placement, pace, step length and packet loss. It cannot test a different *gait*
(limp, shuffle, cane), because all the data is one person. Expect somewhat worse results with real other people,
and test with one before relying on it. The system only knows this one route: a new destination needs new
labelled walks (`receiver_labeled.py`) or a hand-written `route.json`.
