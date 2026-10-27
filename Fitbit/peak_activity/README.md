# Single-axis peak-acceleration activity recognition

Classifies **standing / biking / walking / running** from **one number per window: the peak acceleration on
one accelerometer axis**. No gyro, no other axes, no machine learning. This is a separate package; nothing in
`activity/` or the Pico files was modified (it only *imports* `activity.data_io` for the CSV loader, resampler
and UDP receiver, so keep that file in place). Run every command from the `Fitbit/` folder.

```
pip install -r peak_activity/requirements.txt
python3 -m peak_activity.calibrate        # fit thresholds from your *_data_*.csv files
python3 -m peak_activity.live --ip 10.247.137.46
python3 -m peak_activity.live --ip 10.247.137.46 --plot   # + live plot window
python3 -m peak_activity.live --source replay      # replay recorded CSVs as if live (no hardware)
python3 -m peak_activity.live --source replay --plot --speed 5
```

| File | Role |
|---|---|
| `config.py` | **All tunables**: axis, peak percentile, the 3 thresholds, window, vote, Pico IP |
| `peak.py` | The algorithm: `peak_accel()`, `classify_peak()`, majority vote (~60 lines) |
| `calibrate.py` | Compares axes, fits thresholds, leave-one-block-out check, saves `models/thresholds.json` |
| `live.py` | Real-time console output from the Pico (or replay), optional `--plot` window |

Close `pc_receiver.py` before running `live` - the Pico streams to one PC at a time. Recording new data is
unchanged: use `python -m activity.collect` (same filenames, `<activity>_data_<n>.csv`), then re-run `calibrate`.

## Live plot (`--plot`)
A Qt window with the current activity in large coloured text and two plots:
- **Top**: the last 10 s of accelerometer data (dropdown to view ax / ay / az; defaults to the axis in use).
- **Bottom**: the **peak history** (last 60 s), one dot per window coloured by its raw label, over bands shaded for
  each activity with the three thresholds as dashed lines. This is exactly what the classifier sees, so you can
  watch dots drift toward a threshold before the label flips.

## How it works
1. Resample to a uniform 100 Hz and take a 2.5 s window (same as `activity/`).
2. Take the **x-axis** acceleration, subtract the window median (removes gravity), and take the
   **95th percentile of the absolute deviation** = the window's peak, in g.
3. Compare against three thresholds:

| Peak (x, g) | Label | Observed range in your recordings |
|---|---|---|
| < 0.20 | standing | 0.01 - 0.07 |
| 0.20 - 0.78 | biking | 0.33 - 0.72 |
| 0.78 - 2.12 | walking | 0.83 - 1.72 |
| >= 2.12 | running | 2.52 - 3.61 |

4. Majority vote over the last 3 windows; if the most recent second is flat, report standing immediately.

## Why x, and why heuristics instead of ML
- **Axis choice**: `calibrate` prints, for each axis, the ratio between neighbouring classes' worst-case peaks
  (>1 = no overlap). On your data x is the only axis with no overlap anywhere (worst ratio 1.15); y overlaps
  walking/biking (1.02) and z overlaps everything and clips at the sensor's +-4 g limit when running. Use
  `--axis auto` to re-pick it after recording new data.
- **Peak = percentile, not max**: the raw max of a window put the walking and running ranges on top of each other
  (3.55 vs 3.54 g) because one spike or clipped sample decides it. The 95th percentile (about the top 12 of 250
  samples) is stable and separates cleanly.
- **Median, not mean, as baseline** so gravity and strap orientation drop out of that axis.
- **Why not ML**: the input is a single number and the classes are in strict order of size, so the optimal
  classifier is just three cut points. A model on ~170 correlated windows from one session would only add the
  risk of memorising them. The thresholds are fitted automatically (minimum per-class error; ties resolved to the
  middle of the gap), so "heuristic" here does not mean hand-tuned.

## Results (and how much to trust them)
- Leave-one-block-out: **99.4 %** (1 walking window of 173 called biking). Replaying all four recordings through
  `live` at 10x speed gave the right label on every output.
- That number is optimistic: one subject, one recording per activity, validation blocks come from the same
  sessions as the training blocks (see `activity/README.md` for the same caveat).
- **Weakest link: walking vs. biking.** Their peaks are 0.72 g vs 0.83 g apart - a margin of ~0.1 g. A brisk bike
  ride or a slow shuffle could cross it. Record more of both, at several speeds, and re-run `calibrate`.
- **Running** has only 14 windows and a threshold between 1.72 and 2.52 g, so there is plenty of room, but jogging
  slowly may land in the walking band.
- **Orientation matters here.** Unlike the full pipeline, this uses a fixed physical axis, so the strap must sit
  the same way as when you calibrated. If you re-mount the IMU, run `calibrate --axis auto` on new recordings.
- **Peak size alone cannot tell activities apart that load the ankle similarly** (e.g. stairs, hopping). Adding
  a second feature - such as peak spacing (cadence) - would be the next step if you need more classes.
