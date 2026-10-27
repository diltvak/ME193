# Ankle activity recognition

Classifies **standing / walking / running / biking** in real time from the MPU6050 stream sent by
`pico_main.py`. Nothing in the existing files was modified; this is a separate `activity/` package.
Run every command from the `Fitbit/` folder.

```
pip install -r activity/requirements.txt
```

| File | Role |
|---|---|
| `config.py` | **All tunables** (sampling rate, window length/overlap, thresholds, vote length, Pico IP) |
| `data_io.py` | CSV loading, resampling to a fixed rate, UDP receiver, simulated + replay sources |
| `features.py` | `extract_features()` - add/remove features here, then retrain |
| `classifier.py` | Stage-1 standing rule, fallback decision tree, hybrid classifier, majority vote |
| `train.py` | Leave-one-session-out CV, confusion matrix, importances, saves model + config |
| `collect.py` | Guided recording of labeled CSVs from the live stream |
| `live.py` | Real-time inference with console output and optional matplotlib plot |

## 1. Collect data (optional - you already have `*_data_1.csv`)
```
python -m activity.collect --ip 10.247.137.46 --duration 60
```
Prompts for each activity, counts down, records, and writes `<activity>_data_<n>.csv` (same format as
`pc_receiver.py`) next to `pico_main.py`. **Close `pc_receiver.py` first** - the Pico streams to one PC.
Record several sessions per activity (different days, speeds, strap positions): training uses each file as a
session once a class has 3+ files.

## 2. Train
```
python -m activity.train
```
Prints per-class precision/recall, a confusion matrix, and feature importances, and saves
`activity/models/model.joblib` + `config.json`. Files are matched by name (`cycling_*` and `biking_*` both = biking).

## 3. Run live
```
python3 -m activity.live --ip 10.247.137.46            # real hardware
python3 -m activity.live --ip 10.247.137.46 --plot     # + live plot
python3 -m activity.live --source sim --plot           # no hardware: simulated data
python3 -m activity.live --source replay               # replay your recorded CSVs as if live
python3 -m activity.live --rules                       # rule-based tree, no model needed
```
With no trained model, `live` automatically uses the rule-based tree (thresholds in `config.py`, `tree_*`).
Dropped packets are interpolated up to `max_gap_s`; longer gaps skip that window; a silent stream shows
"stream lost" and resumes by itself (hellos are re-sent every second, so the Pico reconnects).

## How it works
- **Resampling**: samples are placed on a uniform 100 Hz grid by interpolation (timestamps come from the Pico's microsecond clock).
- **Windows**: 2.5 s, 50 % overlap (`window_s`, `overlap`, `fs` in `Config`).
- **Stage 1**: standing if std of |acc| < 0.05 g *and* std of |gyro| < 8 dps.
- **Stage 2**: random forest on 30 orientation-robust features (magnitudes, sorted per-axis std, peaks, FFT dominant
  frequency / fundamental power fraction / harmonic ratio, autocorrelation, kurtosis of the main gyro swing axis).
  Biking shows up as a gyro spectrum concentrated at one frequency (`g1_fund_frac` ~0.9 vs ~0.5 for walking/running).
- **Smoothing**: majority vote over the last 5 windows.

## Design notes

- **Why leave-one-session-out, not random splits.** With 50 % overlap, neighbouring windows share half their samples.
  Even non-overlapping neighbours come from the same bout, with the same speed, posture and strap tightness. A random
  split puts near-duplicates in both train and test, so accuracy measures memorisation. Holding out a whole session
  forces the model to generalise to a recording it has never seen.
- **Why pseudo-sessions.** Each activity currently has one file, so `train.py` cuts each file into 4 contiguous time
  blocks and treats each block as a session. This is better than random splits but still weaker than real separate
  recordings, which is why the cross-validation score is optimistic.
- **Why resample.** The Pico's timestamps are irregular and packets are sometimes dropped. Interpolating onto a uniform
  100 Hz grid makes the FFT and autocorrelation features comparable across recordings.
- **Why the buffering works this way.** The Pico buffers samples in a ring and sends them in small batches. On the PC,
  a dedicated receive thread keeps the socket drained, so feature extraction or plotting can't stall reception.
- **Why the standing rule comes first.** Standing has near-zero variance in both acceleration and gyro, so a simple
  threshold is reliable and the forest never has to learn it. The forest only sees moving windows.
- **How to read the numbers.**
  - The reported 99.4 % leave-one-session-out accuracy is optimistic (see Limitations).
  - The rule-tree baseline reports 100 % because its thresholds were set by eye from these same files. Treat it as a
    sanity check, not a result.
  - The simulated source tests plumbing (connection handling, gaps, smoothing). Its waveforms are rough stand-ins
    tuned toward your measured statistics, so it does not predict real-world accuracy.
- **Before trusting live output**, close `pc_receiver.py` (the Pico streams to one PC at a time), confirm the Pico IP,
  and check that the console shows `rx` counts increasing.

## Limitations / improving accuracy
- **One subject, one session per activity.** The reported cross-validation holds out time blocks *within* each
  recording, so it is optimistic - neighbouring blocks share the same speed, strap tightness and day. Expect lower
  numbers on a new session or person. Collect several sessions (and people) and the CV becomes a true leave-one-session-out.
- **Running has little data** (~13 windows, and a ~10 s dropout in `running_data_1.csv`). Record more, at several speeds.
- **Speeds/terrain**: slow jogging vs. fast walking overlap; add intermediate speeds, stairs, and non-activity motion (sitting, fidgeting) as extra classes or "other".
- **Real bike vs. air-bike** differ (real cycling has no impacts and gentler gyro; air-biking lies on your back, so gravity direction differs). Record both into the *biking* class, or split them into two classes.
- **Placement**: keep the strap position consistent (or record at several); features are orientation-robust but not immune.
- **Rule tree thresholds** were read off your single set of recordings; treat them as a starting point.
- Higher-level upgrades: longer vote/hysteresis, per-user calibration, or a small CNN once you have much more data.
