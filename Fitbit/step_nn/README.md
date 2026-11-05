# Neural-net step counter and distance estimator (ankle IMU, PyTorch)

Counts steps and estimates distance walked from an MPU6050 on one ankle, using a small PyTorch
network. The Pico 2 W still samples at 100 Hz and broadcasts UDP over Wi-Fi exactly as before
(`pico_main.py` is unchanged); all the machine learning runs on the PC.

- **GitHub:** https://github.com/diltvak/ME193 (folder `Fitbit/step_nn/`)
- **Video demo:** _TODO: add link_

```
pip3 install -r step_nn/requirements.txt
python3 -m step_nn.evaluate                                  # uncertainty analysis (~14 min) -> results/
python3 -m step_nn.train                                     # final model -> models/
python3 -m step_nn.live --ip 10.247.137.46 --plot            # live from the Pico on Tufts_Robot
python3 -m step_nn.live --ip 10.0.0.251 --plot            # live from the Pico on Tufts_Robot
python3 -m step_nn.live --source replay --file 33_steps --plot --speed 3   # no hardware
```

Run everything from the `Fitbit/` folder. Close `pc_receiver.py` before `live` (the Pico streams to one
PC at a time). In the plot window, `r` resets the counts for a new walk and `q` quits.

| File | Role |
|---|---|
| `config.py` | All tunables, file lists, the 64 ft constant |
| `data.py` | Loading, resampling, end-trimming, stride pseudo-labels, network input features |
| `model.py` | `StepNet` (PyTorch), augmentation, training loop |
| `counter.py` | Streaming step + distance counter (used by both `live.py` and `evaluate.py`), ridge baseline |
| `train.py` | Trains the final model on every recording |
| `evaluate.py` | Held-out and leave-one-trial-out uncertainty analysis, plots, `results/results.json` |
| `uncertainty.py` | Turns the `results.json` error statistics into the live uncertainty bands |
| `live.py` | Real-time console output and optional plot from the Pico (or replay) |

## How it works

**Sensors.** All six IMU channels (ax, ay, az, gx, gy, gz) plus two derived magnitudes (|a| and |gyro|)
go into the network, scaled by the sensor ranges. Temperature carries no step information and is not used.
It runs on the raw stream: no hand-picked axis or threshold.

**Step detection.** `StepNet` is a 15k-parameter dilated 1-D CNN (6 residual conv blocks, dilations
1-16, receptive field +/-0.62 s). At every 10 ms sample it outputs the probability that a *stride peak
of the sensor leg* is happening there. The live counter takes peaks of that probability (threshold 0.5,
0.4 s refractory period). A peak is final once 0.65 s of later data exists, so the display lags by about
0.65 s.

**Steps from strides.** One ankle sees one peak per stride (two steps). Your three hand counts (33, 29,
23 steps) gave 17, 15, 12 peaks, so steps = 2 x strides - 1: the sensor leg took the first and last step. That is the
default (`first_step_on_sensor_leg`). If a walk starts on the other leg the count can be off by one.

**Distance.** Each detected stride adds a predicted stride length; distance is the sum. The network has a
second output head that predicts stride length from the same signal (cadence and swing intensity are in
there implicitly). I also built a ridge-regression baseline on four hand-made per-stride features
(cadence, peak gz, gz peak-to-peak, 95th-percentile |a|-1). Cross-validation picked the network head
(`len_source = nn`).

**Training labels (the part that needs your attention).** Nobody counted strides in the ten trials, so
labels come from a classical detector (low-passed gz, autocorrelation period, prominence peak picking,
merging double bumps, filling strides hidden in dropped-packet gaps). It is used for *training labels only*. I
checked it three ways: against your hand counts (exact on all three: 17, 15, 12), by eye on every
trial, and by the cross-validation below. Stride length per trial = 19.5 m / (strides in the trial), i.e. every
stride in a trial has the same length, as the assignment allows.

**Data processing.** Trimming (`data.trim_ends`) keeps the walking bout plus 0.75 s of quiet either side.
These recordings start and end almost immediately, so it removes very little. The standing and cycling files are
used as negative examples, and `walking_data_1.csv` is pseudo-labelled and used for step detection only.
Augmentation: time-stretch (changes cadence), small amplitude scaling, sensor noise, simulated dropped packets.

## Uncertainty in the live display

The GUI and console show each reading with its uncertainty:
- **Distance:** 68% (1 sigma, +/-7.9%) and 95% intervals from the leave-one-trial-out analysis below, e.g.
  `18.1 m (68%: 16.8-19.7 m, 95%: 15.3-22.4 m)`. Calibrated on ~19.5 m walks; approximate for other lengths.
- **Steps:** the typical error (worst held-out miss: 2 steps) plus a signal-based range. Each counted stride carries the net's peak
  probability; strides under 0.7 are drawn as hollow orange triangles (low confidence), and sub-threshold bumps (0.2-0.5)
  as grey diamonds ("possible missed stride"). The range assumes every low-confidence stride is false / every possible miss is real.
  On `33_steps` it shows 31 steps, range 31-33, flagging the missed stopping step; the true count is 33.

## Uncertainty analysis: how good is it?

`evaluate.py` runs the real streaming counter on data the model did not train on. (Full output in
`results/results.json`; plots in `results/`.)

### 1. Step count vs your hand counts (held-out)
Trained on the 10 trials and extras only; tested on the three recordings you counted yourself, 3 random seeds.

| Recording | Hand count | Net count | Error |
|---|---|---|---|
| `33_steps` | 33 | 31 | -2 |
| `29_steps` | 29 | 29 | 0 |
| `23_steps` | 23 | 23 | 0 |

Mean absolute error **0.67 steps (2.0%)**, 6 of 9 runs exact, all within 2 steps. The result was identical
for all three seeds, so run-to-run training variation is small here. The one miss is always the same: the final
**stopping step** of `33_steps`, whose swing is gentle (about 170 dps vs about 400 for a normal stride) and gets
only 0.10 probability (`results/heldout_steps.png`). The net is least reliable at the very end of a walk.

### 2. Leave-one-trial-out (10 folds)
Each trial is counted by a net trained on the other nine (+ the three counted walks and extras).

| | Result |
|---|---|
| Stride detection (vs labels) | precision 0.993, recall 0.987, F1 0.99; exact stride count on 9 of 10 trials (trial 8 missed one; trial 9 had one miss and one extra); mean timing error 5 ms |
| Distance, neural-net head | mean abs error **6.0%** (bootstrap 95% CI 3.3-8.8%), bias -0.7%, std 7.9%, worst 12.8% |
| Distance, ridge baseline | mean abs error 8.0% (CI 4.4-11.9%), std 10.6%, worst 20.3% |

For a single new walk of about 19.5 m, the 95% prediction interval for the net's distance error is
**-19% to +18%** (about +/-3.5 m). That is wide on purpose: n = 10 and the errors are not small.

Per trial (neural net): +0.5, -8.5, -12.8, +7.5, +9.9, -0.4, +4.5, +2.3, -11.8, +2.0 %
(`results/loto_summary.png`).

### What limits the distance accuracy
- **Same IMU signal, different stride length.** Trials 3 and 4 have nearly identical cadence (1.02 s per stride) and
  swing intensity, yet trial 3 covered 64 ft in 14 strides and trial 4 in 16 (1.39 m vs 1.22 m). An ankle
  IMU cannot tell those apart, and the net is off by -12.8% and +7.5% on them. This looks like a floor of roughly
  +/-7-8% for any method that only sees this sensor, not something more training would fix.
- **Only 10 distinct walks.** The model cannot learn much beyond "faster and harder swing = longer stride", and it
  pulls extremes toward the mean (slowest trial 5 over-estimated, trial 9, one of the two fastest, under-estimated).
- **Trial 9 is doubly uncertain.** Its fastest strides fall in packet gaps; the labeler inserted one stride.
  If that is wrong, its true error moves by about 10%.
- The net beating the ridge baseline (RMSE 7.5% vs 10.1%) is suggestive, not proven, with 10 trials.

### Caveats
- One subject, one sensor mounting, one pace range (about 0.7-2.5 m/s), one session. It will not transfer to another
  person or a re-mounted sensor without new data; running, stairs and turning are untested.
- The labels in the ten trials are not hand-verified counts; they are verified indirectly (the same labeler gets all three
  hand counts exactly right). Only three recordings have true step counts.
- Distance truth is one number per trial (64 ft); there is no ground truth for distance on the 33/29/23-step walks.
- Live operation over Wi-Fi was developed against recordings (replay through the same code path). Dropped packets
  are covered by training augmentation and the evaluation, but test it with the Pico before relying on it.

## Reflection (draft - adapt to the assignment's exact questions)
- **Why a neural net here, when `peak_activity/` needed none?** The earlier task had one number per window and ordered
  classes, so three thresholds were optimal. Step detection needs the *shape* of a stride across six channels, and
  it has to work for strolling and near-running cadence at once. A fixed threshold on one axis breaks on the clipped,
  double-bumped fast trials; the net handles them (F1 0.99 on held-out trials).
- **What did the network learn that the classical detector did not?** It was trained on that detector's labels yet
  matches your hand counts on held-out walks, and it is robust to dropped packets. It did not discover anything the
  labels lacked, so it cannot beat them.
- **What was hardest?** No step labels. Making and validating pseudo-labels (and trial 9's gaps) took more effort than
  the network. The second hardest was honest evaluation with 10 distinct walks: leave-one-out, not a random split of windows.
- **Which sensors helped?** All six channels feed the net; gz is the most informative for strides. I did not run an
  ablation, so I cannot say how much the others add.
- **Next steps:** record walks at known, varied step lengths (and with a GPS or tape measure), more subjects,
  an ablation over sensors, and a better model of the stopping step.
