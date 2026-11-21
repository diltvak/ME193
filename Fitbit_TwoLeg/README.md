# Two-leg IMU (two Pico Ws)

Same hardware as `../Fitbit` (Pico 2 W + MPU6050 at 100 Hz over Wi-Fi), but with **one Pico on each leg**.
A single PC program receives both streams at once, plots them on a shared time axis, and records them.

| File | Role |
|---|---|
| `pico_main.py` | Firmware. Flash the **same file** to both Picos as `main.py`; set `LEG = "left"` / `"right"` (printouts only). LED on = streaming to the PC. |
| `receiver.py` | PC GUI: connects to both Picos, live plots, Record / Pause / Stop. |
| `receiver_labeled.py` | Same GUI plus route labeling (Start, Waypoints 1-4, End) and a merged `labeled.csv` for training. |
| `fake_pico.py` | Pretends to be both Picos on this computer, replaying a recorded CSV, so you can test without hardware. |
| `navigator/` | Walking directions to the route's destination from both IMUs, built to generalise to other walkers. See [navigator/README.md](navigator/README.md). |

## Run

```
pip3 install -r requirements.txt
python3 receiver.py                                   # type both IPs in the GUI (remembered next time)
python3 receiver.py --left 10.0.0.251 --right 10.0.0.252
```

Each Pico prints its IP over USB serial at boot (`[left leg] Pico IP address: ...`). Close the old
`pc_receiver.py` / `step_nn.live` first: they use the same UDP port 5005.

**No hardware?** In one terminal `python3 fake_pico.py` (add `--loss 0.05` to simulate Wi-Fi drops), in another
`python3 receiver.py --left 127.0.0.1:5006 --right 127.0.0.1:5007`.

## How it works

- The PC sends `hello` once a second to each Pico from one UDP socket; each Pico streams back to it and the packets
  are sorted into left/right by sender address. Packet format is unchanged from `../Fitbit/pico_main.py`.
- **Shared time base.** Each Pico has its own clock. Each one is mapped onto PC time using the least-delayed packet seen
  (smallest receive time minus Pico timestamp), with a small allowance for crystal drift. In testing with
  `fake_pico.py` the two legs lined up to within about 12 ms; over real Wi-Fi expect roughly 10-20 ms.
  That is fine for comparing gait phase (a stride is about 1 s).
- **Recording** creates `recordings/<timestamp>/left.csv` and `right.csv`. Both use the old
  `t_s,pc_time,ax_g,...,temp_c` columns and the same `t_s = 0`, so the `step_nn` / `activity` loaders read each leg as-is.
  A leg that is not connected still gets a file (header only).
- Plots show any channel (ax..gz, temp, |a|, |gyro|) for both legs overlaid; tick boxes hide a leg, "Freeze plot" stops scrolling.

## Labeling route data for training (`receiver_labeled.py`)

```
python3 receiver_labeled.py --left 10.0.0.251 --right 10.0.0.252
```

1. Connect both Picos and type a trial name (used in the folder name).
2. At the start point press **Space** (or the green Start button). This starts recording.
3. Press **Space** at each waypoint and at the end point. Only the next point in order can be marked
   (the green button), and **Undo last label** removes a mistake.
4. End stops recording 1.5 s later and writes the files (untick *Auto-stop at End* to keep recording).

Each walk is saved to `recordings/<trial>_<timestamp>/`:

| File | Contents |
|---|---|
| `labeled.csv` | **Training file.** Both legs resampled onto one 100 Hz grid plus labels (columns below). |
| `labels.csv` | One row per button press: `t_s,label,pc_time` |
| `left.csv`, `right.csv` | Raw samples per leg (old `pc_receiver.py` format) |

`labeled.csv` columns: `t_s`, `left_ax..left_gz`, `left_valid`, `right_ax..right_gz`, `right_valid`,
`label` (`start`/`wp1`..`wp4`/`end` on the row nearest the press, else blank), `last_point`, `next_point`,
`segment` (0 before start, 1 = start->wp1 ... 5 = wp4->end, 6 after end), `segment_progress` (0-1 through the current segment, by time).
Where a leg has no data for more than 0.1 s (lost packets, pause) its values are blank and `*_valid` is 0.

If you fix `labels.csv` by hand, regenerate the merged file with
`python3 receiver_labeled.py --rebuild recordings/<folder>`.
