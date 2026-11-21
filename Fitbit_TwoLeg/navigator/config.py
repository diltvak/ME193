"""All tunables for the navigator in one place."""
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent                              # Fitbit_TwoLeg/
RECORDINGS = ROOT / "recordings"
ROUTE_FILE = PKG / "route.json"
RESULTS = PKG / "results"

LEGS = ("left", "right")
FS = 100.0                                     # nominal IMU rate (Hz)

# ---- per-leg signal processing (signals.py) --------------------------------------------
STILL_WIN = 50                                 # samples (0.5 s) used to decide "standing still"
STILL_GYRO_STD = 3.0                           # dps: every gyro axis std below this -> still
STILL_ACC_TOL = 0.08                           # g: | |a| - 1 | below this on average -> still
BIAS_ALPHA = 0.1                               # EMA weight of each new still window for gyro bias
GRAVITY_GAIN = 0.02                            # complementary filter: accel correction per sample
GRAVITY_ACC_BAND = (0.8, 1.2)                  # only trust accel as "up" when |a| is near 1 g
MAX_DT = 0.1                                   # s: a larger sample spacing is a data gap
BRIDGE_DT = 0.35                               # s: gaps shorter than this (a few lost packets) are integrated
                                               # across; longer ones reset the orientation filter

SWING_MIN_DPS = 70.0                           # a swing peak in |gyro| must exceed this ...
SWING_REL = 0.5                                # ... and this fraction of the recent typical peak
SWING_REFRACTORY = 0.45                        # s between swing peaks of the same leg
SWING_HISTORY = 8                              # peaks used for the "typical peak" estimate
AXIS_WIN = 400                                 # samples (4 s) of gyro used to find the swing axis
AXIS_MIN_STD = 30.0                            # dps: only update the axis while clearly walking

# ---- two-leg fusion (fusion.py) ----------------------------------------------------------
TICK = 0.25                                    # s: the tracker advances in ticks of this length
LATENCY = 0.4                                  # s: wait this long for late packets before using a tick
COVER_MIN = 0.5                                # fraction of a tick a leg must cover to count as present
TURN_WINDOW = 3.0                              # s: heading change summed over this window
TURN_MIN_DEG = 45.0                            # |heading change| above this is a turn
TURN_SETTLE = 0.75                             # s of low turn rate that closes a turn event
TURN_RATE_END = 15.0                           # deg/s: below this the turn is finished
CADENCE_WINDOW = 6.0                           # s of history for the cadence used to coast through gaps

# ---- route (build_route.py) ---------------------------------------------------------------
STEP_LENGTH_PER_HEIGHT = 0.415                 # step length ~ 0.415 x height (a common rule of thumb)
DEFAULT_TRAINER_HEIGHT = 1.75                  # m, height of the person who recorded the training walks
TURN_ASSOC_S = (-2.0, 6.0)                     # a turn this many s after a waypoint press belongs to it

# ---- particle filter (tracker.py) ----------------------------------------------------------
N_PARTICLES = 1500
STEP_LEN_PRIOR_SD = 0.15                       # relative uncertainty of the height-based step length
STEP_NOISE = 0.08                              # relative per-tick noise on distance walked
TURN_ANGLE_SD = 35.0                           # deg: observed vs mapped turn angle
TURN_POS_SD_MIN = 2.5                          # m: where along the route a turn can be detected
TURN_POS_SD_REL = 0.08                         # ... plus this fraction of the distance since the last turn
TURN_CLUTTER = 0.02                            # prior weight of "this turn is not on the route"
OVERSHOOT_M = 4.0                              # m a particle may walk past an untaken turn
MISS_TURN_P = 0.002                            # per tick: chance a waiting particle missed the turn
MISS_TURN_P_GAP = 0.15                         # ... when a data gap since the last turn could have hidden it
GAP_SPEED_RANGE = (0.7, 1.2)                   # coast through gaps at this fraction of recent cadence
ARRIVE_M = 3.0                                 # m from the end -> arrived
ARRIVE_STOP_M = 10.0                           # ... or this close and standing still for
ARRIVE_STOP_S = 1.5                            # this long
WRONG_TURN_P = 0.5                             # P(turn is on the route) below this -> wrong turn alert
