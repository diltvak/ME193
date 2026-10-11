"""Q-table learning to drive a LEGO Education Double Motor robot straight.

Follows the "Q-tables" Notion page (Silly Walks example):
  * State   = yaw error (IMU yaw, reset to 0 at the start) put into 9 bins.
  * Action  = one of 5 differential speed corrections on top of BASE_SPEED:
                  left_speed  = BASE_SPEED - diff // 2
                  right_speed = BASE_SPEED + diff // 2
  * Reward  = +1 if |error| < 2 deg, 0 if |error| < 10 deg, else -1.
  * Update  = Q_new(s,a) = (1-ALPHA)*Q_old(s,a) + ALPHA*(r + GAMMA*max Q(s'))
  * Explore = epsilon-greedy; epsilon starts high and decays.

Training is split into EPISODES. An episode ends when the robot has driven
MAX_STEPS_PER_EPISODE steps or its heading error passes FAIL_ANGLE. Between
episodes the robot turns itself back to heading 0, backs up to where it
started, and turns to a random small heading error so every state gets visited.

Run: python3 RL_test.py   (Ctrl+C to quit; the Q-table is printed and saved)
"""

import json
import os
import random
import time

import legoeducation as le

# ---- Configuration ----
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = "1151"

BASE_SPEED = 40           # average forward speed (%), fixed regardless of correction
ACTION_DIFFS = [-40, -20, 0, 20, 40]   # A0..A4 speed differentials (%)
ACTION_NAMES = ["A0 -40% (right turn)", "A1 -20% (gentle right)", "A2   0% (straight)",
                "A3 +20% (gentle left)", "A4 +40% (left turn)"]

# State bin edges (deg of yaw error): 9 bins -> 8 edges (see the page's table).
STATE_EDGES = [-20, -10, -5, -2, 2, 5, 10, 20]
N_STATES = len(STATE_EDGES) + 1
N_ACTIONS = len(ACTION_DIFFS)

ALPHA = 0.3               # learning rate (smaller number)
GAMMA = 0.9               # discount factor (close to 1)
EPSILON_START = 1.0       # exploration rate; starts high...
EPSILON_DECAY = 0.93      # ...and is multiplied by this once per EPISODE
EPSILON_MIN = 0.05

N_EPISODES = 40
MAX_STEPS_PER_EPISODE = 20
FAIL_ANGLE = 30           # deg: episode ends if |error| reaches this
START_SPREAD = 15         # deg: random starting heading error is +/- this
STEP_TIME_S = 0.3         # how long each action runs before observing the result
YAW_SCALE = 0.1           # multiply raw IMU yaw by this to get degrees

TEST_ACTIONS = True       # first run each action once and print the yaw change
REVERSE_TO_START = True   # back up between episodes so the robot stays in place
TURN_SPEED = 25           # in-place turn speed (%) when resetting heading
TURN_TOLERANCE = 3        # deg: reset turn is done within this of the target
TURN_TIMEOUT_S = 8        # give up a reset turn after this long

RESUME = False            # True: load Q_FILE and keep learning from it
Q_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")

yaw_deg = 0.0             # latest yaw, updated by the notification callback


def notification_callback(data):
    global yaw_deg
    for item in le.device_notification_parser(data):
        if isinstance(item, le.ImuDeviceNotification):
            yaw_deg = item.yaw * YAW_SCALE


def wrap(angle):
    """Wrap an angle into -180..180 so crossing +/-180 isn't a jump."""
    return (angle + 180) % 360 - 180


def yaw_error():
    return wrap(yaw_deg)  # desired heading is 0


def get_state(error):
    """Bin the yaw error into state 0..8 (4 = on target)."""
    for state, edge in enumerate(STATE_EDGES):
        if error < edge:
            return state
    return N_STATES - 1


def get_reward(error):
    """Reward from the heading error magnitude after the action (checked in order)."""
    if abs(error) < 2:
        return 1.0
    if abs(error) < 10:
        return 0.0
    return -1.0


def choose_action(q_table, state, epsilon):
    """Explore with probability epsilon, otherwise exploit the best-known action."""
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    best = max(q_table[state])
    return random.choice([a for a, q in enumerate(q_table[state]) if q == best])


def run_action(doublemotor, action):
    diff = ACTION_DIFFS[action]
    doublemotor.movement_move_tank(speed_left=BASE_SPEED - diff // 2,
                                   speed_right=BASE_SPEED + diff // 2,
                                   blocking=False)
    time.sleep(STEP_TIME_S)


def turn_to_heading(doublemotor, target):
    """Spin in place until the yaw is within TURN_TOLERANCE of target.

    Positive error = right drift, which a left turn (right wheel faster) fixes.
    Returns False if it timed out (e.g. the turn direction is wrong).
    """
    start = time.time()
    while time.time() - start < TURN_TIMEOUT_S:
        err = wrap(yaw_deg - target)
        if abs(err) <= TURN_TOLERANCE:
            doublemotor.movement_stop()
            time.sleep(0.3)
            return True
        speed = TURN_SPEED if abs(err) > 15 else 15
        spin = -speed if err > 0 else speed   # spin>0: left wheel faster -> right turn
        doublemotor.movement_move_tank(speed_left=spin, speed_right=-spin, blocking=False)
        time.sleep(0.05)
    doublemotor.movement_stop()
    return False


def reset_robot(doublemotor, driven_s):
    """Return to heading 0, back up to the start, then pick a random start heading."""
    doublemotor.movement_stop()
    ok = turn_to_heading(doublemotor, 0)
    if ok and REVERSE_TO_START and driven_s > 0:
        doublemotor.movement_move_tank(speed_left=-BASE_SPEED, speed_right=-BASE_SPEED,
                                       blocking=False)
        time.sleep(driven_s)
        doublemotor.movement_stop()
        time.sleep(0.3)
        ok = turn_to_heading(doublemotor, 0)
    if ok:
        ok = turn_to_heading(doublemotor, random.uniform(-START_SPREAD, START_SPREAD))
    if not ok:
        input("Reset turn timed out - reposition the robot by hand, then press Enter...")
        doublemotor.imu_reset_yaw_axis()
        time.sleep(0.5)


def test_actions(doublemotor):
    """Run each action from a straight heading and print the resulting yaw change."""
    print("Testing each action (positive yaw change = right drift):")
    for action in range(N_ACTIONS):
        before = yaw_error()
        run_action(doublemotor, action)
        doublemotor.movement_stop()
        time.sleep(0.2)
        print(f"  {ACTION_NAMES[action]:<22} yaw change {wrap(yaw_deg - before):+6.1f} deg")
        reset_robot(doublemotor, STEP_TIME_S)
        turn_to_heading(doublemotor, 0)


def print_q(q_table, visits):
    print("state | " + " | ".join(f"{'A' + str(a):>7}" for a in range(N_ACTIONS))
          + " | visits")
    for s, row in enumerate(q_table):
        print(f"{s:>5} | " + " | ".join(f"{q:7.3f}" for q in row)
              + f" | {visits[s]:>6}")
    print("Greedy action per state: "
          + ", ".join(f"s{s}->A{row.index(max(row))}" if any(row) else f"s{s}->?"
                      for s, row in enumerate(q_table)))


def main():
    doublemotor = le.DoubleMotor()
    q_table = [[0.0] * N_ACTIONS for _ in range(N_STATES)]  # start with 0s
    visits = [0] * N_STATES
    if RESUME and os.path.exists(Q_FILE):
        with open(Q_FILE) as f:
            q_table = json.load(f)
        print(f"Loaded Q-table from {Q_FILE}")

    # try/finally so the motor is always stopped, however the script ends.
    try:
        print("Connecting to Double Motor...")
        doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
        if not doublemotor.connected:
            print("Error connecting to Double Motor.")
            return

        doublemotor.set_notification_callback(notification_callback)
        doublemotor.device_notification_request(50)
        doublemotor.imu_reset_yaw_axis()   # desired heading = 0 deg
        time.sleep(0.5)

        if TEST_ACTIONS:
            test_actions(doublemotor)

        epsilon = EPSILON_START
        for episode in range(1, N_EPISODES + 1):
            reset_robot(doublemotor, 0)
            state = get_state(yaw_error())
            total_reward = 0.0
            steps = 0
            failed = False
            print(f"\nEpisode {episode}/{N_EPISODES}  eps={epsilon:.2f}  "
                  f"start yaw={yaw_error():+.1f}")

            for step in range(1, MAX_STEPS_PER_EPISODE + 1):
                action = choose_action(q_table, state, epsilon)
                run_action(doublemotor, action)

                error = yaw_error()
                next_state = get_state(error)
                reward = get_reward(error)
                failed = abs(error) >= FAIL_ANGLE

                # A failed episode has no future, so the target is just the reward.
                future = 0.0 if failed else max(q_table[next_state])
                q_table[state][action] = ((1 - ALPHA) * q_table[state][action]
                                          + ALPHA * (reward + GAMMA * future))

                visits[state] += 1
                total_reward += reward
                steps = step
                print(f"  step {step:>2}  s={state} {ACTION_NAMES[action]:<22} "
                      f"yaw={error:+6.1f}  s'={next_state} r={reward:+.1f}")
                state = next_state
                if failed:
                    break

            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            print(f"  -> {steps} steps, total reward {total_reward:+.1f}"
                  + ("  (went off course)" if failed else ""))
            reset_robot(doublemotor, steps * STEP_TIME_S)

        print("\nDone.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print("Cleaning up...")
        print_q(q_table, visits)
        with open(Q_FILE, "w") as f:
            json.dump(q_table, f)
        if doublemotor.connected:
            try:
                doublemotor.movement_stop()
            except Exception as exc:
                print(f"Error stopping motor: {exc}")
            doublemotor.disconnect()


if __name__ == "__main__":
    main()
