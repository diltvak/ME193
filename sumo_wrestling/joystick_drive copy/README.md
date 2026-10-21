# 🕹️ Sumo Robot

LEGO Controller levers drive the Cytron motors. The front LEGO Color Sensor watches for the
blue tape that marks the ring. When the robot **leaves the ring** it stops its wheels and
publishes `Defeat` to MQTT topic `ME193/Sumo`. The MPU6050 x-axis (**+x = forward, −x = reverse**)
separates three cases:

| Color sensor / IMU | Meaning | Action |
|---|---|---|
| Blue, then no blue; moving **forward** ≥ 80 % of the time on the tape | Left the ring forwards | Stop + `Defeat` |
| Blue, then no blue; moving **backward** ≥ 80 % of the time on the tape | Left the ring backwards | Stop + `Defeat` |
| Blue, then no blue; direction **changed** while on the tape | Near miss, recovered | Keep fighting |

## Behavior tree (py_trees, ticked at 50 Hz)

```
[-] Sumo Round                       (Sequence, no memory)
    [-] Perceive                     (Sequence)
        --> Read Sensors             LEGO snapshot + MPU6050 x accel (Bridge get_accel_x)
        --> Estimate Velocity (IMU x)  bias-corrected, leaky integral of accel x -> forward velocity
        --> Track Blue Crossing      blue seen -> accumulate time moving fwd/back;
                                     blue gone for 0.1 s -> classify: forward / backward / recovered
    [o] Decide                       (Selector, no memory)
        [-] Defeat Protocol          (Sequence)
            --> Left Arena?          condition: a forward/backward exit was classified (latched)
            --> Stop Wheels          wheels 0, pusher stopped
            --> Send 'Defeat' (MQTT) publishes "Defeat" once (QoS 1)
        --> Joystick Drive + Pusher  normal play: levers -> wheels, hand-held IMU punch -> pusher
```

Perceive always succeeds. Decide tries Defeat Protocol first; while the robot is still in the
ring `Left Arena?` fails, so the Selector falls through to Joystick Drive. Once defeated the
state latches, so the robot stays stopped until the App is restarted.

## How direction is estimated

An accelerometer only reports *changes* in speed, so `Estimate Velocity` integrates
`(accel_x − bias) × g` into a velocity estimate. To limit drift it decays with a 2 s time
constant, and whenever both wheels have been commanded to 0 for 0.3 s the robot is assumed at
rest: velocity is reset to 0 and the accel bias is re-learned. While the sensor sees blue,
`Track Blue Crossing` adds up the time spent with velocity above +0.05 m/s (forward) and below
−0.05 m/s (backward).

## Tuning (constants at the top of `python/main.py`)

`DIRECTION_FRACTION`, `MOVING_MPS`, `BLUE_LOST_S`, `VEL_LEAK_TAU_S`, `STOPPED_HOLD_S`.
Watch `arduino-app-cli app logs` — every crossing prints its on-blue time and forward/backward
percentages. If the sensor's +x points the wrong way on the chassis, negate the accel in
`ReadSensors`.

The old "reverse for 1 s when blue is seen" override was removed: it would have driven the robot
backward on every touch of the tape and masked the direction signal.

Requires `py_trees` (listed in `python/requirements.txt`).
