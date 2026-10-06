# Second MQTT Motor Control

Subscribes to the MQTT topic `position` (public broker `test.mosquitto.org`) and
drives two motors through a Cytron Maker Drive so the robot moves toward
`(x, y) = (0, 0)` using proportional control.

Messages are JSON:

```json
{"x": 0.527, "y": -0.891}
{"x": 0.527, "y": -0.891, "theta": 1.57}
```

`theta` (heading in radians, 0 = +x axis, counter-clockwise positive) is
optional. Without it, heading is estimated from the direction of the last
significant movement, so the robot always creeps forward (never turns purely in
place) and drives straight until it has moved enough to know where it faces.

## Wiring

| Motor | Pins |
|-------|------|
| Left  | 5 & 6 |
| Right | 10 & 11 |

If a motor spins the wrong way, flip `LEFT_SIGN` / `RIGHT_SIGN` in
`python/main.py`. Gains and speed limits are at the top of that file.

## LED matrix

The onboard 8x13 matrix shows the latest received `x` as an integer percentage
(`x * 100`, clamped to -99..99), e.g. `x = -0.527` shows `-53`.

When no message has arrived for `STALE_S` seconds (or none has arrived since
startup), the number is replaced by a lit outer ring around the matrix edge.
The number returns as soon as messages resume.

## Safety

The motors stop when the robot is within `DEADBAND` of the origin, when no
message has arrived for `STALE_S`, and (on the MCU) when no motor command has
arrived for 500 ms, e.g. if the Python side crashes.
