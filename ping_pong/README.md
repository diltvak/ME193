

## powerup_pong/ — ping pong with minifigure powerups

`first_pong.py` split into modules, plus two once-per-game powerups triggered by showing a LEGO
minifigure to the camera (YOLO model in `weights.pt`, classes `Yellow MiniFig` / `Green MiniFig`).

```
python -m powerup_pong [--arm right|left] [--difficulty easy|medium|hard] [--points 7] [--no-imu] [--no-powerups]
```

- **Yellow → brick wall**: a wall rises at the end of the table and returns the ball with no swing.
- **Green → ice storm**: the arena freezes and the ball is slowed for your next 3 returns.
- Setup `M` toggles powerups; in a game `Y` / `G` fire them from the keyboard (for testing). In warm-up they recharge.

Layout: `main.py` (camera loop) · `game.py` (rally logic + powerup hooks) · `detection.py` (threaded YOLO,
debounced) · `render.py` (all drawing/HUD) · `config.py` · `powerups/` (`base.py` framework, one file per powerup).
To add a powerup: subclass `Powerup`, then add it to `ALL_POWERUPS` in `powerups/__init__.py`.
