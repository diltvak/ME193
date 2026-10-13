

## powerup_pong/ — ping pong with minifigure powerups

`first_pong.py` split into modules, plus two once-per-game powerups triggered by showing a LEGO
minifigure to the camera (YOLO model in `weights.pt`, classes `Yellow MiniFig` / `Green MiniFig`).

```
python -m powerup_pong [--arm right|left] [--difficulty easy|medium|hard] [--points 7] [--no-imu] [--no-powerups]
```

- **Yellow → brick wall**: a wall rises at the end of the table and returns the ball with no swing.
- **Green → ice storm**: the arena freezes and the ball is slowed for your next 3 returns.
- **CPU opponent** (`opponent.py`): an animated character with limbs that tracks the ball with its eyes, swings its paddle, cheers/groans at points, flinches at smashes, shivers in the ice storm, cowers behind the wall, and talks in speech bubbles.
- **Window**: drag the edge to resize; `F` fullscreen, `V` game-only view (camera becomes an inset), `+` / `-` render size.
- Setup `M` toggles powerups; in a game `Y` / `G` fire them from the keyboard (for testing). In warm-up they recharge.

Layout: `main.py` (camera loop) · `game.py` (rally logic + powerup hooks) · `detection.py` (threaded YOLO,
debounced) · `render.py` (all drawing/HUD) · `opponent.py` (CPU character) · `view.py` (window/fullscreen) · `config.py` · `powerups/` (`base.py` framework, one file per powerup).
To add a powerup: subclass `Powerup`, then add it to `ALL_POWERUPS` in `powerups/__init__.py`.
A powerup can set `opponent_reaction` / `opponent_line` to control how the CPU reacts to it.
