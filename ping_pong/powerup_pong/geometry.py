"""Projection from table-space world coordinates to the game panel."""

from powerup_pong.config import G_OFF, VP


def proj(X, Y, z):
    """Normalized (0-1) panel position and perspective scale of a world point."""
    k = 1.0 / z
    return VP[0] + X * k, VP[1] + (G_OFF - Y) * k, k


def to_px(X, Y, z, w, h):
    """Pixel position (x, y) of a world point on a w x h panel."""
    sx, sy, _ = proj(X, Y, z)
    return int(sx * w), int(sy * h)
