"""Powerup registry. To add a powerup, write its module and add the class to ALL_POWERUPS."""

from powerup_pong.powerups.base import Powerup, PowerupManager
from powerup_pong.powerups.brick_wall import BrickWall
from powerup_pong.powerups.ice_storm import IceStorm

ALL_POWERUPS = (BrickWall, IceStorm)


def make_powerups():
    """A fresh set of unused powerups, one of each registered kind (call once per game)."""
    return [cls() for cls in ALL_POWERUPS]
