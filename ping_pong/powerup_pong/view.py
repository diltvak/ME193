"""The game window: render size, game-only view and fullscreen.

The window is a normal resizable OpenCV window, so dragging its edge scales the picture with the
aspect ratio kept. Because everything is drawn relative to the panel width, the game can also be
re-rendered at a bigger size (+ / - keys) to stay sharp instead of being stretched.

Keys:  F fullscreen   V game-only view (the camera shrinks to a corner inset)   + / - bigger / smaller
"""

import cv2
import numpy as np

from powerup_pong.config import WHITE
from powerup_pong.render import make_court

MIN_WIDTH, MAX_WIDTH, STEP = 400, 1360, 80
FULLSCREEN_WIDTH = {True: 1000, False: 720}   # render width used in fullscreen, by game_only
INSET_FRAC = 0.26                             # camera inset width, as a fraction of the game width


class View:
    def __init__(self, title, width):
        self.title = title
        self.game_only = False
        self.fullscreen = False
        self.pw = self.ph = 0
        self.court = self.top = None
        self._windowed_pw = width
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        self._set_width(width)

    def _set_width(self, pw):
        self.pw = max(MIN_WIDTH, min(MAX_WIDTH, int(pw) // 8 * 8))
        self.ph = self.pw * 3 // 4
        self.court = make_court(self.pw, self.ph)
        self.top = np.empty((self.ph, self.pw, 3), np.uint8)
        if not self.fullscreen:
            cv2.resizeWindow(self.title, self.pw, self.ph if self.game_only else 2 * self.ph)

    def compose(self, bottom):
        """The window image: the game (self.top), with the camera panel below it or as an inset."""
        if not self.game_only:
            canvas = np.vstack([self.top, bottom])
            cv2.line(canvas, (0, self.ph), (self.pw, self.ph), WHITE, 2)
            return canvas
        canvas = self.top
        iw = int(self.pw * INSET_FRAC)
        ih = iw * 3 // 4
        x, y = self.pw - iw - 8, self.ph - ih - 8
        canvas[y:y + ih, x:x + iw] = cv2.resize(bottom, (iw, ih))
        cv2.rectangle(canvas, (x, y), (x + iw, y + ih), WHITE, 1)
        return canvas

    def show(self, canvas):
        cv2.imshow(self.title, canvas)

    def handle_key(self, key):
        """Apply a view key; returns True if the key was one."""
        if key == ord("f"):
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                self._windowed_pw = self.pw
            cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
            self._set_width(FULLSCREEN_WIDTH[self.game_only] if self.fullscreen else self._windowed_pw)
        elif key == ord("v"):
            self.game_only = not self.game_only
            if self.fullscreen:
                self._set_width(FULLSCREEN_WIDTH[self.game_only])
            else:
                self._set_width(self.pw)
        elif key in (ord("+"), ord("=")):
            self._set_width(self.pw + STEP)
        elif key in (ord("-"), ord("_")):
            self._set_width(self.pw - STEP)
        else:
            return False
        return True
