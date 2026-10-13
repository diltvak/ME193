"""AprilTag (tag36h11) reader for the setup screen: tags choose the difficulty and start the game.

Each frame, `update(frame)` returns the tag IDs that were *newly confirmed* (seen in
TAG_CONFIRM_FRAMES consecutive frames). Holding a tag up doesn't repeat; it has to leave view first.
"""

from collections import defaultdict

import cv2

from powerup_pong.config import TAG_CONFIRM_FRAMES, TAG_DETECT_WIDTH, TAG_EVERY_N


class TagReader:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self._streak = defaultdict(int)
        self._n = 0
        self.frame_w = self.frame_h = 1
        self.visible = {}   # tag id -> 4x2 corner array (pixels of the frame passed to update)

    def update(self, frame):
        """Detect tags in an unflipped BGR frame; returns IDs that just got confirmed."""
        self._n += 1
        if self._n % TAG_EVERY_N:
            return []   # detect on every Nth frame only; tags are held up for far longer than that
        h, w = frame.shape[:2]
        scale = min(1.0, TAG_DETECT_WIDTH / w)   # small frames detect much faster, and tags are big
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        self.frame_h, self.frame_w = small.shape[:2]
        corners, ids, _ = self._detector.detectMarkers(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        self.visible = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)} if ids is not None else {}
        confirmed = []
        for tag in set(self.visible) | set(self._streak):
            self._streak[tag] = self._streak[tag] + 1 if tag in self.visible else 0
            if self._streak[tag] == TAG_CONFIRM_FRAMES:
                confirmed.append(tag)
        return confirmed

    def reset(self):
        """Forget what's been seen, so a tag still in view must be re-confirmed."""
        self._streak.clear()
        self.visible = {}
