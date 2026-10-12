"""Small OpenCV text helpers shared by the renderer and the powerups."""

import cv2


def put(img, text, x, y, scale, color, thickness=1):
    # Outline by offset copies; a thicker stroke would also widen the glyph spacing.
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_centered(img, text, y, scale, color, thickness=1):
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    put(img, text, (img.shape[1] - width) // 2, y, scale, color, thickness)
