#!/usr/bin/env python3
"""Interactive threshold tool for greyscale images.

Pixels below the threshold become white; pixels at or above it become
black. Adjust the threshold live with a slider.

Usage:
    python threshold_ui.py [input_image]
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

MAX_DISPLAY_SIZE = 700


def apply_threshold(image: Image.Image, threshold: int) -> Image.Image:
    grey = image.convert("L")
    return grey.point(lambda pixel: 255 if pixel < threshold else 0)


class ThresholdApp:
    def __init__(self, root: tk.Tk, image_path: str | None):
        self.root = root
        self.root.title("Threshold Tool")

        self.source_image: Image.Image | None = None
        self.display_photo: ImageTk.PhotoImage | None = None

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Open Image...", command=self.open_image).pack(side=tk.LEFT)

        self.threshold_var = tk.IntVar(value=128)
        self.slider = ttk.Scale(
            controls,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.on_slider_change,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.threshold_label = ttk.Label(controls, text="Threshold: 128", width=16)
        self.threshold_label.pack(side=tk.LEFT)

        self.canvas = tk.Label(root, background="#808080")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        if image_path:
            self.load_image(image_path)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a greyscale image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str) -> None:
        self.source_image = Image.open(path).convert("L")
        self.render()

    def on_slider_change(self, _value: str) -> None:
        self.threshold_label.config(text=f"Threshold: {self.threshold_var.get()}")
        self.render()

    def render(self) -> None:
        if self.source_image is None:
            return

        thresholded = apply_threshold(self.source_image, self.threshold_var.get())

        display = thresholded.copy()
        display.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE))
        self.display_photo = ImageTk.PhotoImage(display)
        self.canvas.config(image=self.display_photo)


if __name__ == "__main__":
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    if initial_path and not Path(initial_path).exists():
        print(f"File not found: {initial_path}")
        sys.exit(1)

    root = tk.Tk()
    app = ThresholdApp(root, initial_path)
    root.mainloop()
