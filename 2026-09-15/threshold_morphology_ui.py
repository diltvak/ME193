#!/usr/bin/env python3
"""Interactive threshold + binary morphology tool for greyscale images.

Extends threshold_ui.py: after thresholding, the binary image can be
eroded and/or dilated with a live-adjustable kernel size and iteration
count. Choose whether erosion or dilation runs first (opening vs.
closing) to see how the two classic morphological operations differ.

Usage:
    python threshold_morphology_ui.py [input_image]
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PIL import Image, ImageChops, ImageFilter, ImageTk

MAX_DISPLAY_SIZE = 700
KERNEL_MIN = 1
KERNEL_MAX = 15
ITER_MAX = 10

OPENING = "Erode then Dilate (opening)"
CLOSING = "Dilate then Erode (closing)"


def apply_threshold(image: Image.Image, threshold: int) -> Image.Image:
    grey = image.convert("L")
    return grey.point(lambda pixel: 255 if pixel < threshold else 0)


def apply_morphology(
    image: Image.Image,
    kernel_size: int,
    erosion_iterations: int,
    dilation_iterations: int,
    order: str,
) -> Image.Image:
    def erode(img: Image.Image) -> Image.Image:
        for _ in range(erosion_iterations):
            img = img.filter(ImageFilter.MinFilter(kernel_size))
        return img

    def dilate(img: Image.Image) -> Image.Image:
        for _ in range(dilation_iterations):
            img = img.filter(ImageFilter.MaxFilter(kernel_size))
        return img

    if order == CLOSING:
        return erode(dilate(image))
    return dilate(erode(image))


class ThresholdMorphologyApp:
    def __init__(self, root: tk.Tk, image_path: str | None):
        self.root = root
        self.root.title("Threshold + Morphology Tool")

        self.source_image: Image.Image | None = None
        self.display_photo: ImageTk.PhotoImage | None = None

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill=tk.X)
        controls.columnconfigure(1, weight=1)

        top_row = ttk.Frame(controls)
        top_row.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Button(top_row, text="Open Image...", command=self.open_image).pack(side=tk.LEFT)

        ttk.Label(top_row, text="Order:").pack(side=tk.LEFT, padx=(16, 4))
        self.order_var = tk.StringVar(value=OPENING)
        order_box = ttk.Combobox(
            top_row,
            textvariable=self.order_var,
            values=[OPENING, CLOSING],
            state="readonly",
            width=24,
        )
        order_box.pack(side=tk.LEFT)

        self.outline_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top_row,
            text="Show Outline (Morphology − Original)",
            variable=self.outline_var,
        ).pack(side=tk.LEFT, padx=(16, 0))

        self.threshold_var = tk.IntVar(value=128)
        self.threshold_label = self._add_slider(
            controls, row=1, text="Threshold", variable=self.threshold_var,
            from_=0, to=255, on_change=self.on_threshold_change,
        )

        self.kernel_var = tk.IntVar(value=3)
        self.kernel_label = self._add_slider(
            controls, row=2, text="Kernel Size", variable=self.kernel_var,
            from_=KERNEL_MIN, to=KERNEL_MAX, on_change=self.on_kernel_change,
        )

        self.erosion_var = tk.IntVar(value=0)
        self.erosion_label = self._add_slider(
            controls, row=3, text="Erosion Iterations", variable=self.erosion_var,
            from_=0, to=ITER_MAX, on_change=self.on_erosion_change,
        )

        self.dilation_var = tk.IntVar(value=0)
        self.dilation_label = self._add_slider(
            controls, row=4, text="Dilation Iterations", variable=self.dilation_var,
            from_=0, to=ITER_MAX, on_change=self.on_dilation_change,
        )

        self.recalculate_button = ttk.Button(
            controls, text="Recalculate", command=self.render
        )
        self.recalculate_button.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self.canvas = tk.Label(root, background="#808080")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        if image_path:
            self.load_image(image_path)

    def _add_slider(
        self,
        parent: ttk.Frame,
        row: int,
        text: str,
        variable: tk.IntVar,
        from_: int,
        to: int,
        on_change,
    ) -> ttk.Label:
        ttk.Label(parent, text=text, width=16).grid(row=row, column=0, sticky="w", pady=2)
        slider = ttk.Scale(
            parent,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=on_change,
        )
        slider.grid(row=row, column=1, sticky="ew", padx=10)
        value_label = ttk.Label(parent, text=f"{variable.get()}", width=4)
        value_label.grid(row=row, column=2, sticky="w")
        return value_label

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

    def on_threshold_change(self, _value: str) -> None:
        # Only update the label live; the image recalculates on button press.
        self.threshold_label.config(text=f"{self.threshold_var.get()}")

    def on_kernel_change(self, _value: str) -> None:
        # Kernel size must be odd for PIL's Min/MaxFilter; snap the slider.
        raw = self.kernel_var.get()
        snapped = max(KERNEL_MIN, raw - 1 if raw % 2 == 0 else raw)
        if snapped != raw:
            self.kernel_var.set(snapped)
        self.kernel_label.config(text=f"{snapped}")

    def on_erosion_change(self, _value: str) -> None:
        self.erosion_label.config(text=f"{self.erosion_var.get()}")

    def on_dilation_change(self, _value: str) -> None:
        self.dilation_label.config(text=f"{self.dilation_var.get()}")

    def render(self) -> None:
        if self.source_image is None:
            return

        thresholded = apply_threshold(self.source_image, self.threshold_var.get())
        processed = apply_morphology(
            thresholded,
            kernel_size=self.kernel_var.get(),
            erosion_iterations=self.erosion_var.get(),
            dilation_iterations=self.dilation_var.get(),
            order=self.order_var.get(),
        )

        if self.outline_var.get():
            # Boundary extraction: what morphology added/removed relative to
            # the original binary image (clipped at 0, so only the positive
            # difference survives).
            processed = ImageChops.subtract(processed, thresholded)

        display = processed.copy()
        display.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE))
        self.display_photo = ImageTk.PhotoImage(display)
        self.canvas.config(image=self.display_photo)


if __name__ == "__main__":
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    if initial_path and not Path(initial_path).exists():
        print(f"File not found: {initial_path}")
        sys.exit(1)

    root = tk.Tk()
    app = ThresholdMorphologyApp(root, initial_path)
    root.mainloop()
