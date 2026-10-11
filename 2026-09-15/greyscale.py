#!/usr/bin/env python3
"""Convert an image to greyscale.

Usage:
    python greyscale.py input.jpg
    python greyscale.py input.jpg output.jpg
"""

import sys
from pathlib import Path

from PIL import Image


def to_greyscale(input_path: str, output_path: str | None = None) -> str:
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_greyscale")
    else:
        output_path = Path(output_path)

    with Image.open(input_path) as img:
        grey = img.convert("L")
        grey.save(output_path)

    return str(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python greyscale.py <input_image> [output_image]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    result = to_greyscale(src, dst)
    print(f"Saved greyscale image to {result}")
