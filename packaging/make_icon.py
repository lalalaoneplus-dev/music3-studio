#!/usr/bin/env python3
"""Music3 Studio app mark: deep violet rounded square, five white waveform bars, an amber dot.

    python3 packaging/make_icon.py

Writes Music3Studio.icns (bundle icon) and Music3Studio_256.png (window icon) next to this file.
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
S = 1024
INSET = round(0.075 * S)
R = S - 2 * INSET
VIOLET, WHITE, AMBER = "#2E1F6B", "#FFFFFF", "#F4A261"


def mark() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([INSET, INSET, S - INSET, S - INSET], radius=0.225 * R, fill=VIOLET)
    heights = (0.22, 0.46, 0.62, 0.40, 0.28)
    bar_w = 0.085 * R
    gap = 0.055 * R
    total = len(heights) * bar_w + (len(heights) - 1) * gap
    x = INSET + (R - total) / 2
    cy = INSET + 0.52 * R
    for h in heights:
        half = h * R / 2
        d.rounded_rectangle([x, cy - half, x + bar_w, cy + half], radius=bar_w / 2, fill=WHITE)
        x += bar_w + gap
    r = 0.055 * R
    cx, cy2 = INSET + 0.80 * R, INSET + 0.80 * R
    d.ellipse([cx - r, cy2 - r, cx + r, cy2 + r], fill=AMBER)
    return img


def main() -> None:
    img = mark()
    img.resize((256, 256), Image.LANCZOS).save(HERE / "Music3Studio_256.png")
    with tempfile.TemporaryDirectory() as td:
        iconset = Path(td) / "Music3Studio.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            img.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            img.resize((size * 2, size * 2), Image.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(HERE / "Music3Studio.icns")], check=True)
    print("wrote", HERE / "Music3Studio.icns")


if __name__ == "__main__":
    main()
