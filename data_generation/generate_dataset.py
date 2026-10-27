"""
Synthetic dataset: one reference object + one target object, 4 spatial relations.

    python data_generation/generate_dataset.py

Writes  dataset/trainval/0000_left.png ...  and  dataset/test/0000_front.png ...
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SEED = 2026
CANVAS = 224                                   # image size in pixels
OUT_DIR = Path(__file__).resolve().parent.parent / "dataset"

RELATIONS = ["left", "right", "front", "back"]
N_PER_RELATION = 250                           # 200 go to trainval, 50 to test
N_TEST = 50

REF_SHAPES = ["circle", "square"]
TGT_SHAPES = ["triangle", "star", "cross"]

COLORS = {
    "red":    (215, 45, 45),
    "green":  (40, 155, 65),
    "blue":   (45, 85, 215),
    "yellow": (225, 185, 20),
    "purple": (140, 55, 180),
    "teal":   (20, 160, 170),
}
TEST_COLORS = ["purple", "teal"]               # never appear in trainval

BASE_SIZE = 0.14 * CANVAS                      # side of the box the shape fills
SCALE = {"left": 1.0, "right": 1.0, "front": 1.5, "back": 0.65}
SIZE_JITTER = 0.10                             # every object: size * U(0.9, 1.1)
REF_JITTER = 0.15 * CANVAS                     # reference centre: canvas centre +- 15%
MARGIN = 0.06 * CANVAS                         # objects stay this far from the border
LR_GAP = 0.02 * CANVAS                         # min empty space between left/right objects


def draw_shape(draw, shape, cx, cy, size, color):
    h = size / 2
    box = [cx - h, cy - h, cx + h, cy + h]
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        draw.polygon([(cx, cy - h), (cx - h, cy + h), (cx + h, cy + h)], fill=color)
    elif shape == "star":
        angles = -np.pi / 2 + np.arange(10) * np.pi / 5
        radii = np.where(np.arange(10) % 2 == 0, h, 0.45 * h)
        draw.polygon(list(zip(cx + radii * np.cos(angles), cy + radii * np.sin(angles))), fill=color)
    elif shape == "cross":
        t = 0.17 * size                        # half of the arm thickness
        draw.rectangle([cx - h, cy - t, cx + h, cy + t], fill=color)
        draw.rectangle([cx - t, cy - h, cx + t, cy + h], fill=color)


def background(rng):
    """Pale colour fading into another pale colour, in a random direction."""
    c0 = rng.uniform(195, 240, size=3)
    c1 = np.clip(c0 + rng.uniform(-25, 25, size=3), 0, 255)
    angle = rng.uniform(0, 2 * np.pi)
    y, x = np.mgrid[0:CANVAS, 0:CANVAS] / CANVAS - 0.5
    t = np.clip(0.5 + np.cos(angle) * x + np.sin(angle) * y, 0, 1)[..., None]
    return Image.fromarray((c0 + (c1 - c0) * t).astype(np.uint8))


def inside(cx, cy, size):
    h = size / 2
    return MARGIN <= min(cx, cy) - h and max(cx, cy) + h <= CANVAS - MARGIN


def place(relation, rng):
    """Return (ref_x, ref_y, ref_size), (tgt_x, tgt_y, tgt_size), resampling until both fit."""
    while True:
        ref_size = BASE_SIZE * rng.uniform(1 - SIZE_JITTER, 1 + SIZE_JITTER)
        tgt_size = BASE_SIZE * SCALE[relation] * rng.uniform(1 - SIZE_JITTER, 1 + SIZE_JITTER)
        rx, ry = CANVAS / 2 + rng.uniform(-REF_JITTER, REF_JITTER, size=2)
        half_sum = (ref_size + tgt_size) / 2

        if relation in ("left", "right"):
            dx = half_sum + LR_GAP + rng.uniform(0, 0.03 * CANVAS)
            dx = -dx if relation == "left" else dx
            dy = rng.uniform(-0.04, 0.04) * CANVAS          # a little up/down jitter
        else:                                               # overlap, but keep both recognisable
            dx = rng.uniform(-0.3, 0.3) * half_sum
            if relation == "front":
                dy = rng.uniform(0.45, 0.85) * half_sum     # lower (image y grows downwards)
            else:
                dy = -rng.uniform(0.65, 0.9) * half_sum     # higher

        tx, ty = rx + dx, ry + dy
        if inside(rx, ry, ref_size) and inside(tx, ty, tgt_size):
            return (rx, ry, ref_size), (tx, ty, tgt_size)


def make_image(relation, split, rng):
    palette = TEST_COLORS if split == "test" else [c for c in COLORS if c not in TEST_COLORS]
    ref_color, tgt_color = rng.choice(palette, size=2, replace=False)
    ref_pos, tgt_pos = place(relation, rng)
    ref = (rng.choice(REF_SHAPES), *ref_pos, COLORS[ref_color])
    tgt = (rng.choice(TGT_SHAPES), *tgt_pos, COLORS[tgt_color])

    img = background(rng)
    draw = ImageDraw.Draw(img)
    for obj in ([tgt, ref] if relation == "back" else [ref, tgt]):   # drawn last = on top
        draw_shape(draw, *obj)
    return img


def main():
    rng = np.random.default_rng(SEED)
    for split, n in [("trainval", N_PER_RELATION - N_TEST), ("test", N_TEST)]:
        folder = OUT_DIR / split
        folder.mkdir(parents=True, exist_ok=True)
        i = 0
        for relation in RELATIONS:
            for _ in range(n):
                make_image(relation, split, rng).save(folder / f"{i:04d}_{relation}.png")
                i += 1
        print(f"{split}: {i} images -> {folder}")


if __name__ == "__main__":
    main()
