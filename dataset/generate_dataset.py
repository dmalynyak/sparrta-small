from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SEED = 2026
CANVAS = 224  # image size in pixels
OUT_DIR = Path("dataset")

RELATIONS = ["left", "right", "front", "back"]
TV_PER_RELATION = 200  # 200 images per relation in trainval
TEST_PER_RELATION = 50 # 50 images per relation in test

REF_SHAPES = ["circle", "square"]
TGT_SHAPES = ["triangle", "star", "cross"]

COLORS = {
    "red":    (215, 45, 45),
    "green":  (40, 155, 65),
    "blue":   (45, 85, 215),
    "yellow": (225, 185, 20),
    "purple": (140, 55, 180),
    "orange": (255, 165, 0),
}
TEST_COLORS = ["purple", "orange"]

BASE_SIZE = 0.18 * CANVAS
SCALE = {"left": 1.0, "right": 1.0, "front": 1.5, "back": 0.65}
SIZE_JITTER = 0.10
REF_JITTER = 0.10 * CANVAS
BORDER_GAP = 0.04 * CANVAS
OBJECTS_GAP = 0.05 * CANVAS


def draw_shape(draw, shape, cx, cy, size, color):
    # x -   y |
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
        t = 0.17 * size
        draw.rectangle([cx - h, cy - t, cx + h, cy + t], fill=color)
        draw.rectangle([cx - t, cy - h, cx + t, cy + h], fill=color)


def inside(cx, cy, size):
    h = size / 2
    return BORDER_GAP <= min(cx, cy) - h and max(cx, cy) + h <= CANVAS - BORDER_GAP


def place(relation, rng):
    """returns (ref_x, ref_y, ref_size), (tgt_x, tgt_y, tgt_size)"""
    ref_size = BASE_SIZE * rng.uniform(1 - SIZE_JITTER, 1 + SIZE_JITTER)
    tgt_size = BASE_SIZE * SCALE[relation] * rng.uniform(1 - SIZE_JITTER, 1 + SIZE_JITTER)
    rx, ry = CANVAS / 2 + rng.uniform(-REF_JITTER, REF_JITTER, size=2)
    half_sum = (ref_size + tgt_size) / 2

    if relation in ("left", "right"):
        max_dx = CANVAS - BORDER_GAP - tgt_size / 2 - rx if relation == "right" else rx - BORDER_GAP - tgt_size / 2
        dx = rng.uniform(half_sum + OBJECTS_GAP, max_dx)  # anywhere between min gap and border
        dx = -dx if relation == "left" else dx
        dy = rng.uniform(-0.04, 0.04) * CANVAS          # up/down jitter
    else:                                               # overlap, up to 0.3
        dx = rng.uniform(-0.3, 0.3) * half_sum
        if relation == "front":
            dy = rng.uniform(0.45, 0.85) * half_sum     # lower
        else:
            dy = -rng.uniform(0.65, 0.9) * half_sum     # higher

    tx, ty = rx + dx, ry + dy
    assert inside(rx, ry, ref_size) and inside(tx, ty, tgt_size), f"object out of bounds, relation: {relation}, ref: ({rx:.1f}, {ry:.1f}, {ref_size:.1f}), tgt: ({tx:.1f}, {ty:.1f}, {tgt_size:.1f})"
    return (rx, ry, ref_size), (tx, ty, tgt_size)


def make_image(relation, split, rng):
    palette = TEST_COLORS if split == "test" else [c for c in COLORS if c not in TEST_COLORS]
    ref_color, tgt_color = rng.choice(palette, size=2, replace=False)
    ref_pos, tgt_pos = place(relation, rng)
    ref = (rng.choice(REF_SHAPES), *ref_pos, COLORS[ref_color])
    tgt = (rng.choice(TGT_SHAPES), *tgt_pos, COLORS[tgt_color])

    bg_color = tuple(rng.integers(195, 241, size=3))
    img = Image.new("RGB", (CANVAS, CANVAS), bg_color)
    draw = ImageDraw.Draw(img)
    for obj in ([tgt, ref] if relation == "back" else [ref, tgt]):
        draw_shape(draw, *obj)
    return img


def main():
    rng = np.random.default_rng(SEED)
    for split, n in [("trainval", TV_PER_RELATION), ("test", TEST_PER_RELATION)]:
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
