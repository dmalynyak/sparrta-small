import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config as C


# =============================================================================
# 1. Drawing one shape
# =============================================================================
def shape_polygon(shape, cx, cy, s):
    """Return the points of a shape that fits in a square of side s centred at (cx, cy).
    circle/square are drawn with PIL primitives, so they return a bounding box instead."""
    h = s / 2
    if shape in ("circle", "square"):
        return [cx - h, cy - h, cx + h, cy + h]
    if shape == "triangle":                       # pointing up, fills the box
        return [(cx, cy - h), (cx - h, cy + h), (cx + h, cy + h)]
    if shape == "star":                           # 5-pointed star
        pts = []
        for i in range(10):
            r = h if i % 2 == 0 else h * 0.45
            a = -np.pi / 2 + i * np.pi / 5
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        return pts
    if shape == "cross":                          # plus sign, returned as two rectangles
        t = s * 0.17                              # half the arm thickness
        return [[cx - h, cy - t, cx + h, cy + t], [cx - t, cy - h, cx + t, cy + h]]
    raise ValueError(shape)


def draw_shape(draw, shape, cx, cy, s, fill):
    p = shape_polygon(shape, cx, cy, s)
    if shape == "circle":
        draw.ellipse(p, fill=fill)
    elif shape == "square":
        draw.rectangle(p, fill=fill)
    elif shape == "cross":
        for rect in p:
            draw.rectangle(rect, fill=fill)
    else:
        draw.polygon(p, fill=fill)


def shape_mask(shape, cx, cy, s, size):
    """Boolean mask (size x size) of where the shape is. Used to measure overlap."""
    m = Image.new("1", (size, size), 0)
    draw_shape(ImageDraw.Draw(m), shape, cx, cy, s, 1)
    return np.array(m, dtype=bool)


# =============================================================================
# 2. Choosing where things go (the actual relation rules)
# =============================================================================
def sample_layout(relation, rng):
    """Sample sizes and positions (in final-image pixels) for one image."""
    W = C.CANVAS
    jit = lambda: rng.uniform(1 - C.SIZE_JITTER, 1 + C.SIZE_JITTER)
    scale = {"left": 1.0, "right": 1.0, "front": C.FRONT_SCALE, "back": C.BACK_SCALE}[relation]

    s_ref = C.BASE_SIZE * W * jit()
    s_tgt = C.BASE_SIZE * W * scale * jit()
    h_ref, h_tgt = s_ref / 2, s_tgt / 2            # half-extents (shapes fill their box)

    # reference: near the centre, +-15% of the canvas
    rx = W / 2 + rng.uniform(-C.REF_POS_JITTER, C.REF_POS_JITTER) * W
    ry = W / 2 + rng.uniform(-C.REF_POS_JITTER, C.REF_POS_JITTER) * W

    if relation in ("left", "right"):
        gap = rng.uniform(*C.LR_GAP) * W
        dist = h_ref + h_tgt + gap                 # centre distance > sum of half-extents
        dx = -dist if relation == "left" else dist
        dy = rng.uniform(-C.LR_VERTICAL_JITTER, C.LR_VERTICAL_JITTER) * W
    else:
        hs = h_ref + h_tgt
        dx = rng.uniform(-C.FB_DX_JITTER, C.FB_DX_JITTER) * hs
        if relation == "front":                    # lower in frame (image y grows downwards)
            dy = rng.uniform(*C.FRONT_DY) * hs
        else:                                      # higher in frame
            dy = -rng.uniform(*C.BACK_DY) * hs

    return dict(ref_cx=rx, ref_cy=ry, ref_size=s_ref,
                tgt_cx=rx + dx, tgt_cy=ry + dy, tgt_size=s_tgt, dx=dx, dy=dy)


def inside_canvas(cx, cy, s):
    """Check 1: the object's bounding box stays EDGE_MARGIN away from every border."""
    m, W, h = C.EDGE_MARGIN * C.CANVAS, C.CANVAS, s / 2
    return cx - h >= m and cy - h >= m and cx + h <= W - m and cy + h <= W - m


def lr_separated(L):
    """Check 2 (left/right only): horizontal centre distance > sum of half-extents + margin."""
    min_gap = C.LR_GAP[0] * C.CANVAS
    return abs(L["dx"]) > L["ref_size"] / 2 + L["tgt_size"] / 2 + min_gap - 1e-6


# =============================================================================
# 3. Colours and background
# =============================================================================
def pick_colors(pool, rng):
    """Train/val pool uses the 4 training colours, test pool the 2 held-out colours.
    Reference and target always get different colours (otherwise occlusion is invisible)."""
    def allowed(role):
        test = C.TEST_COLORS[role]
        return test if pool == "test" else [c for c in C.PALETTE if c not in test]

    ref_name = rng.choice(allowed("reference"))
    tgt_name = rng.choice([c for c in allowed("target") if c != ref_name])
    return str(ref_name), str(tgt_name)


def jitter_color(name, rng):
    base = np.array(C.PALETTE[name])
    return tuple(int(v) for v in np.clip(base + rng.integers(-C.COLOR_JITTER, C.COLOR_JITTER + 1, 3), 0, 255))


def make_background(rng):
    """Pale random colour + linear gradient in a random direction."""
    W = C.CANVAS
    c0 = rng.uniform(*C.BG_BASE_RANGE, size=3)
    c1 = np.clip(c0 + rng.uniform(-C.BG_GRADIENT, C.BG_GRADIENT, size=3), 0, 255)
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:W, 0:W] / (W - 1)
    t = (np.cos(ang) * (xx - 0.5) + np.sin(ang) * (yy - 0.5)) / np.sqrt(0.5) + 0.5  # 0..1
    bg = c0 + (c1 - c0) * np.clip(t, 0, 1)[..., None]
    return bg


# =============================================================================
# 4. One complete image
# =============================================================================
def render(L, ref_shape, tgt_shape, ref_rgb, tgt_rgb, relation, rng):
    S, W = C.SUPERSAMPLE, C.CANVAS
    bg = Image.fromarray(make_background(rng).astype(np.uint8)).resize((W * S, W * S), Image.BILINEAR)
    draw = ImageDraw.Draw(bg)

    ref = (ref_shape, L["ref_cx"] * S, L["ref_cy"] * S, L["ref_size"] * S, ref_rgb)
    tgt = (tgt_shape, L["tgt_cx"] * S, L["tgt_cy"] * S, L["tgt_size"] * S, tgt_rgb)
    # drawing order = occlusion: whatever is drawn second is on top
    order = [tgt, ref] if relation == "back" else [ref, tgt]
    for shape, cx, cy, s, rgb in order:
        draw_shape(draw, shape, cx, cy, s, rgb)

    img = np.asarray(bg.resize((W, W), Image.LANCZOS), dtype=np.float32)   # anti-aliased
    img += rng.normal(0, C.BG_NOISE_STD, img.shape)                         # mild noise
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def occlusion_stats(L, ref_shape, tgt_shape):
    """Fraction of the reference / target hidden by the other object (measured on masks)."""
    S, size = C.SUPERSAMPLE, C.CANVAS * C.SUPERSAMPLE
    mr = shape_mask(ref_shape, L["ref_cx"] * S, L["ref_cy"] * S, L["ref_size"] * S, size)
    mt = shape_mask(tgt_shape, L["tgt_cx"] * S, L["tgt_cy"] * S, L["tgt_size"] * S, size)
    overlap = (mr & mt).sum()
    return int(overlap), overlap / mr.sum(), overlap / mt.sum()


def make_one(relation, pool, rng):
    ref_shape = str(rng.choice(C.REF_SHAPES))
    tgt_shape = str(rng.choice(C.TGT_SHAPES))
    ref_c, tgt_c = pick_colors(pool, rng)

    for attempt in range(1, C.MAX_TRIES + 1):
        L = sample_layout(relation, rng)
        if not (inside_canvas(L["ref_cx"], L["ref_cy"], L["ref_size"]) and
                inside_canvas(L["tgt_cx"], L["tgt_cy"], L["tgt_size"])):
            continue                                           # would be clipped -> resample
        overlap, ref_occ, tgt_occ = occlusion_stats(L, ref_shape, tgt_shape)
        if relation in ("left", "right"):
            if not lr_separated(L) or overlap > 0:
                continue                                       # touching -> resample
        else:
            hidden = ref_occ if relation == "front" else tgt_occ
            if not (C.OCCLUDED_FRACTION[0] <= hidden <= C.OCCLUDED_FRACTION[1]):
                continue                                       # too little / too much occlusion
        break
    else:
        raise RuntimeError(f"Could not place {relation} after {C.MAX_TRIES} tries")

    img = render(L, ref_shape, tgt_shape, jitter_color(ref_c, rng), jitter_color(tgt_c, rng), relation, rng)
    meta = dict(relation=relation, pool=pool, ref_shape=ref_shape, tgt_shape=tgt_shape,
                ref_color=ref_c, tgt_color=tgt_c, **{k: round(v, 2) for k, v in L.items()},
                overlap_px=overlap, ref_occluded=round(ref_occ, 3), tgt_occluded=round(tgt_occ, 3),
                attempts=attempt)
    return img, meta


# =============================================================================
# 5. Main loop
# =============================================================================
def main():
    rng = np.random.default_rng(C.DATASET_SEED)
    os.makedirs(C.IMAGE_DIR, exist_ok=True)

    n_test = round(C.N_PER_RELATION * C.TEST_FRACTION)          # 50 per relation
    n_trainval = C.N_PER_RELATION - n_test                      # 200 per relation

    rows, idx = [], 0
    for relation in C.RELATIONS:
        for pool, n in [("trainval", n_trainval), ("test", n_test)]:
            for _ in range(n):
                img, meta = make_one(relation, pool, rng)
                fname = f"img_{idx:04d}.png"
                img.save(os.path.join(C.IMAGE_DIR, fname))
                rows.append(dict(image_id=idx, filename=fname, **meta))
                idx += 1
        print(f"{relation:>5}: done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.DATA_DIR, "metadata.csv"), index=False)

    # save the exact settings next to the data, so it can be reproduced
    settings = {k: v for k, v in vars(C).items() if k.isupper()}
    with open(os.path.join(C.DATA_DIR, "dataset_config.json"), "w") as f:
        json.dump(settings, f, indent=2)

    print(f"\nSaved {len(df)} images to {C.IMAGE_DIR}/")
    print(df.groupby(["relation", "pool"]).size().unstack())


if __name__ == "__main__":
    main()
