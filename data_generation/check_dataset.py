"""
Sanity checks: LOOK at the data before training anything.

Run:  python check_dataset.py
Out:  figures/examples_trainval.png, figures/examples_test.png,
      figures/offsets.png, figures/reference_positions.png, figures/sizes.png
      + automatic checks printed to the terminal
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import config as C

COLORS = {"left": "#1f77b4", "right": "#ff7f0e", "front": "#2ca02c", "back": "#d62728"}


def example_grid(df, pool, n=6, seed=0):
    """n random images per relation, one row per relation."""
    fig, axes = plt.subplots(4, n, figsize=(n * 1.8, 4 * 1.9))
    for r, rel in enumerate(C.RELATIONS):
        sub = df[(df.relation == rel) & (df.pool == pool)].sample(n, random_state=seed)
        for c, (_, row) in enumerate(sub.iterrows()):
            ax = axes[r, c]
            ax.imshow(Image.open(os.path.join(C.IMAGE_DIR, row.filename)))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{row.ref_color[:3]}/{row.tgt_color[:3]}", fontsize=7)
            if c == 0:
                ax.set_ylabel(rel, fontsize=12, color=COLORS[rel], fontweight="bold")
    fig.suptitle(f"{pool} pool  (title = reference / target colour)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, f"examples_{pool}.png"), dpi=120)
    plt.close(fig)


def scatter_by_relation(df, x, y, title, fname, invert_y=True, center_lines=False):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for rel in C.RELATIONS:
        s = df[df.relation == rel]
        ax.scatter(s[x], s[y], s=8, alpha=0.6, c=COLORS[rel], label=rel)
    if center_lines:
        ax.axhline(0, c="gray", lw=0.5), ax.axvline(0, c="gray", lw=0.5)
    if invert_y:
        ax.invert_yaxis()                    # image coordinates: y grows downwards
    ax.set_aspect("equal"), ax.legend(), ax.set_xlabel(x), ax.set_ylabel(y), ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, fname), dpi=120)
    plt.close(fig)


def size_hist(df):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, df[["ref_size", "tgt_size"]].values.max() * 1.05, 50)
    ax.hist(df.ref_size, bins, alpha=0.4, color="gray", label="reference (all)")
    for rel in C.RELATIONS:
        ax.hist(df[df.relation == rel].tgt_size, bins, histtype="step", lw=2, color=COLORS[rel], label=f"target {rel}")
    ax.set_xlabel("object size (px)"), ax.legend(fontsize=8), ax.set_title("Object sizes")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "sizes.png"), dpi=120)
    plt.close(fig)


def run_checks(df):
    W, m = C.CANVAS, C.EDGE_MARGIN * C.CANVAS
    ok = lambda b: "PASS" if b else "FAIL"
    print("\n=== automatic checks ===")

    counts = df.groupby(["relation", "pool"]).size()
    print(f"[{ok(counts.nunique() == 2)}] balanced: {counts.to_dict()}")

    inside = True
    for p in ("ref", "tgt"):
        h = df[f"{p}_size"] / 2
        inside &= ((df[f"{p}_cx"] - h >= m) & (df[f"{p}_cx"] + h <= W - m) &
                   (df[f"{p}_cy"] - h >= m) & (df[f"{p}_cy"] + h <= W - m)).all()
    print(f"[{ok(inside)}] every object at least {m:.1f}px from every border")

    lr = df[df.relation.isin(["left", "right"])]
    gap = lr.dx.abs() - (lr.ref_size + lr.tgt_size) / 2
    print(f"[{ok((lr.overlap_px == 0).all() and (gap > 0).all())}] left/right never overlap "
          f"(min edge gap {gap.min():.1f}px)")
    print(f"[{ok(((df.relation == 'left') == (df.dx < 0))[lr.index].all())}] left => target centre left of reference")

    fb = df[df.relation.isin(["front", "back"])]
    print(f"[{ok((fb.overlap_px > 0).all())}] front/back always overlap")
    print(f"[{ok((df[df.relation == 'front'].dy > 0).all() and (df[df.relation == 'back'].dy < 0).all())}] "
          f"front lower, back higher")

    tv, te = df[df.pool == "trainval"], df[df.pool == "test"]
    test_cols = set(C.TEST_COLORS["reference"]) | set(C.TEST_COLORS["target"])
    leak = tv.ref_color.isin(C.TEST_COLORS["reference"]).any() or tv.tgt_color.isin(C.TEST_COLORS["target"]).any()
    only_test = te.ref_color.isin(C.TEST_COLORS["reference"]).all() and te.tgt_color.isin(C.TEST_COLORS["target"]).all()
    print(f"[{ok(not leak and only_test)}] held-out colours {sorted(test_cols)} appear only in test")

    # --- shortcut check: can the label be guessed from the REFERENCE position alone?
    # It should not: the reference position is supposed to carry no label information.
    X = df[["ref_cx", "ref_cy"]].values
    acc = cross_val_score(LogisticRegression(max_iter=1000), X, df.relation, cv=5).mean()
    print(f"[{ok(acc < 0.35)}] label from reference position only: {acc:.1%} (chance = 25%)")

    print("\nper-relation means (px):")
    print(df.groupby("relation")[["ref_cx", "ref_cy", "ref_size", "tgt_size", "dx", "dy"]].mean().round(1))


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(C.DATA_DIR, "metadata.csv"))

    example_grid(df, "trainval")
    example_grid(df, "test")
    scatter_by_relation(df, "dx", "dy", "Target offset from reference (px)", "offsets.png", center_lines=True)
    scatter_by_relation(df, "ref_cx", "ref_cy", "Reference centre position (px)", "reference_positions.png")
    size_hist(df)
    run_checks(df)
    print(f"\nFigures saved to {C.FIG_DIR}/")


if __name__ == "__main__":
    main()
