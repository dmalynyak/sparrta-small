"""
All settings for the project live here, so every script uses the same numbers.

Sizes and distances are FRACTIONS of the canvas width (0.14 = 14% of 224 px),
so the dataset looks the same if you change the canvas resolution.
"""

# ---------------------------------------------------------------- paths
DATA_DIR = "data"                 # images + metadata.csv are written here
IMAGE_DIR = "data/images"
FIG_DIR = "figures"               # sanity-check plots

# ---------------------------------------------------------------- reproducibility
DATASET_SEED = 2026               # seed for dataset generation ONLY
PROBE_SEEDS = [0, 1, 2]           # used later for the 3 train/val splits (not here)

# ---------------------------------------------------------------- canvas
CANVAS = 224                      # final image size (= model input size, no resizing artifacts)
SUPERSAMPLE = 4                   # draw at 4x resolution, then shrink -> smooth edges

# ---------------------------------------------------------------- classes
RELATIONS = ["left", "right", "front", "back"]
N_PER_RELATION = 250              # 250 x 4 = 1000 images
TEST_FRACTION = 0.20              # 20% of each relation uses held-out colors (test pool)
                                  # the other 80% is the train/val pool (split 70/10 later)

# ---------------------------------------------------------------- shapes (shape tells the role)
REF_SHAPES = ["circle", "square"]
TGT_SHAPES = ["triangle", "star", "cross"]

# ---------------------------------------------------------------- colors
PALETTE = {
    "red":    (215, 45, 45),
    "green":  (40, 155, 65),
    "blue":   (45, 85, 215),
    "yellow": (225, 185, 20),
    "purple": (140, 55, 180),
    "teal":   (20, 160, 170),
}
# 2 of the 6 colors per role appear ONLY in the test pool.
# We reserve the same two for both roles, so test colors are never seen in training at all.
TEST_COLORS = {"reference": ["purple", "teal"], "target": ["purple", "teal"]}
COLOR_JITTER = 12                 # +-12 per RGB channel, small natural variation

# ---------------------------------------------------------------- sizes
BASE_SIZE = 0.14                  # object side length (bounding square) as fraction of canvas
FRONT_SCALE = 1.5                 # front target is 1.5x larger
BACK_SCALE = 0.65                 # back target is 0.65x smaller
SIZE_JITTER = 0.10                # every object: size * U(0.9, 1.1)

# ---------------------------------------------------------------- positions
REF_POS_JITTER = 0.15             # reference center = canvas center + U(-15%, +15%) in x and y
EDGE_MARGIN = 0.06                # objects must stay this far from every border

# left / right
LR_GAP = (0.02, 0.05)             # empty space between the two objects' edges (always > 0)
LR_VERTICAL_JITTER = 0.04         # small vertical offset so left/right are not perfectly aligned

# front / back  (offsets are multiples of (ref half-size + target half-size))
FB_DX_JITTER = 0.30               # small horizontal offset
FRONT_DY = (0.45, 0.85)           # front target sits LOWER  (positive dy = down in image coords)
BACK_DY = (0.50, 0.90)            # back target sits HIGHER
OCCLUDED_FRACTION = (0.15, 0.65)  # hidden object must be 15%-65% covered: real occlusion,
                                  # but still recognisable

# ---------------------------------------------------------------- background
BG_BASE_RANGE = (195, 240)        # pale random base color per channel
BG_GRADIENT = 25                  # gradient endpoint differs by up to +-25 per channel
BG_NOISE_STD = 4.0                # mild pixel noise

MAX_TRIES = 200                   # resampling attempts per image before giving up
