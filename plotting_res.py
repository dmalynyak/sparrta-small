from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results/results.csv")
OUT_DIR = Path("results/plots")

TASKS = ["4way", "left_right", "front_back"]
TASK_NAMES = {"4way": "4-way", "left_right": "Left/right", "front_back": "Front/back"}
LAYERS = range(1, 13)
RESNET_X = 3  # resnet stages 1-4 are drawn at layers 3, 6, 9, 12

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e5e4e0"
SUP, DINO, RESNET = "#2a78d6", "#eb6834", "#1baf7a"

# color = model, line style and marker = pooling
LINES = {
    ("vit_sup", "cls"): dict(label="Supervised ViT, CLS", color=SUP, ls="-", marker="o"),
    ("vit_sup", "mean"): dict(label="Supervised ViT, mean", color=SUP, ls=":", marker="^"),
    ("vit_dino", "cls"): dict(label="DINO ViT, CLS", color=DINO, ls="-", marker="o"),
    ("vit_dino", "mean"): dict(label="DINO ViT, mean", color=DINO, ls=":", marker="^"),
    ("resnet50", "gap"): dict(label="ResNet-50, GAP", color=RESNET, ls="--", marker="s"),
}
SPLITS = {
    "train_acc": dict(label="Train", color="#8a8984", ls="-", marker="o"),
    "val_acc": dict(label="Val (same colors)", color="#6250d6", ls="-", marker="s"),
    "test_acc": dict(label="Test (unseen colors)", color="#e34948", ls="-", marker="^"),
}


def load_means():
    df = pd.read_csv(RESULTS)
    return df.groupby(["model", "pooling", "task", "layer"])[list(SPLITS)].mean()


def curve(means, model, pooling, task, metric="test_acc"):
    # returns x (layer positions) and y (mean over seeds)
    s = means.loc[(model, pooling, task), metric]
    x = s.index * RESNET_X if model == "resnet50" else s.index
    return x, s.values


def model_style(model):
    # for panels with one pooling only: same look for cls and mean, no pooling in the label
    style = LINES[(model, "cls")]
    return {**style, "label": style["label"].split(",")[0]}


def draw(ax, x, y, style):
    ax.plot(x, y, lw=1.6, ms=5, markeredgecolor="white", markeredgewidth=0.6, clip_on=False, **style)


def style_axes(ax, title, ylabel=None, ylim=None):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel("Layer", fontsize=9, color=INK_SOFT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    ax.set_xticks(list(LAYERS))
    ax.set_xlim(0.5, 12.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=8, colors=INK_SOFT, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def reference_line(ax, y, label=None):
    ax.axhline(y, color=INK_SOFT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.text(12.4, y + 0.008, label, ha="right", va="bottom", fontsize=8, color=INK_SOFT)


def finish(fig, name, title, handles, labels, ncol):
    fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left")
    fig.legend(handles, labels, loc="lower left", ncol=ncol, frameon=False, fontsize=9,
               bbox_to_anchor=(0.01, 0.0), labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_cls_mean_diff(means, task, name, chance=None):
    # panels: CLS / mean / mean minus CLS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, pooling, title in zip(axes, ["cls", "mean"], ["CLS token", "Mean-pooled patches"]):
        for model in ["vit_sup", "vit_dino"]:
            x, y = curve(means, model, pooling, task)
            draw(ax, x, y, model_style(model))
        x, y = curve(means, "resnet50", "gap", task)
        draw(ax, x, y, LINES[("resnet50", "gap")])
        if chance:
            reference_line(ax, chance, "chance")
        style_axes(ax, title, "Test accuracy", (0.4, 1.02))

    for model in ["vit_sup", "vit_dino"]:
        x, cls = curve(means, model, "cls", task)
        _, mean = curve(means, model, "mean", task)
        draw(axes[2], x, mean - cls, model_style(model))
    reference_line(axes[2], 0)
    style_axes(axes[2], "Mean minus CLS (above 0: mean is better)", "Test accuracy difference")

    handles, labels = axes[0].get_legend_handles_labels()
    finish(fig, name, f"{TASK_NAMES[task]} test accuracy by layer", handles, labels, 3)


def fig_front_back(means):
    # panels: front/back accuracy / front/back minus left/right
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for (model, pooling), style in LINES.items():
        x, fb = curve(means, model, pooling, "front_back")
        _, lr = curve(means, model, pooling, "left_right")
        draw(axes[0], x, fb, style)
        draw(axes[1], x, fb - lr, style)
    style_axes(axes[0], "Front/back ", "Test accuracy", (0.9, 1.005))
    reference_line(axes[1], 0)
    style_axes(axes[1], "Front/back minus left/right (above 0: front/back is easier)", "Test accuracy difference")

    handles, labels = axes[0].get_legend_handles_labels()
    finish(fig, "fig3_front_back", "Front/back test accuracy and its gap to left/right", handles, labels, 5)


def fig_splits(means):
    # one panel per model and pooling, 5 panels on a 2x3 grid
    order = [("vit_sup", "cls"), ("vit_sup", "mean"), ("resnet50", "gap"), ("vit_dino", "cls"), ("vit_dino", "mean")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for i, (ax, (model, pooling)) in enumerate(zip(axes.flat, order)):
        for metric, style in SPLITS.items():
            x, y = curve(means, model, pooling, "left_right", metric)
            draw(ax, x, y, style)
        reference_line(ax, 0.5, "chance")
        style_axes(ax, LINES[(model, pooling)]["label"], "Accuracy" if i % 3 == 0 else None, (0.4, 1.02))
    axes[1, 2].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    finish(fig, "fig4_train_val_test", "Left/right train, val and test accuracy by layer", handles, labels, 3)


def write_tables():
    df = pd.read_csv(RESULTS)
    # std is over the 3 train/val splits with one fixed test set: split sensitivity
    stats = df.groupby(["model", "pooling", "task", "layer"]).test_acc.agg(["mean", "std"]).round(6)

    # table B: every layer, one mean and std column per task
    full = stats.unstack("task")
    full.columns = [f"{task}_{stat}" for stat, task in full.columns]
    full = full[[f"{task}_{stat}" for task in TASKS for stat in ("mean", "std")]]
    full.round(4).to_csv(OUT_DIR / "table_full.csv")

    # table A: peak of each line. ties go to the earliest layer.
    rows = []
    for (model, pooling) in LINES:
        for task in TASKS:
            s = stats.loc[(model, pooling, task)]
            peak = s["mean"].idxmax()
            rows.append(dict(model=LINES[(model, pooling)]["label"], task=TASK_NAMES[task], peak_layer=peak,
                             test_acc_mean=s.loc[peak, "mean"], test_acc_std=s.loc[peak, "std"]))
    peaks = pd.DataFrame(rows)
    peaks.round(4).to_csv(OUT_DIR / "table_peaks.csv", index=False)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    means = load_means()
    fig_cls_mean_diff(means, "4way", "fig1_4way")
    fig_cls_mean_diff(means, "left_right", "fig2_left_right", chance=0.5)
    fig_front_back(means)
    fig_splits(means)
    write_tables()


if __name__ == "__main__":
    main()
