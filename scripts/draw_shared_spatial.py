#!/usr/bin/env python3
"""Draw the shared-spatial study figures from the committed result tables.

    MPLBACKEND=Agg .venv/bin/python scripts/draw_shared_spatial.py

The architecture drawing is a static diagram of the registered head, in the same
visual language as scripts/draw_looped_head.py. The charts read
results/shared-spatial/five_seed.csv and screen.csv. No model is imported and
no fit is launched.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
TABLES = ROOT / "results" / "shared-spatial"

SCALE = {"P": "#3B76C4", "M": "#E8862A", "G": "#3E9B4F"}
SHARED = "#7B4FC9"
INK, GREY, LINE, RULE = "#1F2430", "#6B7280", "#9CA3AF", "#E5E7EB"
TEAL = "#0F766E"
H80 = 7.2


def H(ch: float) -> float:
    return H80 * ch / 80


def box(ax, x, y, w, h, color, text="", fs=9, alpha=0.22, lw=1.6, bold=False, tc=INK):
    style = "round,pad=0.12,rounding_size=0.45"
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=style, fc=color, ec="none", alpha=alpha, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=style, fc="none", ec=color, lw=lw, zorder=2.2))
    if text:
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3,
                fontweight="bold" if bold else "normal")


def arrow(ax, a, b, color=LINE, lw=1.3, ms=11):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=1.5))


def seg(ax, xs, ys, color=LINE, lw=1.3):
    ax.plot(xs, ys, color=color, lw=lw, zorder=1.4, solid_capstyle="round")


def scale_input(ax, x, s, y):
    box(ax, x, y, 16, 10, SCALE[s], "", alpha=0.12)
    ax.text(x + 8, y + 1.7, s, ha="center", va="center", fontsize=14, fontweight="bold", color=SCALE[s], zorder=3)
    desc = {"P": "64 ch \u00b7 64\u00d764", "M": "128 ch \u00b7 32\u00d732", "G": "256 ch \u00b7 16\u00d716"}[s]
    ax.text(x + 8, y - 2.2, desc, ha="center", va="center", fontsize=8, color=INK, zorder=3)


def draw_architecture() -> None:
    fig, ax = plt.subplots(figsize=(17.2, 22.4), dpi=150)
    ax.set_xlim(0, 172)
    ax.set_ylim(0, 224)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(86, 218.6, "YOLO26n detection head: native weights and shared spatial weights",
            ha="center", va="center", fontsize=18, fontweight="bold", color=INK)
    ax.text(86, 214.2,
            "blue / orange / green = weights of one scale     purple = one kernel used by P, M and G",
            ha="center", va="center", fontsize=11, color=GREY)
    ax.text(86, 210.6,
            "one internal copy is drawn (one-to-many)  \u00b7  the one-to-one copy has the same pattern and its own weights",
            ha="center", va="center", fontsize=10.5, color=GREY)

    def section(y, title, subtitle):
        ax.text(86, y, title, ha="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(86, y - 3.5, subtitle, ha="center", fontsize=10.5, color=GREY)

    def pipeline(y, s, shared):
        c = SCALE[s]
        yb, yc = y + 5.2, y - 5.0
        box_h, cls_h = 3.5, H(80)
        scale_input(ax, 4, s, y)
        arrow(ax, (20.2, y + 1.2), (25.7, yb))
        arrow(ax, (20.2, y - 1.2), (25.7, yc))
        # Block 1 is never shared: it maps 64/128/256 channels to the tower width.
        box(ax, 26, yb, 13, box_h, c, "3\u00d73", fs=8.5, alpha=0.55)
        box(ax, 26, yc, 13, cls_h, c, "DW+PW", fs=8, alpha=0.28)
        seg(ax, [39.2, 43.6], [yb, yb])
        seg(ax, [39.2, 43.6], [yc, yc])
        b2 = SHARED if shared else c
        b2_ink = "white" if shared else INK
        box(ax, 44, yb, 13, box_h, b2, "3\u00d73", fs=8.5, alpha=0.85 if shared else 0.55, tc=b2_ink, bold=shared)
        box(ax, 44, yc, 13, cls_h, b2, "DW+PW", fs=8, alpha=0.55 if shared else 0.28, tc=b2_ink if shared else INK, bold=shared)
        for yy in (yb, yc):
            ax.add_patch(FancyBboxPatch((58.3, yy - 1.35), 7.2, 2.7,
                                        boxstyle="round,pad=0.02,rounding_size=0.35",
                                        fc=c, ec=c, lw=0, zorder=3))
            ax.text(61.9, yy, "BN", ha="center", va="center", fontsize=7.4, color="white",
                    fontweight="bold", zorder=4)
        seg(ax, [57.2, 58.3], [yb, yb])
        seg(ax, [57.2, 58.3], [yc, yc])
        seg(ax, [65.5, 69.2], [yb, yb])
        seg(ax, [65.5, 69.2], [yc, yc])
        pred = SHARED if shared else c
        pred_ink = "white" if shared else INK
        box(ax, 69.4, yb, 8.2, box_h, pred, "1\u00d71", fs=8.5, alpha=0.9 if shared else 0.7, tc=pred_ink, bold=True)
        box(ax, 69.4, yc, 8.2, cls_h, pred, "1\u00d71", fs=8.5, alpha=0.7 if shared else 0.4, tc=pred_ink, bold=shared)
        if shared:
            for yy in (yb, yc):
                ax.add_patch(Circle((80.4, yy), 1.45, fc=c, ec="white", lw=0.7, zorder=4))
                ax.text(80.4, yy, "b", ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold", zorder=5)
            seg(ax, [77.8, 78.9], [yb, yb])
            seg(ax, [77.8, 78.9], [yc, yc])
        seg(ax, [81.9, 86.6], [yb, y])
        seg(ax, [81.9, 86.6], [yc, y])
        arrow(ax, (86.6, y), (90.4, y))
        ax.text(91.2, y, f"{s} predictions", va="center", fontsize=11, color=c, fontweight="bold")

    section(205.5, "NATIVE \u2014 three separate heads", "every convolution, BatchNorm and predictor belongs to one scale")
    ax.text(32.5, 198.6, "block 1", ha="center", fontsize=9, color=GREY)
    ax.text(50.5, 198.6, "block 2", ha="center", fontsize=9, color=GREY)
    ax.text(61.9, 198.6, "BN", ha="center", fontsize=9, color=GREY)
    ax.text(73.5, 198.6, "predictor", ha="center", fontsize=9, color=GREY)
    for s, y in (("P", 186), ("M", 164), ("G", 142)):
        pipeline(y, s, shared=False)
    ax.text(118, 186, "Thin track \u00b7 box", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(118, 182.2, "Conv 3\u00d73 to 16 channels, twice,\nthen 1\u00d71 to 4 box outputs.",
            fontsize=9.5, color=INK, va="top", linespacing=1.35)
    ax.text(118, 170, "Tall track \u00b7 class", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(118, 166.2, "Depthwise 3\u00d73 plus pointwise 1\u00d71\nto 80 channels, twice, then\n1\u00d71 to 80 class outputs.",
            fontsize=9.5, color=INK, va="top", linespacing=1.35)
    ax.text(118, 148.5, "The colored BN tag is that scale's\nBatchNorm. In the native head the\nblock-2 kernel is also per scale.",
            fontsize=9, color=GREY, va="top", linespacing=1.35)
    seg(ax, [5, 167], [128.5, 128.5], RULE, 1.2)

    section(123.2, "SPATIAL_SEPBN_SHAREDPRED \u2014 later kernels shared, BatchNorm per scale",
            "block 1 stays native \u00b7 block 2 convolution and the predictor kernel are one object for P, M and G")
    ax.text(32.5, 116.2, "block 1 \u00b7 own", ha="center", fontsize=9, color=GREY)
    ax.text(50.5, 116.2, "block 2 \u00b7 shared", ha="center", fontsize=9, color=SHARED)
    ax.text(61.9, 116.2, "BN \u00b7 own", ha="center", fontsize=9, color=GREY)
    ax.text(73.5, 116.2, "kernel shared", ha="center", fontsize=9, color=SHARED)
    for s, y in (("P", 103), ("M", 81), ("G", 59)):
        pipeline(y, s, shared=True)
    ax.text(118, 104, "What is shared", fontsize=10.5, color=SHARED, fontweight="bold")
    ax.text(118, 100.2, "Inside one family, P, M and G use\nthe same block-2 kernel and the\nsame predictor kernel.",
            fontsize=9.5, color=INK, va="top", linespacing=1.35)
    ax.text(118, 86, "What stays per scale", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(118, 82.2, "Block 1, because the three maps\ndo not have the same channel count.\nBatchNorm, including its running\nstatistics. The predictor bias,\nthe colored b.",
            fontsize=9.5, color=INK, va="top", linespacing=1.35)
    ax.text(118, 62.5, "Box and class are not tied to each\nother. The one-to-one head repeats\nthis pattern with its own kernels.",
            fontsize=9, color=GREY, va="top", linespacing=1.35)
    seg(ax, [5, 167], [45.6, 45.6], RULE, 1.2)

    ax.text(86, 41.2, "One location in the class tower", ha="center", fontsize=13, fontweight="bold", color=INK)

    def path(y, label, steps, note):
        ax.text(5, y + 4.6, label, fontsize=10, fontweight="bold", color=INK)
        box(ax, 5, y, 8, 4.2, SCALE["P"], "in", fs=8, alpha=0.15)
        x = 16.5
        for text, color, w in steps:
            arrow(ax, (x - 3.2, y), (x - 0.2, y))
            tc = SHARED if color == SHARED else INK
            box(ax, x, y, w, 4.4, color, text, fs=8, alpha=0.32, tc=tc, bold=color == SHARED)
            x += w + 3.4
        ax.text(x + 1, y, note, va="center", fontsize=9.2, color=GREY)

    path(30.5, "native",
         [("block 1 \u00b7 own", SCALE["P"], 18), ("block 2 \u00b7 own", SCALE["P"], 18), ("1\u00d71 \u00b7 own", SCALE["P"], 14)],
         "three scale-specific stages")
    path(18.5, "shared spatial",
         [("block 1 \u00b7 own", SCALE["P"], 18), ("block 2 \u00b7 shared", SHARED, 20), ("1\u00d71 \u00b7 shared", SHARED, 16)],
         "same later kernels at P, M and G; own BN and predictor bias")

    ax.text(86, 10.6,
            "The shared kernel is still executed once per scale, so the ConvMAC count does not fall. "
            "Unique head parameters: 309,656 native, 246,104 shared.",
            ha="center", fontsize=9.5, color=GREY)
    ax.text(86, 7.2,
            "Head ConvMACs at 512 px, batch 1: 384,606,208 for both. Full detector: 1,906,679,808 for both.",
            ha="center", fontsize=9.5, color=GREY)

    fig.savefig(ASSETS / "shared-spatial-head.png", dpi=150, facecolor="white")
    plt.close(fig)


def _read(name):
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis="y", color="#F3F4F6", zorder=0)
    ax.set_axisbelow(True)


def draw_charts() -> None:
    seeds = _read("five_seed.csv")
    screen = _read("screen.csv")
    xs = list(range(len(seeds)))
    labels = [row["seed"] for row in seeds]
    native = [float(row["native_ap50_95_x100"]) for row in seeds]
    shared = [float(row["sharedpred_ap50_95_x100"]) for row in seeds]
    deltas = [float(row["paired_delta_pp"]) for row in seeds]
    mean = sum(deltas) / len(deltas)
    # Student t critical value, df = 4, two-sided 95%.
    tcrit = 2.776445105
    sd = (sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)) ** 0.5
    half = tcrit * sd / (len(deltas) ** 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), dpi=160)
    fig.patch.set_facecolor("white")
    ax = axes[0]
    _style(ax)
    for x, a, b in zip(xs, native, shared):
        ax.plot([x, x], [a, b], color="#E5E7EB", lw=4, zorder=1)
    ax.scatter(xs, native, s=58, color="#3B76C4", zorder=3, label="native")
    ax.scatter(xs, shared, s=58, color=SHARED, zorder=3, label="spatial_sepbn_sharedpred")
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(14.75, 15.9)
    ax.set_xlabel("Seed", color=INK)
    ax.set_ylabel("Development AP50:95 \u00d7100", color=INK)
    ax.set_title("Paired development accuracy", loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    _style(ax)
    ax.axhline(0, color="#9CA3AF", lw=1, zorder=1)
    ax.scatter(xs, deltas, s=64, color=TEAL, zorder=3)
    ax.scatter([5.15], [mean], s=70, marker="D", color=INK, zorder=3)
    ax.plot([5.15, 5.15], [mean - half, mean + half], color=INK, lw=1.6, zorder=2)
    ax.set_xticks(xs + [5.15], labels + ["mean"])
    ax.set_xlim(-0.45, 5.65)
    ax.set_ylabel("Paired difference, points", color=INK)
    ax.set_xlabel("Seed", color=INK)
    ax.set_title("sharedpred \u2212 native", loc="left", color=INK, fontsize=13, fontweight="bold", pad=18)
    ax.text(0.0, 1.02,
            f"mean {mean:+.3f}     95% t interval [{mean - half:+.3f}, {mean + half:+.3f}]     df = 4",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=GREY, clip_on=False)
    fig.tight_layout()
    fig.savefig(TABLES / "paired.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    order = ["native", "spatial_untied", "spatial_sepbn", "spatial_gn", "spatial_sepbn_k5", "spatial_sepbn_sharedpred"]
    by_name = {row["variant"]: row for row in screen}
    rows = [by_name[name] for name in order]
    short = {
        "native": "native",
        "spatial_untied": "untied control",
        "spatial_sepbn": "shared conv\nseparate BN",
        "spatial_gn": "shared conv\nGroupNorm",
        "spatial_sepbn_k5": "shared conv\n5\u00d75, separate BN",
        "spatial_sepbn_sharedpred": "shared conv + predictor\nseparate BN",
    }
    fig, ax = plt.subplots(figsize=(11.6, 5.4), dpi=160)
    fig.patch.set_facecolor("white")
    _style(ax)
    ax.axhline(0, color="#9CA3AF", lw=1)
    for i, row in enumerate(rows):
        d11 = float(row["seed11_ap50_95_x100"]) - float(by_name["native"]["seed11_ap50_95_x100"])
        d33 = float(row["seed33_ap50_95_x100"]) - float(by_name["native"]["seed33_ap50_95_x100"])
        if row["variant"] == "native":
            d11 = d33 = 0.0
        mean_d = float(row["mean_paired_delta_pp"])
        color = SHARED if row["variant"] == "spatial_sepbn_sharedpred" else "#94A3B8"
        ax.plot([i - 0.12, i + 0.12], [d11, d33], color="#E5E7EB", lw=5, zorder=1)
        ax.scatter([i - 0.12, i + 0.12], [d11, d33], s=42, color="#334155", zorder=3)
        ax.scatter([i], [mean_d], s=90, marker="D", color=color, zorder=4)
    ax.set_xticks(range(len(rows)), [short[row["variant"]] for row in rows])
    ax.set_ylabel("Development AP difference vs native, points", color=INK)
    ax.set_title("Screen on seeds 11 and 33", loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.text(0.0, -0.22, "Dots are the two seeds. Diamonds are the paired means. Only the purple head was trained on the other three seeds.",
            transform=ax.transAxes, color=GREY, fontsize=9)
    fig.tight_layout()
    fig.savefig(TABLES / "screen.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=160)
    fig.patch.set_facecolor("white")
    specs = [
        (axes[0], "Unique head parameters", [309656, 246104], "parameters", "\u221220.5%"),
        (axes[1], "Head ConvMACs, 512 px, batch 1", [384606208, 384606208], "ConvMACs", "0%"),
    ]
    for ax, title, values, ylabel, tag in specs:
        _style(ax)
        bars = ax.bar([0, 1], values, width=0.62, color=["#3B76C4", SHARED], zorder=3)
        ax.set_xticks([0, 1], ["native", "sharedpred"])
        ax.set_ylabel(ylabel, color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(0, max(values) * 1.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03, f"{value:,}",
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.text(0.98, 0.90, tag, transform=ax.transAxes, ha="right", va="top", fontsize=14,
                fontweight="bold", color=TEAL if tag != "0%" else GREY)
    fig.tight_layout()
    fig.savefig(TABLES / "compute.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw_architecture()
    draw_charts()
    print("wrote", ASSETS / "shared-spatial-head.png")
    print("wrote", TABLES / "paired.svg")
    print("wrote", TABLES / "screen.svg")
    print("wrote", TABLES / "compute.svg")
