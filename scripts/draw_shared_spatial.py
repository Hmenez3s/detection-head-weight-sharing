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
H80 = 9.0                                   # drawn height of 80 channels, as in draw_looped_head.py


def H(ch: float) -> float:
    return H80 * ch / 80                    # block height is proportional to channels


def box(ax, x, y, w, h, color, text="", fs=9, alpha=0.22, lw=1.6, bold=False, tc=INK, ls="-", opaque=False):
    style = "round,pad=0.15,rounding_size=0.5"
    if opaque:  # hides a lane line that runs behind a translucent block
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=style, fc="white", ec="none", zorder=1.9))
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=style, fc=color, ec="none", alpha=alpha, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=style, fc="none", ec=color, lw=lw, ls=ls, zorder=2.2))
    if text:
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3,
                fontweight="bold" if bold else "normal")


def arrow(ax, a, b, color=LINE, lw=1.3, ms=10):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=1.5))


def seg(ax, xs, ys, color=LINE, lw=1.3):
    ax.plot(xs, ys, color=color, lw=lw, zorder=1.4, solid_capstyle="round")


DESC = {"P": "64 ch \u00b7 64\u00d764", "M": "128 ch \u00b7 32\u00d732", "G": "256 ch \u00b7 16\u00d716"}


def scale_input(ax, x, s, y):
    box(ax, x, y, 14, 9, SCALE[s], "", alpha=0.1)
    ax.text(x + 7, y + 1.5, s, ha="center", va="center", fontsize=13, fontweight="bold", color=SCALE[s], zorder=3)
    ax.text(x + 7, y - 2, DESC[s], ha="center", va="center", fontsize=8, color=INK, zorder=3)


def bn_tag(ax, cx, y, color):
    ax.add_patch(FancyBboxPatch((cx - 1.9, y - 1.1), 3.8, 2.2, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc=color, ec=color, lw=0, zorder=3))
    ax.text(cx, y, "BN", ha="center", va="center", fontsize=6.6, color="white", fontweight="bold", zorder=4)


def bias_dot(ax, cx, y, color):
    ax.add_patch(Circle((cx, y), 1.15, fc=color, ec="white", lw=0.7, zorder=4))
    ax.text(cx, y, "+", ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=5)


def chip_width(label):
    return 0.78 * max(len(line) for line in label.split("\n")) + 2.4


def weight_chip(ax, cx, y, label):
    """A stored weight tensor: the only purple rectangle, drawn once per tensor."""
    w = chip_width(label)
    h = 3.0 if "\n" not in label else 5.0
    ax.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc=SHARED, ec=SHARED, lw=0, zorder=3))
    ax.text(cx, y, label, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=4,
            linespacing=1.15)
    return h


def application(ax, cx, y):
    """One execution of a stored weight on one scale's map: a ring, never a rectangle."""
    ax.add_patch(Circle((cx, y), 1.4, fc="white", ec=SHARED, lw=1.5, zorder=3))


# One entry per shared tensor in the drawn internal copy: (x of its three applications, label, side).
# Box tensors sit above the lanes and class tensors below, so each weight link runs straight
# to its applications without sharing a column with any other tensor or any BN tag.
SHARED_WEIGHTS = {
    "box3x3": (38, "box 3\u00d73", "top"),
    "classdw": (45, "class DW\n3\u00d73", "bottom"),
    "classpw": (58, "class PW\n1\u00d71", "bottom"),
    "boxpred": (77, "box 1\u00d71", "top"),
    "classpred": (83, "class 1\u00d71", "bottom"),
}


def draw_architecture() -> None:
    """Native head against spatial_sepbn_sharedpred, one internal copy, no recurrence.

    Each scale keeps its own map end to end (solid grey lanes). A purple rectangle is a weight
    tensor stored once; dashed purple links run from it to the three rings where P, M and G
    each apply it. BN tags and bias dots are solid scale colours and never shared.
    """
    fig, ax = plt.subplots(figsize=(15.4, 19.6), dpi=150)
    ax.set_xlim(0, 154)
    ax.set_ylim(0, 196)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    CX = 77

    ax.text(CX, 193, "YOLO26n detection head: native vs. shared spatial weights", ha="center", va="center",
            fontsize=19, fontweight="bold", color=INK)
    ax.text(CX, 189, "block height = number of channels (coloured rectangles)   \u00b7   blue / orange / green = weights of one scale"
            "   \u00b7   purple = weights stored once", ha="center", va="center", fontsize=10.5, color=GREY)
    ax.text(CX, 185.7, "one internal copy is drawn (one-to-many)  \u00b7  the one-to-one copy has the same pattern with its own weights",
            ha="center", va="center", fontsize=10, color=GREY)

    def section(y, title, subtitle):
        ax.text(CX, y, title, ha="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(CX, y - 3.6, subtitle, ha="center", fontsize=10.5, color=GREY)

    def note(x, y, head, body):
        ax.text(x, y, head, fontsize=10, color=INK, fontweight="bold")
        ax.text(x, y - 1.7, body, fontsize=9.3, color=INK, va="top", linespacing=1.4)

    uses = {name: [] for name in SHARED_WEIGHTS}

    def scale_lanes(g, s, shared):
        """One scale: its input card once, then the box lane (thin) and class lane (tall)."""
        c = SCALE[s]
        yb, yc = g + 5.2, g - 1.6
        scale_input(ax, 3, s, g)
        arrow(ax, (17.3, g + 1), (22.8, yb))
        arrow(ax, (17.3, g - 1), (22.8, yc))
        box(ax, 23, yb, 7, H(16), c, alpha=0.55)
        box(ax, 23, yc, 7, H(80), c, alpha=0.28)
        seg(ax, [30.4, 94.2], [yb, yb])
        seg(ax, [30.4, 94.2], [yc, yc])

        def op(name, y, native_x, native_h):
            if shared:
                application(ax, SHARED_WEIGHTS[name][0], y)
                uses[name].append(y)
            else:
                box(ax, native_x, y, 7, native_h, c, alpha=0.55 if y == yb else 0.28, opaque=True)

        op("box3x3", yb, 39, H(16))
        bn_tag(ax, 51 if shared else 50, yb, c)
        op("boxpred", yb, 79.5, H(4))
        op("classdw", yc, 39, H(80))
        bn_tag(ax, 51 if shared else 50, yc, c)
        op("classpw", yc, 53.5, H(80))
        bn_tag(ax, 65 if shared else 64.1, yc, c)
        op("classpred", yc, 79.5, H(80))

        for y in (yb, yc):
            bias_dot(ax, 95.4, y, c)
            seg(ax, [96.7, 102], [y, g])
        arrow(ax, (102, g), (105.4, g))
        ax.text(106.2, g, f"{s} predictions", va="center", fontsize=10, color=c, fontweight="bold")

    # ------------------------------------------------------------ NATIVE
    T = 179
    section(T, "NATIVE \u2014 three separate heads", "every conv, BatchNorm and predictor belongs to one scale  \u00b7  "
            "each head runs on its own map")
    for x, t in ((26.5, "block 1"), (56, "block 2  (conv + BN)"), (89, "predictor  1\u00d71 + bias")):
        ax.text(x, T - 10, t, ha="center", fontsize=9, color=GREY)
    for s, dy in (("P", -19), ("M", -35), ("G", -51)):
        scale_lanes(T + dy, s, shared=False)
    note(123, T - 14.5, "thin row \u00b7 box", "conv 3\u00d73 \u2192 16 ch, twice,\nthen 1\u00d71 \u2192 4 box outputs")
    note(123, T - 24, "tall row \u00b7 class", "separable 3\u00d73 \u2192 80 ch, twice,\nthen 1\u00d71 \u2192 80 class outputs")
    ax.text(123, T - 33, "each separable block = DW 3\u00d73 + PW 1\u00d71,\neach followed by BN (two BN in the\n"
            "tall row's block 2)", fontsize=9, color=GREY, va="top", linespacing=1.4)
    ax.text(123, T - 42, "Nothing is shared: block 2 and the\npredictor weight are also per scale.",
            fontsize=9, color=GREY, va="top", linespacing=1.4)
    seg(ax, [5, 151], [T - 61, T - 61], RULE, 1.2)

    # ------------------------------------------------------------ SHARED
    T2 = 112.5
    section(T2, "SPATIAL_SEPBN_SHAREDPRED \u2014 block-2 kernels and predictor weight shared by P, M and G",
            "not looped  \u00b7  the three maps stay separate  \u00b7  BatchNorm and predictor bias stay per scale")
    ax.text(26.5, T2 - 10.5, "block 1 \u00b7 own", ha="center", fontsize=9, color=GREY)
    ax.text(97.5, T2 - 10.5, "own bias", ha="center", fontsize=9, color=GREY)
    for s, dy in (("P", -22.5), ("M", -38.5), ("G", -54.5)):
        scale_lanes(T2 + dy, s, shared=True)

    # One weight object per tensor, one dashed link to each of its three applications.
    top_y, bottom_y = T2 - 10.5, T2 - 67
    for name, (x, label, side) in SHARED_WEIGHTS.items():
        y0 = top_y if side == "top" else bottom_y
        h = weight_chip(ax, x, y0, label)
        far = max(uses[name]) if side == "bottom" else min(uses[name])
        start = y0 - h / 2 if side == "top" else y0 + h / 2
        ax.plot([x, x], [start, far], color=SHARED, lw=1.5, ls="--", zorder=1.7, dash_capstyle="butt")
    ax.text(57, top_y, "box weights, stored once", ha="center", va="center", fontsize=8.5, color=SHARED)
    ax.text(38, bottom_y, "class weights,\nstored once", ha="right", va="center", fontsize=8.5, color=SHARED,
            linespacing=1.2)

    note(123, T2 - 14.5, "one kernel, three maps",
         "each purple weight is applied to the\nP, M and G maps separately, with\nthe same weight values")
    note(123, T2 - 26, "still per scale", "block 1, every BatchNorm and the\npredictor bias")
    note(123, T2 - 35.5, "not tied to each other", "box and class families, and the\none-to-many and one-to-one heads")
    note(123, T2 - 44.5, "no recurrence", "no loop, feedback, concat or sum of\nthe P, M and G maps")

    ly = T2 - 73.5
    weight_chip(ax, 8, ly, "weight")
    ax.text(12.5, ly, "stored weight, drawn once", va="center", fontsize=8.5, color=GREY)
    ax.plot([37, 43], [ly, ly], color=SHARED, lw=1.5, ls="--")
    ax.text(44.2, ly, "dashed link = uses this weight", va="center", fontsize=8.5, color=GREY)
    application(ax, 70, ly)
    ax.text(72.2, ly, "one application on one scale", va="center", fontsize=8.5, color=GREY)
    bn_tag(ax, 99, ly, SCALE["M"])
    ax.text(101.6, ly, "per-scale BN", va="center", fontsize=8.5, color=GREY)
    bias_dot(ax, 114, ly, SCALE["M"])
    ax.text(115.7, ly, "bias", va="center", fontsize=8.5, color=GREY)
    arrow(ax, (123, ly), (129, ly))
    ax.text(130.2, ly, "grey = feature path", va="center", fontsize=8.5, color=GREY)

    ax.text(CX, T2 - 77.2,
            "Each purple weight is stored once and used by P, M and G on their own separate maps: nothing is concatenated, "
            "summed or looped.\n"
            "Class block 2 has two weights (DW 3\u00d73, PW 1\u00d71), each followed by its own per-scale BN. "
            "The predictor shares its weight, not its bias.",
            ha="center", va="top", fontsize=9, color=GREY, linespacing=1.5)
    seg(ax, [5, 151], [27.8, 27.8], RULE, 1.2)

    # ------------------------------------------------------------ NUMBERS
    rows = [("", "native", "spatial_sepbn_sharedpred"),
            ("stored once for P, M and G", "nothing", "block-2 kernels + predictor weight"),
            ("unique head parameters, both internal branches", "309,656", "246,104"),
            ("head ConvMACs, 512 px, batch 1", "384,606,208", "384,606,208"),
            ("full detector ConvMACs", "1,906,679,808", "1,906,679,808")]
    ax.add_patch(FancyBboxPatch((8, 8.4), 138, 17.4, boxstyle="round,pad=0.2,rounding_size=1",
                                fc="#F9FAFB", ec=RULE, lw=1, zorder=0.5))
    for i, row in enumerate(rows):
        yy = 23.6 - i * 3.3
        ax.text(60, yy, row[0], ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
        for xc, txt, col in zip((86, 121), row[1:], (SCALE["P"], SHARED)):
            ax.text(xc, yy, txt, ha="center", va="center", fontsize=10, color=col if i == 0 else INK,
                    fontweight="bold" if i == 0 else "normal")
    ax.text(CX, 5.4, "The shared kernel is still executed once per scale, so the ConvMAC count does not fall. "
            "Counts are convolution multiply-adds only.", ha="center", va="center", fontsize=8.6, color=GREY)

    fig.savefig(ASSETS / "shared-spatial-head.png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.25)
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
