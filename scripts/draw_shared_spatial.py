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
DASH = (0, (3, 2))


def H(ch: float) -> float:
    return H80 * ch / 80                    # block height is proportional to channels


def box(ax, x, y, w, h, color, text="", fs=9, alpha=0.22, lw=1.6, bold=False, tc=INK, ls="-"):
    style = "round,pad=0.15,rounding_size=0.5"
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
    return 0.78 * len(label) + 2.4


def weight_chip(ax, x, y, label):
    """The one authoritative object for a stored weight tensor."""
    w = chip_width(label)
    ax.add_patch(FancyBboxPatch((x, y - 1.5), w, 3.0, boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc=SHARED, ec=SHARED, lw=0, zorder=3))
    ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=4)
    return w


def chip_row(ax, x0, x1, y, labels):
    widths = [chip_width(t) for t in labels]
    gap = 1.2
    x = (x0 + x1) / 2 - (sum(widths) + gap * (len(labels) - 1)) / 2
    for text, w in zip(labels, widths):
        weight_chip(ax, x, y, text)
        x += w + gap


def draw_architecture() -> None:
    """Native head against spatial_sepbn_sharedpred, one internal copy, no recurrence.

    Each scale keeps its own map end to end. A purple dashed block is one application of a
    weight tensor stored once; a solid scale-coloured tag (BN, bias) is never shared.
    """
    fig, ax = plt.subplots(figsize=(15.4, 19.6), dpi=150)
    ax.set_xlim(0, 154)
    ax.set_ylim(0, 196)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    CX = 77

    ax.text(CX, 193, "YOLO26n detection head: native vs. shared spatial weights", ha="center", va="center",
            fontsize=19, fontweight="bold", color=INK)
    ax.text(CX, 189, "block height = number of channels   \u00b7   blue / orange / green = weights of one scale   \u00b7   "
            "purple = weights stored once, applied at P, M and G", ha="center", va="center", fontsize=10.5, color=GREY)
    ax.text(CX, 185.7, "one internal copy is drawn (one-to-many)  \u00b7  the one-to-one copy has the same pattern with its own weights",
            ha="center", va="center", fontsize=10, color=GREY)

    def section(y, title, subtitle):
        ax.text(CX, y, title, ha="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(CX, y - 3.6, subtitle, ha="center", fontsize=10.5, color=GREY)

    def note(x, y, head, body, grey=False):
        ax.text(x, y, head, fontsize=10, color=INK, fontweight="bold")
        ax.text(x, y - 1.7, body, fontsize=9.3, color=GREY if grey else INK, va="top", linespacing=1.4)

    def scale_lanes(g, s, shared):
        """One scale: its input card once, then the box lane (thin) and class lane (tall)."""
        c = SCALE[s]
        yb, yc = g + 5.2, g - 1.6
        scale_input(ax, 3, s, g)
        arrow(ax, (17.3, g + 1), (22.8, yb))
        arrow(ax, (17.3, g - 1), (22.8, yc))
        box(ax, 23, yb, 7, H(16), c, alpha=0.55)
        box(ax, 23, yc, 7, H(80), c, alpha=0.28)

        def block(x, y, h, letter, w=7):
            if shared:  # application of a stored weight: dashed purple, never solid
                box(ax, x, y, w, h, SHARED, letter, fs=7.5, alpha=0.14, lw=1.3, bold=True, tc=SHARED, ls="--")
            else:
                box(ax, x, y, w, h, c, alpha=0.55 if y == yb else 0.28)

        seg(ax, [30.4, 38.8], [yb, yb])
        block(39, yb, H(16), "a")
        seg(ax, [46.2, 48.1], [yb, yb])
        bn_tag(ax, 50, yb, c)
        seg(ax, [51.9, 79.3], [yb, yb])
        block(79.5, yb, H(4), "")  # 4 output channels: too thin to hold its letter
        if shared:
            ax.text(83, yb + 1.7, "d", ha="center", va="center", fontsize=7.5, color=SHARED, fontweight="bold")

        seg(ax, [30.4, 38.8], [yc, yc])
        block(39, yc, H(80), "b")
        seg(ax, [46.2, 48.1], [yc, yc])
        bn_tag(ax, 50, yc, c)
        seg(ax, [51.9, 53.3], [yc, yc])
        block(53.5, yc, H(80), "c")
        seg(ax, [60.7, 62.2], [yc, yc])
        bn_tag(ax, 64.1, yc, c)
        seg(ax, [66, 79.3], [yc, yc])
        block(79.5, yc, H(80), "e")

        for y in (yb, yc):
            seg(ax, [86.7, 94.2], [y, y])
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
    top, bottom = T2 - 8, T2 - 66.5
    for x0, x1, title in ((34, 70, "block 2 \u00b7 shared kernels"), (73.5, 92.5, "predictor weight")):
        ax.add_patch(FancyBboxPatch((x0, bottom), x1 - x0, top - bottom, boxstyle="round,pad=0.2,rounding_size=1.5",
                                    fc="#F6F2FD", ec=SHARED, lw=1.2, ls="--", zorder=0.8))
        ax.text((x0 + x1) / 2, top - 2.4, title, ha="center", fontsize=9.5, color=SHARED, fontweight="bold")
    for s, dy in (("P", -21), ("M", -37), ("G", -53)):
        scale_lanes(T2 + dy, s, shared=True)
    chip_row(ax, 34, 70, T2 - 63.3, ["a box 3\u00d73", "b class DW", "c class PW"])
    chip_row(ax, 73.5, 92.5, T2 - 63.3, ["d box", "e class"])

    note(123, T2 - 14.5, "one kernel, three maps",
         "each stored tensor is applied to the\nP, M and G maps separately, with\nthe same weight values")
    note(123, T2 - 26, "still per scale", "block 1, every BatchNorm and the\npredictor bias")
    note(123, T2 - 35.5, "not tied to each other", "box and class families, and the\none-to-many and one-to-one heads")
    note(123, T2 - 44.5, "no recurrence", "no loop, feedback, concat or sum of\nthe P, M and G maps")

    ly = T2 - 71.5
    weight_chip(ax, 5, ly, "stored weight")
    ax.text(5 + chip_width("stored weight") + 1.2, ly, "one tensor, drawn once", va="center", fontsize=8.5, color=GREY)
    box(ax, 44, ly, 7, 2.4, SHARED, "x", fs=7, alpha=0.14, lw=1.2, bold=True, tc=SHARED, ls="--")
    ax.text(52.5, ly, "one application of it, on one scale's map", va="center", fontsize=8.5, color=GREY)
    bn_tag(ax, 93, ly, SCALE["M"])
    ax.text(95.6, ly, "per-scale BN", va="center", fontsize=8.5, color=GREY)
    bias_dot(ax, 110.5, ly, SCALE["M"])
    ax.text(112.2, ly, "bias", va="center", fontsize=8.5, color=GREY)
    arrow(ax, (123, ly), (129, ly))
    ax.text(130.2, ly, "lanes stay per scale", va="center", fontsize=8.5, color=GREY)

    ax.text(CX, T2 - 76.4,
            "The P, M and G maps (64 ch 64\u00d764, 128 ch 32\u00d732, 256 ch 16\u00d716 at 512 px) stay separate: the same stored "
            "kernel is applied to each one on its own.\n"
            "Class block 2 stores two tensors, b (depthwise 3\u00d73) and c (pointwise 1\u00d71), each followed by its own per-scale BN. "
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
