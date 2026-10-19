#!/usr/bin/env python3
"""Draw the shared-spatial study figures from the committed result tables.

    MPLBACKEND=Agg .venv/bin/python scripts/draw_shared_spatial.py

The architecture drawing is a static diagram of the registered head. It reuses the native
panel and the looped-panel composition of scripts/draw_looped_head.py (inputs left, one purple
shared region, outputs right) and changes only what differs: no loop, no concat, separate maps.
The charts read results/shared-spatial/five_seed.csv and screen.csv. No model is imported and
no fit is launched.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import same_color
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
TABLES = ROOT / "results" / "shared-spatial"

SCALE = {"P": "#3B76C4", "M": "#E8862A", "G": "#3E9B4F"}
SHARED = "#7B4FC9"
INK, GREY, LINE, RULE = "#1F2430", "#6B7280", "#9CA3AF", "#E5E7EB"
TEAL = "#0F766E"
H80 = 9.0                                   # drawn height of 80 channels, as in draw_looped_head.py
DESC = {"P": "64 ch \u00b7 64\u00d764", "M": "128 ch \u00b7 32\u00d732", "G": "256 ch \u00b7 16\u00d716"}


def H(ch: float) -> float:
    return H80 * ch / 80                    # block height is proportional to channels


def box(ax, x, y, w, h, color, text="", fs=9, alpha=0.22, lw=1.6, bold=False, tc=INK):
    style = "round,pad=0.15,rounding_size=0.5"
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=style, fc=color, ec="none", alpha=alpha, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=style, fc="none", ec=color, lw=lw, zorder=2.2))
    if text:
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3,
                fontweight="bold" if bold else "normal")


def arrow(ax, a, b, color=LINE, lw=1.3, ms=10):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=1.5))


def seg(ax, xs, ys, color=LINE, lw=1.3):
    ax.plot(xs, ys, color=color, lw=lw, zorder=1.4, solid_capstyle="round")


def scale_input(ax, x, s, y):
    box(ax, x, y, 14, 9, SCALE[s], "", alpha=0.1)
    ax.text(x + 7, y + 1.5, s, ha="center", va="center", fontsize=13, fontweight="bold", color=SCALE[s], zorder=3)
    ax.text(x + 7, y - 2, DESC[s], ha="center", va="center", fontsize=8, color=INK, zorder=3)


def section(ax, y, title, subtitle):
    ax.text(85, y, title, ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(85, y - 3.6, subtitle, ha="center", fontsize=10.5, color=GREY)


def native_panel(ax, T):
    """Copied from draw_looped_head.native_panel."""
    section(ax, T, "NATIVE \u2014 three separate heads, two blocks per branch",
            "weights separate per scale  \u00b7  each head runs on its own map  \u00b7  24 conv applications per internal copy")
    for x, t in ((46, "block 1"), (64, "block 2"), (78, "output")):
        ax.text(x, T - 9.8, t, ha="center", fontsize=9, color=GREY)
    for s, dy in (("P", -20), ("M", -37.5), ("G", -55)):
        y, c = T + dy, SCALE[s]
        scale_input(ax, 6, s, y)
        yb, yc = y + 4.6, y - 3.2
        arrow(ax, (20.3, y + 1), (34.8, yb)); arrow(ax, (20.3, y - 1), (34.8, yc))
        for x in (35, 53):
            box(ax, x, yb, 11, H(16), c, alpha=0.55)
            box(ax, x, yc, 11, H(80), c, alpha=0.25)
        seg(ax, [46.2, 52.8], [yb, yb]); seg(ax, [46.2, 52.8], [yc, yc])
        seg(ax, [64.2, 76.3], [yb, yb]); seg(ax, [64.2, 76.3], [yc, yc])
        box(ax, 76.5, yb, 3, H(4) + 0.4, c, alpha=0.8); box(ax, 76.5, yc, 3, H(80), c, alpha=0.5)
        seg(ax, [79.7, 84], [yb, y]); seg(ax, [79.7, 84], [yc, y])
        arrow(ax, (84, y), (87.5, y))
        ax.text(88.4, y, f"{s} predictions", va="center", fontsize=10, color=c, fontweight="bold")
    x0 = 110
    ax.text(x0, T - 17, "thin row \u00b7 box branch", fontsize=10, color=INK, fontweight="bold")
    ax.text(x0, T - 18.8, "conventional 3\u00d73 conv \u2192 16 ch, twice,\nthen 1\u00d71 \u2192 4 box outputs", fontsize=9.5,
            color=INK, va="top", linespacing=1.4)
    ax.text(x0, T - 29, "tall row \u00b7 class branch", fontsize=10, color=INK, fontweight="bold")
    ax.text(x0, T - 30.8, "separable 3\u00d73 \u2192 80 ch, twice,\nthen 1\u00d71 \u2192 80 class outputs", fontsize=9.5,
            color=INK, va="top", linespacing=1.4)
    ax.text(x0, T - 40.6, "each separable block = depthwise 3\u00d73 + pointwise 1\u00d71,\ntwo primitive convs",
            fontsize=9, color=GREY, va="top", linespacing=1.4)
    ax.text(x0, T - 47.6, "Block 1 adapts each scale (64 / 128 / 256 ch)\nto the tower width. Head weights are not shared\n"
            "across P, M and G.", fontsize=9, color=GREY, va="top", linespacing=1.4)
    seg(ax, [6, 164], [T - 65, T - 65], RULE, 1.2)


# A map stack is the input/output glyph of the shared region: three gapped bars, P over M over G,
# one per map. It replaces the old concat bar; a P|M|G row would claim a concat this head never does.
MAP_W, MAP_H, MAP_GAP = 8.0, 1.0, 1.7
MAP_DY = {"P": MAP_GAP, "M": 0.0, "G": -MAP_GAP}


def map_stack(ax, x, yc):
    for s, dy in MAP_DY.items():
        ax.add_patch(FancyBboxPatch((x, yc + dy - MAP_H / 2), MAP_W, MAP_H, boxstyle="square,pad=0", fc=SCALE[s],
                                    ec="none", alpha=0.85, zorder=3, gid=f"map:{s}"))
    pad = 0.6
    ax.add_patch(FancyBboxPatch((x - pad, yc - MAP_GAP - MAP_H / 2 - pad), MAP_W + 2 * pad, 2 * MAP_GAP + MAP_H + 2 * pad,
                                boxstyle="round,pad=0,rounding_size=0.5", fc="none", ec=LINE, lw=1.0, zorder=2.6))
    return x + MAP_W + pad                  # right edge of the tray


def lane(ax, xs, ys, color, over=False):
    """One scale's wire. Lanes drawn `over` carry a white halo, so a crossing reads as a hop, not a join."""
    effects = [pe.Stroke(linewidth=6, foreground="white"), pe.Normal()] if over else None
    ax.plot(xs, ys, color=color, lw=1.3, zorder=1.45 if over else 1.4, solid_capstyle="butt",
            solid_joinstyle="miter", path_effects=effects)


def shared_panel(ax, T):
    """draw_looped_head.looped_panel with the loop, scale IDs and flatten + concat taken out.

    The coloured lanes end on their own bar of a three-map stack; the stack passes through each
    purple block, drawn once, and leaves as a stack again. The maps are never joined.
    """
    section(ax, T, "SPATIAL_SEPBN_SHAREDPRED \u2014 block 2 and predictor shared by P, M and G, no loop",
            "box 16 \u00b7 class 80 channels  \u00b7  box and class do not share weights  \u00b7  "
            "24 conv applications per internal copy")
    ax.text(32, T - 8.4, "block 1 per scale, per tower: the native first block\n(64 / 128 / 256 ch \u2192 16 box, 80 class)",
            ha="center", va="bottom", fontsize=8.4, color=GREY, linespacing=1.3)
    # The P box stub and the M class stub run straight into their bars; the other lanes turn on two
    # ordered buses (box pair left of class pair), so only class lanes ever hop over box lanes.
    YB, YC = T - 15.4 - MAP_GAP, T - 42.4
    XS = 46.6                                # left edge of the input stacks
    bus = {("box", "M"): 38.2, ("box", "G"): 39.6, ("class", "P"): 42.0, ("class", "G"): 43.4}
    for s, dy in (("P", -20), ("M", -38), ("G", -56)):
        y, c = T + dy, SCALE[s]
        scale_input(ax, 6, s, y)
        yb, yc = y + 4.6, y - 4.4
        arrow(ax, (20.3, y + 1), (26.8, yb)); arrow(ax, (20.3, y - 1), (26.8, yc))
        box(ax, 27, yb, 5, H(16), c, alpha=0.6)
        box(ax, 27, yc, 5, H(80), c, alpha=0.35)
        for tower, y0, yt in (("box", yb, YB), ("class", yc, YC)):
            y1 = yt + MAP_DY[s]
            xv = bus.get((tower, s))
            if xv is None:
                assert abs(y1 - y0) < 1e-9, (tower, s)
                lane(ax, [32.2, XS], [y0, y0], c, tower == "class")
            else:
                lane(ax, [32.2, xv, xv, XS], [y0, y0, y1, y1], c, tower == "class")

    ax.add_patch(FancyBboxPatch((63, T - 56.4), 46.6, 51.4, boxstyle="round,pad=0.2,rounding_size=1.5",
                                fc="#F6F2FD", ec=SHARED, lw=1.2, ls="--", zorder=0.8))
    ax.text(86.3, T - 7.0, "weights shared by P, M and G", ha="center", fontsize=10, color=SHARED, fontweight="bold")
    ax.text(86.3, T - 54.2, "each kernel is applied to P, M and G separately", ha="center", fontsize=9.3,
            color=SHARED)

    bw, gap, x1, xp, xo = 10, 3.4, 67.5, 97, 114
    for yt, ch, label, n_out, out in (
            (YB, 16, "box \u00b7 conv 3\u00d73 \u00b7 BN + SiLU per scale \u00b7 16 ch", 4, "boxes"),
            (YC, 80, "class \u00b7 DW 3\u00d73 \u2192 PW 1\u00d71 \u00b7 BN + SiLU per scale \u00b7 80 ch", 80, "classes")):
        h = H(ch)
        tray = map_stack(ax, XS, yt)
        if ch == 80:                         # two kernels, each its own block
            xs = [x1, x1 + bw + gap]
            for x, layer in zip(xs, ("DW 3\u00d73", "PW 1\u00d71")):
                box(ax, x, yt, bw, h, SHARED, layer, fs=10.5, alpha=0.3, bold=True, tc=SHARED)
            arrow(ax, (xs[0] + bw + 0.2, yt), (xs[1] - 0.2, yt), SHARED)
        else:                                # one 3x3 conv, centred on the class pair
            xs = [x1 + (bw + gap) / 2]
            box(ax, xs[0], yt, bw, h, SHARED, alpha=0.3)
            ax.text(xs[0] + bw / 2, yt + h / 2 + 1.5, "conv 3\u00d73", ha="center", fontsize=9.5, color=SHARED,
                    fontweight="bold")
        arrow(ax, (tray + 0.2, yt), (xs[0] - 0.2, yt), SHARED, lw=1.4)
        arrow(ax, (xs[-1] + bw + 0.2, yt), (xp - 0.2, yt), SHARED)
        ax.text(86.3, yt - h / 2 - 1.8, label, ha="center", va="top", fontsize=8.4, color=SHARED)

        hp = H(n_out) + (0.4 if n_out == 4 else 0)
        box(ax, xp, yt, 3, hp, SHARED, alpha=0.55)
        ax.text(xp + 1.5, yt + hp / 2 + 1.3, f"predictor 1\u00d71 \u2192 {n_out}\nbias per scale", ha="center",
                va="bottom", fontsize=8, color=SHARED, linespacing=1.2)
        seg(ax, [xp + 3.2, xo - 0.6], [yt, yt])
        end = map_stack(ax, xo, yt)
        arrow(ax, (end + 0.2, yt), (end + 3, yt))
        ax.text(end + 3.8, yt, f"{out}, one output\nmap per scale", va="center", fontsize=9.6, color=INK,
                fontweight="bold", linespacing=1.35)
    ax.text(XS + MAP_W / 2, YB + MAP_GAP + MAP_H / 2 + 1.4, "three maps, kept separate", ha="center", fontsize=8.4,
            color=INK, fontweight="bold")
    ax.text(106, T - 60.8, "Block 1 is the native first block and stays per scale. Each purple kernel is stored once per family "
            "and applied to the P, M and G maps\nseparately: the maps are never concatenated, summed or looped, so the head "
            "keeps 24 conv applications per internal copy. Every\nBatchNorm and the predictor bias stay per scale; box and class, "
            "and the one-to-many and one-to-one copies, share nothing.",
            ha="center", va="top", fontsize=9, color=GREY, linespacing=1.45)
    seg(ax, [6, 164], [T - 70.4, T - 70.4], RULE, 1.2)


def _check_shared_stages(ax) -> None:
    """Each shared stage is drawn once (box conv, class DW and PW, two predictors), with no loop
    circles, and the maps only appear as P-over-M-over-G stacks, never as one P|M|G row."""
    outlines = [p for p in ax.patches if isinstance(p, FancyBboxPatch) and p.get_facecolor()[3] == 0
                and same_color(p.get_edgecolor(), SHARED) and p.get_linestyle() in ("-", "solid")]
    assert len(outlines) == 5, len(outlines)
    assert all(type(p) in (FancyBboxPatch, FancyArrowPatch) for p in ax.patches)
    assert not any(t.get_text().startswith("\u00d7") for t in ax.texts)
    bars = [p for p in ax.patches if (p.get_gid() or "").startswith("map:")]
    assert len(bars) == 12 and all(same_color(p.get_facecolor()[:3], SCALE[p.get_gid()[4:]]) for p in bars)
    for a in bars:
        for b in bars:
            ya, yb = a.get_y(), b.get_y()
            if a.get_gid() != b.get_gid():  # different scales must never share a row
                assert ya + a.get_height() <= yb or yb + b.get_height() <= ya, (a.get_gid(), b.get_gid())


def draw_architecture() -> None:
    """Native head against spatial_sepbn_sharedpred, one internal copy, no recurrence."""
    fig, ax = plt.subplots(figsize=(17, 19.0), dpi=150)
    ax.set_xlim(0, 170)
    ax.set_ylim(0, 190)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(85, 186.5, "YOLO26n detection head: native vs. shared spatial weights", ha="center", va="center",
            fontsize=20, fontweight="bold", color=INK)
    ax.text(85, 182.4, "block height = number of channels   \u00b7   blue / orange / green = weights of one scale   \u00b7   "
            "purple = weights shared by P, M and G", ha="center", va="center", fontsize=11, color=GREY)
    ax.text(85, 178.9, "one internal copy is drawn (one-to-many)  \u00b7  the one-to-one copy has the same pattern "
            "with its own weights", ha="center", va="center", fontsize=10, color=GREY)

    native_panel(ax, 172)
    shared_panel(ax, 99)
    _check_shared_stages(ax)

    rows = [("", "native", "spatial_sepbn_sharedpred"),
            ("stored once for P, M and G", "nothing", "block-2 kernels + predictor weight"),
            ("unique head parameters, both internal branches", "309,656", "246,104"),
            ("head ConvMACs, 512 px, batch 1", "384,606,208", "384,606,208"),
            ("full detector ConvMACs", "1,906,679,808", "1,906,679,808")]
    ax.add_patch(FancyBboxPatch((8, 8.4), 154, 17.4, boxstyle="round,pad=0.2,rounding_size=1",
                                fc="#F9FAFB", ec=RULE, lw=1, zorder=0.5))
    for i, row in enumerate(rows):
        yy = 23.6 - i * 3.3
        ax.text(68, yy, row[0], ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
        for xc, txt, col in zip((96, 134), row[1:], (SCALE["P"], SHARED)):
            ax.text(xc, yy, txt, ha="center", va="center", fontsize=10, color=col if i == 0 else INK,
                    fontweight="bold" if i == 0 else "normal")
    ax.text(85, 5.4, "The shared kernel is still executed once per scale, so the ConvMAC count does not fall. "
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
