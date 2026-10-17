#!/usr/bin/env python3
"""Draw assets/looped-head.png: YOLO26n's three heads (top) vs. one shared looped head (bottom).

Block height is proportional to channel width; colors mark separate sets of weights.
Proposed-head compute is a design estimate, not a measurement.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parents[1] / "assets" / "looped-head.png"
SCALE = {"P": "#3B76C4", "M": "#E8862A", "G": "#3E9B4F"}
SHARED = "#7B4FC9"
INK, GREY, LINE, RULE = "#1F2430", "#6B7280", "#9CA3AF", "#E5E7EB"
H80 = 9.0                                   # drawn height of 80 channels
H = lambda ch: H80 * ch / 80                # block height is proportional to channels
DESC = {"P": "64 ch · 64×64", "M": "128 ch · 32×32", "G": "256 ch · 16×16"}
POINTS = {"P": 4096, "M": 1024, "G": 256}   # locations per image = map height × width


def box(ax, x, y, w, h, color, text="", fs=9, alpha=.22, lw=1.6, bold=False, tc=INK):
    style = "round,pad=0.15,rounding_size=0.5"
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=style, fc=color, ec="none", alpha=alpha, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=style, fc="none", ec=color, lw=lw, zorder=2.2))
    if text:
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3,
                fontweight="bold" if bold else "normal")


def arrow(ax, a, b, color=LINE, rad=0., lw=1.3, ms=10):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1.5, shrinkA=0, shrinkB=0))


def seg(ax, xs, ys, color=LINE, lw=1.3, z=1.4):
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round")


def scale_input(ax, x, s, y):
    box(ax, x, y, 14, 9, SCALE[s], "", alpha=.1)
    ax.text(x + 7, y + 1.5, s, ha="center", va="center", fontsize=13, fontweight="bold", color=SCALE[s], zorder=3)
    ax.text(x + 7, y - 2, DESC[s], ha="center", va="center", fontsize=8, color=INK, zorder=3)


def point_list(ax, x, y, length, height=2.2):
    """One list of all locations: P, M and G segments in proportion to their point counts."""
    total = sum(POINTS.values())
    for s, n in POINTS.items():
        w = length * n / total
        ax.add_patch(Rectangle((x, y - height / 2), w, height, fc=SCALE[s], ec="white", lw=.8, alpha=.85, zorder=3))
        x += w


def section(ax, y, title, subtitle):
    ax.text(85, y, title, ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(85, y - 3.6, subtitle, ha="center", fontsize=10.5, color=GREY)


fig, ax = plt.subplots(figsize=(17, 21.4), dpi=170)
ax.set_xlim(0, 170); ax.set_ylim(-2, 212); ax.axis("off")
ax.text(85, 208.5, "YOLO26n head: original vs. proposed shared looped head", ha="center", va="center",
        fontsize=20, fontweight="bold", color=INK)
ax.text(85, 204.6, "block height = number of channels   ·   color = set of weights", ha="center", va="center",
        fontsize=11, color=GREY)

# ---------------------------------------------------------------- ORIGINAL (top)
section(ax, 196, "ORIGINAL — 3 separate heads × 2 layers",
        "each scale has its own weights  ·  each head runs on its own map")
ax.text(46, 186.2, "layer 1", ha="center", fontsize=9, color=GREY)
ax.text(64, 186.2, "layer 2", ha="center", fontsize=9, color=GREY)
ax.text(78, 186.2, "output", ha="center", fontsize=9, color=GREY)
for s, y in (("P", 176), ("M", 158), ("G", 140)):
    c = SCALE[s]
    scale_input(ax, 6, s, y)
    yb, yc = y + 4.6, y - 3.2
    arrow(ax, (20.3, y + 1), (34.8, yb)); arrow(ax, (20.3, y - 1), (34.8, yc))
    for x in (35, 53):
        box(ax, x, yb, 11, H(16), c, alpha=.55)
        box(ax, x, yc, 11, H(80), c, alpha=.25)
    seg(ax, [46.2, 52.8], [yb, yb]); seg(ax, [46.2, 52.8], [yc, yc])
    seg(ax, [64.2, 76.3], [yb, yb]); seg(ax, [64.2, 76.3], [yc, yc])
    box(ax, 76.5, yb, 3, H(4) + .4, c, alpha=.8); box(ax, 76.5, yc, 3, H(80), c, alpha=.5)
    seg(ax, [79.7, 84], [yb, y]); seg(ax, [79.7, 84], [yc, y])
    arrow(ax, (84, y), (87.5, y))
    ax.text(88.4, y, f"{s} predictions", va="center", fontsize=10, color=c, fontweight="bold")
ax.text(106, 172, "thin row: box  ·  conv 3×3  ·  16 ch", fontsize=10, color=INK)
ax.text(106, 167.5, "tall row: class  ·  separable 3×3  ·  80 ch", fontsize=10, color=INK)
ax.text(106, 160, "Layer 1 also compresses each scale\n(64 / 128 / 256 ch) to the tower width.",
        fontsize=9.5, color=GREY, va="top", linespacing=1.5)
ax.text(106, 150, "What the P head learns never\nreaches M or G.", fontsize=9.5, color=GREY, va="top",
        linespacing=1.5)
seg(ax, [6, 164], [131, 131], RULE, 1.2)

# ---------------------------------------------------------------- PROPOSED (bottom)
section(ax, 124, "PROPOSED — 1 shared head: 2 passes × 2 layers, run once on all points",
        "per-scale adapters, then the same purple weights for P, M and G  ·  box and class stay separate")
YB, YC = 108, 84                             # shared box tower, shared class tower
ax.text(32, 115.8, "adapter per scale\nsep 3×3 + scale ID", ha="center", va="bottom", fontsize=8.6,
        color=GREY, linespacing=1.3)
box_ys, cls_ys = [], []
BOX_BUS, CLS_BUS = 41.6, 43.6
for s, y in (("P", 104), ("M", 86), ("G", 68)):
    c = SCALE[s]
    scale_input(ax, 6, s, y)
    yb, yc = y + 4.6, y - 3.4
    box_ys.append(yb); cls_ys.append(yc)
    arrow(ax, (20.3, y + 1), (26.8, yb)); arrow(ax, (20.3, y - 1), (26.8, yc))
    box(ax, 27, yb, 5, H(24), c, alpha=.6)
    box(ax, 27, yc, 5, H(96), c, alpha=.35)
    for yy in (yb, yc):
        seg(ax, [32.2, 35.3], [yy, yy])
        ax.add_patch(Circle((36.6, yy), 1.25, fc="white", ec=c, lw=1.3, zorder=3))
        ax.text(36.6, yy, "id", ha="center", va="center", fontsize=6.5, color=c, fontweight="bold", zorder=4)
    seg(ax, [37.9, BOX_BUS], [yb, yb]); seg(ax, [37.9, CLS_BUS], [yc, yc])
# two buses gather the three scales; the box bus is broken where class stubs pass
crossings = sorted(v for v in cls_ys if min(box_ys) < v < max(box_ys))
edges = [min(box_ys)] + [e for v in crossings for e in (v - .9, v + .9)] + [max(box_ys)]
for lo, hi in zip(edges[::2], edges[1::2]):
    seg(ax, [BOX_BUS, BOX_BUS], [lo, hi])
seg(ax, [CLS_BUS, CLS_BUS], [min(cls_ys), max(cls_ys)])
for bus, yt in ((BOX_BUS, YB), (CLS_BUS, YC)):
    seg(ax, [bus, 47], [yt, yt])
    point_list(ax, 47, yt, 11)
    arrow(ax, (58.2, yt), (64.6, yt), SHARED, lw=1.4)
ax.text(52.5, YB + 2.8, "1 list · 5,376 points", ha="center", fontsize=8.6, color=INK, fontweight="bold")
ax.text(52.5, YB - 2.6, "4,096 P · 1,024 M · 256 G", ha="center", va="top", fontsize=7.3, color=GREY)

ax.add_patch(FancyBboxPatch((61, 70.5), 46, 48.6, boxstyle="round,pad=0.2,rounding_size=1.5",
                            fc="#F6F2FD", ec=SHARED, lw=1.2, ls="--", zorder=.8))
ax.text(84, 116.6, "shared by P, M and G", ha="center", fontsize=10, color=SHARED, fontweight="bold")
ax.text(84, 97.2, "each pass:  h ← h + 0.1 · F(h)", ha="center", fontsize=9.5, color=SHARED)


def tower(yt, height, names, width_label, rad, lift):
    x1, x2, cx = 66, 79, 96.5
    for x, n in zip((x1, x2), names):
        box(ax, x, yt, 10, height, SHARED, n if height > 4 else "", fs=11, alpha=.3, bold=True, tc=SHARED)
        if height <= 4:
            ax.text(x + 5, yt + 2.6, n, ha="center", fontsize=9, color=SHARED, fontweight="bold")
    arrow(ax, (76.2, yt), (78.8, yt), SHARED)
    arrow(ax, (89.2, yt), (95.1, yt), SHARED)
    ax.add_patch(Circle((cx, yt), 1.3, fc="white", ec=SHARED, lw=1.5, zorder=3))
    ax.text(cx, yt, "+", ha="center", va="center", fontsize=11, color=SHARED, fontweight="bold", zorder=4)
    low = yt - height / 2 - 2
    seg(ax, [63.5, 63.5, cx], [yt, low, low], SHARED, 1.2, 1.6)
    arrow(ax, (cx, low), (cx, yt - 1.4), SHARED)
    arrow(ax, (cx + 1, yt + 1), (65.4, yt + lift), SHARED, rad=rad, lw=2, ms=14)
    ax.text(cx + 2.3, yt + 3.2, "×2", fontsize=11, color=SHARED, fontweight="bold")
    ax.text(77.6, low - 2.4, width_label, ha="center", fontsize=8.6, color=SHARED)


tower(YB, H(24), ("C", "D"), "box · two 1×1 layers · 24 ch", .3, 1.6)
tower(YC, H(96), ("A", "B"), "class · two 1×1 layers · 96 ch", .42, 5.8)
for yt, h, label in ((YB, H(4) + .4, "boxes"), (YC, H(80), "classes")):
    seg(ax, [97.8, 110.8], [yt, yt], SHARED)
    box(ax, 111, yt, 3, h, SHARED, alpha=.55)
    seg(ax, [114.2, 117], [yt, yt])
    point_list(ax, 117, yt, 11)
    arrow(ax, (128.2, yt), (131, yt))
    ax.text(131.8, yt, f"{label}, split back\nto P, M and G", va="center", fontsize=9.6, color=INK,
            fontweight="bold", linespacing=1.35)
ax.text(112.5, 113.5, "output", ha="center", fontsize=9, color=GREY)
ax.text(106, 66.5, "Adapters keep each scale's own 3×3 view of its grid. The loop only uses 1×1 layers, so a "
        "point's\ngrid does not matter: P, M and G points can share one list and each layer runs once per pass.",
        ha="center", va="top", fontsize=9.2, color=GREY, linespacing=1.45)
seg(ax, [6, 164], [55.5, 55.5], RULE, 1.2)

# ---------------------------------------------------------------- UNROLLED PATH (stacked)
ax.text(85, 50.5, "The path of one location through the class tower", ha="center", fontsize=12.5,
        fontweight="bold", color=INK)


def path(y, label, steps, note, out_color):
    ax.text(6, y + 5.2, label, fontsize=10, fontweight="bold", color=INK)
    box(ax, 6, y, 6, 4.6, SCALE["P"], "in", fs=8, alpha=.1)
    x, end = 15, 12.2
    for text, color, w, gap, top in steps:
        arrow(ax, (end, y), (x - .2, y))
        end = x + w + .2
        tc = SHARED if color == SHARED else INK
        box(ax, x, y, w, H(80) * .62, color, text, fs=8.5, alpha=.3, tc=tc, bold=color == SHARED)
        if top:
            ax.text(x + w + gap / 2 if top.startswith("pass") else x + w / 2, y + 3.9, top,
                    ha="center", fontsize=7.8, color=SHARED)
        x += w + gap
    arrow(ax, (end, y), (x - .2, y)); box(ax, x, y, 6, 4.6, out_color, "out", fs=8, alpha=.4,
                                              tc=SHARED if out_color == SHARED else INK)
    ax.text(x + 9, y, note, va="center", fontsize=9.3, color=GREY)


path(40, "original", [("sep 3×3 · 80", SCALE["P"], 13, 3.4, ""), ("sep 3×3 · 80", SCALE["P"], 13, 3.4, "")],
     "2 layers, used once, weights of this scale only", SCALE["P"])
path(25, "proposed", [("adapter · 96", SCALE["P"], 13, 3.4, ""), ("A", SHARED, 6, 1.6, "pass 1"),
                      ("B", SHARED, 6, 3.4, ""), ("A", SHARED, 6, 1.6, "pass 2"), ("B", SHARED, 6, 3.4, "")],
     "adapter + 4 layers · A and B reused in both passes and by P, M and G", SHARED)

# ---------------------------------------------------------------- NUMBERS
rows_txt = [("", "original", "proposed"),
            ("width", "box 16 · class 80", "box 24 · class 96"),
            ("depth per tower", "2 layers + output", "adapter + 2 passes × 2 layers + output"),
            ("how often a layer runs", "3×, once per scale", "adapters 3× · loop layers once per pass"),
            ("normalization", "BatchNorm", "per point: no statistics mixed across scales"),
            ("head compute", "385 M MACs (measured)", "≈ 631 M MACs (estimate, ≈ 1.6×)")]
ax.add_patch(FancyBboxPatch((14, 1.2), 142, 15.4, boxstyle="round,pad=0.2,rounding_size=1",
                            fc="#F9FAFB", ec=RULE, lw=1, zorder=.5))
for i, (a, b, c) in enumerate(rows_txt):
    yy = 15 - i * 2.5
    weight = "bold" if i == 0 else "normal"
    ax.text(52, yy, a, ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(76, yy, b, ha="center", va="center", fontsize=10, color=SCALE["P"] if i == 0 else INK,
            fontweight=weight)
    ax.text(122, yy, c, ha="center", va="center", fontsize=10, color=SHARED if i == 0 else INK,
            fontweight=weight)
ax.text(85, -0.8, "YOLO26n runs two internal copies of its head; the figure shows one and compute counts both. "
        "Nothing has been trained; proposed numbers are estimates.", ha="center", va="center", fontsize=8.6,
        color=GREY)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
