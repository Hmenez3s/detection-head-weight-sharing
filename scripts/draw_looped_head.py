#!/usr/bin/env python3
"""Draw assets/looped-head.png: YOLO26n's three heads vs. one looped head shared by all scales.

Block height is proportional to channel width; colors mark separate sets of weights.
Proposed-head compute is a design estimate.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parents[1] / "assets" / "looped-head.png"
SCALE = {"P": "#3B76C4", "M": "#E8862A", "G": "#3E9B4F"}
SHARED = "#7B4FC9"
INK, GREY, LINE = "#1F2430", "#6B7280", "#9CA3AF"
H80 = 9.0                      # drawn height of 80 channels; height is proportional to width
H16 = H80 * 16 / 80
ROWS = {"P": (77, "64 ch · 64×64"), "M": (58, "128 ch · 32×32"), "G": (39, "256 ch · 16×16")}


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


def scale_input(ax, x, s, y, desc):
    box(ax, x, y, 14, 9, SCALE[s], "", alpha=.1)
    ax.text(x + 7, y + 1.5, s, ha="center", va="center", fontsize=13, fontweight="bold", color=SCALE[s], zorder=3)
    ax.text(x + 7, y - 2, desc, ha="center", va="center", fontsize=8, color=INK, zorder=3)


fig, ax = plt.subplots(figsize=(19, 10.8), dpi=180)
ax.set_xlim(0, 180); ax.set_ylim(-11.5, 104); ax.axis("off")
ax.text(90, 101, "YOLO26n head: original  vs.  proposed looped head", ha="center", va="center",
        fontsize=19, fontweight="bold", color=INK)
ax.text(90, 97.4, "block height = number of channels   ·   color = set of weights", ha="center", va="center",
        fontsize=11, color=GREY)
seg(ax, [90, 90], [31, 94], "#E5E7EB", 1.2)
seg(ax, [90, 90], [9, 24], "#E5E7EB", 1.2)

# ---------------------------------------------------------------- ORIGINAL
ax.text(45, 91.5, "ORIGINAL — 3 separate heads", ha="center", fontsize=15, fontweight="bold", color=INK)
ax.text(45, 88.2, "each scale has its own weights  ·  each block runs once", ha="center", fontsize=10.5, color=GREY)
for s, (y, desc) in ROWS.items():
    c = SCALE[s]
    scale_input(ax, 4, s, y, desc)
    yb, yc = y + 4.6, y - 3.2
    arrow(ax, (18.3, y + 1), (22.8, yb)); arrow(ax, (18.3, y - 1), (22.8, yc))
    for x in (23, 35):
        box(ax, x, yb, 9, H16, c, alpha=.55)
        box(ax, x, yc, 9, H80, c, alpha=.25)
    seg(ax, [32.2, 34.8], [yb, yb]); seg(ax, [32.2, 34.8], [yc, yc])
    seg(ax, [44.2, 46.8], [yb, yb]); seg(ax, [44.2, 46.8], [yc, yc])
    box(ax, 47, yb, 3, H16, c, alpha=.8); box(ax, 47, yc, 3, H80, c, alpha=.5)
    seg(ax, [50.2, 53.5], [yb, y]); seg(ax, [50.2, 53.5], [yc, y])
    arrow(ax, (53.5, y), (56.5, y))
    ax.text(57.2, y, f"{s} predictions", va="center", fontsize=9.5, color=c, fontweight="bold")
ax.text(36.5, 84.4, "thin row: box · conv 3×3 · 16 ch     tall row: class · separable 3×3 · 80 ch     last: output",
        ha="center", fontsize=7.8, color=GREY)

# ---------------------------------------------------------------- PROPOSED
ax.text(135, 91.5, "PROPOSED — 1 shared head, looped", ha="center", fontsize=15, fontweight="bold", color=INK)
ax.text(135, 88.2, "the same purple weights serve P, M and G  ·  box and class stay separate",
        ha="center", fontsize=10.5, color=GREY)
YB, YC = 76, 52                 # shared box tower, shared class tower
BOX_BUS, CLS_BUS = 118.6, 120.6
ax.text(113, 84.6, "adapters · sep 3×3", ha="center", fontsize=7.8, color=GREY)
box_ys, cls_ys = [], []
for s, (y, desc) in ROWS.items():
    c = SCALE[s]
    scale_input(ax, 93, s, y, desc)
    yb, yc = y + 4.6, y - 3.2
    box_ys.append(yb); cls_ys.append(yc)
    arrow(ax, (107.3, y + 1), (110.8, yb)); arrow(ax, (107.3, y - 1), (110.8, yc))
    box(ax, 111, yb, 4, H16, c, alpha=.6)
    box(ax, 111, yc, 4, H80, c, alpha=.35)
    seg(ax, [115.2, BOX_BUS], [yb, yb])
    seg(ax, [115.2, CLS_BUS], [yc, yc])
# two buses gather the three scales; the box bus is broken where class stubs pass
crossings = sorted(v for v in cls_ys if min(box_ys) < v < max(box_ys))
edges = [min(box_ys)] + [e for v in crossings for e in (v - .9, v + .9)] + [max(box_ys)]
for lo, hi in zip(edges[::2], edges[1::2]):
    seg(ax, [BOX_BUS, BOX_BUS], [lo, hi])
seg(ax, [CLS_BUS, CLS_BUS], [min(cls_ys), max(cls_ys)])
arrow(ax, (BOX_BUS, YB), (125.8, YB), SHARED, lw=1.4)
arrow(ax, (CLS_BUS, YC), (125.8, YC), SHARED, lw=1.4)

ax.add_patch(FancyBboxPatch((122.6, 42.5), 31.6, 41.8, boxstyle="round,pad=0.2,rounding_size=1.5",
                            fc="#F6F2FD", ec=SHARED, lw=1.2, ls="--", zorder=.8))
ax.text(138.4, 82.3, "shared by P, M and G", ha="center", fontsize=9.5, color=SHARED, fontweight="bold")
ax.text(138.4, 67.6, "each pass:  h ← h + 0.1 · F(h)", ha="center", fontsize=9, color=SHARED)


def tower(yt, height, names, rad, lift):
    x1, x2, cx = 126, 136.2, 147.6
    for x, n in zip((x1, x2), names):
        box(ax, x, yt, 8, height, SHARED, n if height > 4 else "", fs=10, alpha=.3, bold=True, tc=SHARED)
        if height <= 4:
            ax.text(x + 4, yt + 1.9, n, ha="center", fontsize=8.5, color=SHARED, fontweight="bold")
    arrow(ax, (134.2, yt), (136.0, yt), SHARED)
    arrow(ax, (144.4, yt), (146.3, yt), SHARED)
    ax.add_patch(Circle((cx, yt), 1.2, fc="white", ec=SHARED, lw=1.5, zorder=3))
    ax.text(cx, yt, "+", ha="center", va="center", fontsize=11, color=SHARED, fontweight="bold", zorder=4)
    low = yt - height / 2 - 1.8
    seg(ax, [124.3, 124.3, cx], [yt, low, low], SHARED, 1.2, 1.6)
    arrow(ax, (cx, low), (cx, yt - 1.3), SHARED)
    arrow(ax, (cx + .9, yt + .9), (125.5, yt + lift), SHARED, rad=rad, lw=2, ms=14)
    ax.text(cx + 2.1, yt + 3.1, "×3", fontsize=10, color=SHARED, fontweight="bold")


tower(YB, H16, ("C", "D"), .3, 1.2)
tower(YC, H80, ("A", "B"), .4, 5.2)
for yt, h, label in ((YB, H16, "boxes"), (YC, H80, "classes")):
    seg(ax, [148.8, 156.8], [yt, yt], SHARED)
    box(ax, 157, yt, 2.6, h, SHARED, alpha=.55)
    arrow(ax, (159.8, yt), (162.6, yt))
    ax.text(163.2, yt, f"{label} for P, M, G", va="center", fontsize=9.5, color=INK, fontweight="bold")

# ---------------------------------------------------------------- UNROLLED PATH
ax.text(90, 27.6, "The path of one location, unrolled", ha="center", fontsize=12.5, fontweight="bold", color=INK)
y = 16.5
ax.text(4, y + 5.6, "original · class tower", fontsize=9.8, fontweight="bold", color=INK)
box(ax, 4, y, 6, 4.6, SCALE["P"], "in", fs=8, alpha=.1)
x = 12.5
for _ in range(2):
    arrow(ax, (x - 2.2, y), (x - .2, y)); box(ax, x, y, 6.5, H80 * .7, SCALE["P"], "sep", fs=8, alpha=.25); x += 9
arrow(ax, (x - 2.2, y), (x - .2, y)); box(ax, x, y, 6, 4.6, SCALE["P"], "out", fs=8, alpha=.4)
ax.text(4, y - 6.4, "2 blocks, each used once", fontsize=9, color=GREY)

ax.text(94, y + 5.6, "proposed · class tower", fontsize=9.8, fontweight="bold", color=INK)
box(ax, 94, y, 6, 4.6, SCALE["P"], "in", fs=8, alpha=.1)
arrow(ax, (100.3, y), (102.3, y)); box(ax, 102.5, y, 6.2, H80 * .7, SCALE["P"], "adapt", fs=7.6, alpha=.3)
x = 111.5
for turn in range(3):
    arrow(ax, (x - 2.5, y), (x - .2, y))
    box(ax, x, y, 5, H80 * .7, SHARED, "A", fs=8.5, alpha=.3, tc=SHARED, bold=True)
    box(ax, x + 5.6, y, 5, H80 * .7, SHARED, "B", fs=8.5, alpha=.3, tc=SHARED, bold=True)
    ax.text(x + 5.3, y + 4.4, f"pass {turn + 1}", ha="center", fontsize=7.6, color=SHARED)
    x += 13.5
arrow(ax, (x - 2.5, y), (x - .2, y)); box(ax, x, y, 6, 4.6, SHARED, "out", fs=8, alpha=.4, tc=SHARED)
ax.text(94, y - 6.4, "6 blocks deep · only 2 sets of weights (A, B), reused in every pass and every scale",
        fontsize=9, color=GREY)

# ---------------------------------------------------------------- NUMBERS
rows_txt = [("", "original", "proposed"),
            ("width", "box 16 · class 80", "box 16 · class 80"),
            ("depth", "2 blocks", "adapter + 6 blocks"),
            ("repetitions", "none", "3 passes"),
            ("blocks with their own weights", "12", "4 shared + 6 adapters"),
            ("head compute", "385 M MACs", "≈ 782 M MACs (≈ 2×)")]
ax.add_patch(FancyBboxPatch((30, -8.3), 120, 14.2, boxstyle="round,pad=0.2,rounding_size=1",
                            fc="#F9FAFB", ec="#E5E7EB", lw=1, zorder=.5))
for i, (a, b, c) in enumerate(rows_txt):
    yy = 4.4 - i * 2.35
    weight = "bold" if i == 0 else "normal"
    ax.text(64, yy, a, ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(85, yy, b, ha="center", va="center", fontsize=10, color=SCALE["P"] if i == 0 else INK, fontweight=weight)
    ax.text(124, yy, c, ha="center", va="center", fontsize=10, color=SHARED if i == 0 else INK, fontweight=weight)
ax.text(90, -10.4, "YOLO26n runs two internal copies of its head; the figure shows one and compute counts both. "
        "Proposed compute is an estimate.", ha="center", va="center", fontsize=8.6, color=GREY)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
