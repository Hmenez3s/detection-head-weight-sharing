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
INK = "#1F2430"
GREY = "#6B7280"
H80 = 9.0                      # drawn height for 80 channels; height is proportional to width
H16 = H80 * 16 / 80


def box(ax, x, y, w, h, color, text="", fs=9, alpha=.22, tc=INK, lw=1.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
                                fc=color, ec=color, lw=lw, alpha=1, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
                                fc="white", ec="none", alpha=1 - alpha, zorder=2.1))
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
                                fc="none", ec=color, lw=lw, zorder=2.2))
    if text:
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3,
                fontweight="bold" if bold else "normal")


def arrow(ax, a, b, color=GREY, rad=0., lw=1.3, style="-|>", ms=10, z=1.5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(19, 11), dpi=180)
ax.set_xlim(0, 180); ax.set_ylim(-12, 104); ax.axis("off")
ax.text(90, 101, "YOLO26n head: original  vs.  proposed looped head", ha="center", va="center",
        fontsize=19, fontweight="bold", color=INK)
ax.text(90, 97.3, "block height = width (number of channels)   ·   color = set of weights",
        ha="center", va="center", fontsize=11, color=GREY)
ax.plot([90, 90], [9, 94], color="#D1D5DB", lw=1.2)

rows = {"P": (77, "64 channels\n64×64 grid"),
        "M": (58, "128 channels\n32×32 grid"),
        "G": (39, "256 channels\n16×16 grid")}

# ---------------------------------------------------------------- ORIGINAL
ax.text(45, 92, "ORIGINAL — 3 separate heads", ha="center", fontsize=15, fontweight="bold", color=INK)
ax.text(45, 88.6, "each scale has its own weights  ·  no repetition", ha="center", fontsize=10.5, color=GREY)
for s, (y, desc) in rows.items():
    c = SCALE[s]
    box(ax, 2, y, 15, 11, c, f"{s}\n{desc}", fs=8.6, alpha=.12)
    yb, yc = y + 4.6, y - 3.2
    ax.text(21.5, yb + 2.2, "box · 16", fontsize=7.5, color=GREY)
    ax.text(21.5, yc + H80 / 2 + .9, "class · 80", fontsize=7.5, color=GREY)
    arrow(ax, (17.3, y + 1), (21.5, yb)); arrow(ax, (17.3, y - 1), (21.5, yc))
    for i, x in enumerate((21.5, 34.5)):
        box(ax, x, yb, 10.5, H16, c, "conv 3×3", fs=7, alpha=.55)
        box(ax, x, yc, 10.5, H80, c, "separable\nblock\n3×3", fs=7.6, alpha=.22)
        if i == 0:
            arrow(ax, (32.2, yb), (34.3, yb)); arrow(ax, (32.2, yc), (34.3, yc))
    arrow(ax, (45.2, yb), (48.8, yb)); arrow(ax, (45.2, yc), (48.8, yc))
    box(ax, 49, yb, 11, H16, c, "output: box", fs=7, alpha=.55)
    box(ax, 49, yc, 11, 3.4, c, "output: 80 classes", fs=7, alpha=.35)
    arrow(ax, (60.3, yb), (64, y + .6)); arrow(ax, (60.3, yc), (64, y - .6))
    ax.text(64.6, y, f"{s} predictions", va="center", fontsize=9, color=c, fontweight="bold")

# ---------------------------------------------------------------- PROPOSED
ax.text(135, 92, "PROPOSED — 1 shared head, looped", ha="center", fontsize=15, fontweight="bold", color=INK)
ax.text(135, 88.6, "P, M and G share the same weights (purple)  ·  box and class stay separate  ·  adapters are per scale",
        ha="center", fontsize=10, color=GREY)
YB, YC = 74, 52            # shared box tower, shared class tower
ax.text(113.25, 85.4, "adapters", ha="center", fontsize=8.2, color=GREY)
for s, (y, desc) in rows.items():
    c = SCALE[s]
    box(ax, 92, y, 15, 11, c, f"{s}\n{desc}", fs=8.6, alpha=.12)
    yb, yc = y + 4.6, y - 3.2
    arrow(ax, (107.3, y + 1), (109.8, yb)); arrow(ax, (107.3, y - 1), (109.8, yc))
    box(ax, 110, yb, 6.5, H16, c, "", alpha=.55)
    ax.text(113.25, yb + 1.9, "sep 3×3 → 16", ha="center", fontsize=6.2, color=GREY)
    box(ax, 110, yc, 6.5, H80, c, "sep\n3×3\n→ 80", fs=7, alpha=.3)
    # each scale feeds both shared towers; route color = scale
    arrow(ax, (116.8, yb), (121.6, YB), color=c, lw=1.1); arrow(ax, (116.8, yc), (121.6, YC), color=c, lw=1.1)
ax.add_patch(FancyBboxPatch((120.6, 41), 30.6, 42, boxstyle="round,pad=0.2,rounding_size=1.5",
                            fc="#F4EFFC", ec=SHARED, lw=1.2, ls="--", zorder=.8))
ax.text(137.5, 84.9, "3 passes · same weights", ha="center", fontsize=9.6, color=SHARED, fontweight="bold")
ax.text(135.9, 39.2, "the SAME blocks serve P, M and G", ha="center", fontsize=8.6, color=SHARED, fontweight="bold")
ax.text(135.9, 66.6, "h ← h + 0.1 · F(h)   (skip keeps h)", ha="center", fontsize=9, color=SHARED)


def looped_tower(yt, height, names, texts, rad, lift):
    box(ax, 124, yt, 8.6, height, SHARED, texts[0], fs=7.4, alpha=.3)
    arrow(ax, (132.8, yt), (134.4, yt), color=SHARED)
    box(ax, 134.6, yt, 8.6, height, SHARED, texts[1], fs=7.4, alpha=.3)
    ax.add_patch(Circle((146.6, yt), 1.25, fc="white", ec=SHARED, lw=1.5, zorder=3))
    ax.text(146.6, yt, "+", ha="center", va="center", fontsize=11, color=SHARED, fontweight="bold", zorder=4)
    arrow(ax, (143.4, yt), (145.2, yt), color=SHARED)
    low = yt - height / 2 - 1.8
    ax.plot([122.3, 122.3, 146.6], [yt, low, low], color=SHARED, lw=1.2, zorder=1.6)
    arrow(ax, (146.6, low), (146.6, yt - 1.35), color=SHARED)
    arrow(ax, (147.6, yt + .9), (122.6, yt + lift), color=SHARED, rad=rad, lw=2, ms=14)
    if not texts[0]:
        for x, n in zip((128.3, 138.9), names):
            ax.text(x, yt + 1.7, n, ha="center", fontsize=8, color=SHARED, fontweight="bold")


ax.plot([121.6, 124], [YB, YB], color=SHARED, lw=1.2); ax.plot([121.6, 124], [YC, YC], color=SHARED, lw=1.2)
looped_tower(YB, H16, ("C", "D"), ("", ""), .3, 1.2)
ax.text(124, YB - 4.4, "box tower · 16 ch · conv 3×3", fontsize=7.6, color=SHARED)
looped_tower(YC, H80, ("A", "B"), ("A\nseparable\nblock\n3×3 · 80", "B\nseparable\nblock\n3×3 · 80"), .4, 5.4)
ax.text(124, YC - 8.3, "class tower · 80 ch · separable", fontsize=7.6, color=SHARED)
arrow(ax, (147.9, YB), (152.3, YB), color=SHARED); arrow(ax, (147.9, YC), (152.3, YC), color=SHARED)
box(ax, 152.5, YB, 10.5, 3.4, SHARED, "output: box", fs=7, alpha=.35)
box(ax, 152.5, YC, 10.5, 3.4, SHARED, "output: 80 classes", fs=7, alpha=.35)
ax.plot([163.3, 164.6], [YB, 63], color=GREY, lw=1.3); ax.plot([163.3, 164.6], [YC, 63], color=GREY, lw=1.3)
ax.add_patch(Circle((164.6, 63), .45, fc=GREY, ec=GREY, zorder=3))
for s, (y, _) in rows.items():
    arrow(ax, (164.6, 63), (168.3, y), color=SCALE[s], lw=1.2)
    ax.text(168.7, y, f"{s} predictions", va="center", fontsize=9, color=SCALE[s], fontweight="bold")

# ---------------------------------------------------------------- UNROLLED DEPTH
ax.text(90, 27.6, "The path of one location to its prediction, unrolled", ha="center", fontsize=12.5,
        fontweight="bold", color=INK)
y = 18
ax.text(3, y + 6.2, "original (class tower, scale P)", fontsize=10, fontweight="bold", color=INK)
box(ax, 3, y, 8, 5, SCALE["P"], "input", fs=7.5, alpha=.12)
x = 13
for label in ("sep", "sep"):
    arrow(ax, (x - 1.8, y), (x - .2, y)); box(ax, x, y, 7, H80 * .7, SCALE["P"], label, fs=8, alpha=.25); x += 9
arrow(ax, (x - 1.8, y), (x - .2, y)); box(ax, x, y, 8, 3.4, SCALE["P"], "output", fs=7.5, alpha=.35)
ax.text(3, y - 6.6, "2 blocks, each used once", fontsize=9, color=GREY)

ax.text(93, y + 6.2, "proposed (class tower; the box tower repeats C, D the same way)", fontsize=10, fontweight="bold", color=INK)
box(ax, 93, y, 8, 5, SCALE["P"], "input", fs=7.5, alpha=.12)
arrow(ax, (101.2, y), (102.8, y)); box(ax, 103, y, 6.5, H80 * .7, SCALE["P"], "adapt.", fs=7.3, alpha=.3)
x = 111.5
for turn in range(3):
    arrow(ax, (x - 1.8, y), (x - .2, y))
    box(ax, x, y, 5, H80 * .7, SHARED, "A", fs=8.5, alpha=.3)
    box(ax, x + 5.6, y, 5, H80 * .7, SHARED, "B", fs=8.5, alpha=.3)
    ax.text(x + 5.3, y + 4.6, f"pass {turn + 1}", ha="center", fontsize=7.6, color=SHARED)
    x += 13
arrow(ax, (x - 1.8, y), (x - .2, y)); box(ax, x, y, 8, 3.4, SHARED, "output", fs=7.5, alpha=.35)
ax.text(93, y - 6.6, "6 blocks deep, but only 2 sets of weights (A and B), reused across 3 passes and 3 scales",
        fontsize=9, color=GREY)

# ---------------------------------------------------------------- NUMBERS
rows_txt = [("", "original", "proposed"),
            ("width", "box 16 · class 80", "box 16 · class 80 (still separate)"),
            ("depth", "2 blocks", "adapter + 6 blocks"),
            ("repetitions", "none", "3 passes"),
            ("blocks with their own weights", "12", "4 shared + 6 adapters"),
            ("head compute", "385 M MACs", "≈ 782 M MACs (≈ 2×)")]
ax.add_patch(FancyBboxPatch((28, -8.2), 124, 14.6, boxstyle="round,pad=0.2,rounding_size=1",
                            fc="#F9FAFB", ec="#E5E7EB", lw=1, zorder=.5))
for i, (a, b, c) in enumerate(rows_txt):
    yy = 5 - i * 2.45
    weight = "bold" if i == 0 else "normal"
    ax.text(62, yy, a, ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(84, yy, b, ha="center", va="center", fontsize=10, color=SCALE["P"] if i == 0 else INK, fontweight=weight)
    ax.text(126, yy, c, ha="center", va="center", fontsize=10, color=SHARED if i == 0 else INK, fontweight=weight)
ax.text(90, -10.4, "YOLO26n keeps two internal copies of its head (one assists training, the other makes the final predictions). "
        "The figure shows one; compute counts both. Proposed values are estimated with a formula that reproduces the measured original.",
        ha="center", va="center", fontsize=8.6, color=GREY)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
