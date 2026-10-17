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
ax.text(135, 88.6, "P, M and G use the same weights (purple)  ·  only the adapter is per scale",
        ha="center", fontsize=10.5, color=GREY)
cy = 58
for s, (y, desc) in rows.items():
    c = SCALE[s]
    box(ax, 92, y, 15, 11, c, f"{s}\n{desc}", fs=8.6, alpha=.12)
    arrow(ax, (107.3, y), (109.2, y))
    box(ax, 109.4, y, 7.6, H80, c, "adapter\n1×1\n→ 80", fs=7.4, alpha=.3)
    arrow(ax, (117.3, y), (121.6, cy + (y - cy) * .12), rad=.15 if y > cy else (-.15 if y < cy else 0))
# shared looped core
ax.add_patch(FancyBboxPatch((120.3, cy - 13.5), 29.8, 32.5, boxstyle="round,pad=0.2,rounding_size=1.5",
                            fc="#F4EFFC", ec=SHARED, lw=1.2, ls="--", zorder=.8))
ax.text(135.2, cy - 11.9, "the SAME blocks serve P, M and G", ha="center", fontsize=8.6, color=SHARED, fontweight="bold")
box(ax, 123, cy, 9.6, H80, SHARED, "A\nseparable\nblock\n3×3 · 80", fs=7.8, alpha=.3, bold=False)
arrow(ax, (132.9, cy), (134.7, cy), color=SHARED)
box(ax, 135, cy, 9.6, H80, SHARED, "B\nseparable\nblock\n3×3 · 80", fs=7.8, alpha=.3)
ax.add_patch(Circle((147.4, cy), 1.5, fc="white", ec=SHARED, lw=1.6, zorder=3))
ax.text(147.4, cy, "+", ha="center", va="center", fontsize=13, color=SHARED, fontweight="bold", zorder=4)
arrow(ax, (144.9, cy), (145.9, cy), color=SHARED)
# residual skip: carries h around A and B
ax.plot([121.4, 121.4, 147.4], [cy, cy - 7.8, cy - 7.8], color=SHARED, lw=1.3, zorder=1.6)
arrow(ax, (147.4, cy - 7.8), (147.4, cy - 1.6), color=SHARED)
ax.text(134, cy - 9.4, "skip: keep h, add 0.1 · F(h)", ha="center", fontsize=7.8, color=SHARED)
# loop back
arrow(ax, (148.6, cy + 1.3), (121.6, cy + 5.4), color=SHARED, rad=.55, lw=2.2, ms=16)
ax.text(135.2, cy + 16.0, "4 passes, same weights", ha="center", fontsize=10, color=SHARED, fontweight="bold")
ax.text(135.2, cy + 13.2, "h ← h + 0.1 · F(h)", ha="center", fontsize=10, color=SHARED)
# shared outputs
arrow(ax, (148.9, cy), (152.3, cy + 4), color=SHARED); arrow(ax, (148.9, cy), (152.3, cy - 4), color=SHARED)
box(ax, 152.5, cy + 4, 10.5, 3.4, SHARED, "output: box", fs=7, alpha=.35)
box(ax, 152.5, cy - 4, 10.5, 3.4, SHARED, "output: 80 classes", fs=7, alpha=.35)
for s, (y, _) in rows.items():
    arrow(ax, (163.3, cy), (167, y), rad=0)
    ax.text(167.6, y, f"{s} predictions", va="center", fontsize=9, color=SCALE[s], fontweight="bold")

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

ax.text(93, y + 6.2, "proposed", fontsize=10, fontweight="bold", color=INK)
box(ax, 93, y, 8, 5, SCALE["P"], "input", fs=7.5, alpha=.12)
arrow(ax, (101.2, y), (102.8, y)); box(ax, 103, y, 6.5, H80 * .7, SCALE["P"], "adapt.", fs=7.3, alpha=.3)
x = 111.5
for turn in range(4):
    arrow(ax, (x - 1.8, y), (x - .2, y))
    box(ax, x, y, 5, H80 * .7, SHARED, "A", fs=8.5, alpha=.3)
    box(ax, x + 5.6, y, 5, H80 * .7, SHARED, "B", fs=8.5, alpha=.3)
    ax.text(x + 5.3, y + 4.6, f"pass {turn + 1}", ha="center", fontsize=7.6, color=SHARED)
    x += 13
arrow(ax, (x - 1.8, y), (x - .2, y)); box(ax, x, y, 8, 3.4, SHARED, "output", fs=7.5, alpha=.35)
ax.text(93, y - 6.6, "8 blocks deep, but only 2 sets of weights (A and B), reused across 4 passes and 3 scales",
        fontsize=9, color=GREY)

# ---------------------------------------------------------------- NUMBERS
rows_txt = [("", "original", "proposed"),
            ("width", "box 16 · class 80", "80 (box and class together)"),
            ("depth", "2 blocks", "adapter + 8 blocks"),
            ("repetitions", "none", "4 passes"),
            ("blocks with their own weights", "12", "2 shared + 3 adapters"),
            ("head compute", "385 M MACs", "≈ 758 M MACs (≈ 2×)")]
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
