#!/usr/bin/env python3
"""Draw assets/looped-head.png: the three studied heads, stacked: native, loop_ab2 and loop_a3.

Block height is proportional to channel width; blue/orange/green mark weights of one scale and
purple marks weights shared by P, M and G. The numbers in the table are measured values from the
v0.2.0 release profiles (results/looped/profiles.csv, seed 11, batch 1, 512 px); all three arms were trained
(15 closed 40-epoch fits). Only Conv MACs are counted; norm, activation, add and decoding are not.
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


fig, ax = plt.subplots(figsize=(17, 33.4), dpi=150)
ax.set_xlim(0, 170); ax.set_ylim(0, 334); ax.axis("off")
ax.text(85, 330.5, "YOLO26n detection head: native vs. two looped designs", ha="center", va="center",
        fontsize=20, fontweight="bold", color=INK)
ax.text(85, 326.4, "block height = number of channels   ·   blue / orange / green = weights of one scale   ·   "
        "purple = weights shared by P, M and G", ha="center", va="center", fontsize=11, color=GREY)
ax.text(85, 322.9, "one internal copy is drawn  ·  all three arms: 40-epoch fits complete  ·  table values are measured at 512 px", ha="center",
        va="center", fontsize=10, color=GREY)


# ---------------------------------------------------------------- NATIVE
def native_panel(T):
    section(ax, T, "NATIVE — three separate heads, two blocks per branch",
            "weights separate per scale  ·  each head runs on its own map  ·  24 conv applications per internal copy")
    for x, t in ((46, "block 1"), (64, "block 2"), (78, "output")):
        ax.text(x, T - 9.8, t, ha="center", fontsize=9, color=GREY)
    for s, dy in (("P", -20), ("M", -37.5), ("G", -55)):
        y, c = T + dy, SCALE[s]
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
    x0 = 110
    ax.text(x0, T - 17, "thin row · box branch", fontsize=10, color=INK, fontweight="bold")
    ax.text(x0, T - 18.8, "conventional 3×3 conv → 16 ch, twice,\nthen 1×1 → 4 box outputs", fontsize=9.5,
            color=INK, va="top", linespacing=1.4)
    ax.text(x0, T - 29, "tall row · class branch", fontsize=10, color=INK, fontweight="bold")
    ax.text(x0, T - 30.8, "separable 3×3 → 80 ch, twice,\nthen 1×1 → 80 class outputs", fontsize=9.5,
            color=INK, va="top", linespacing=1.4)
    ax.text(x0, T - 40.6, "each separable block = depthwise 3×3 + pointwise 1×1,\ntwo primitive convs",
            fontsize=9, color=GREY, va="top", linespacing=1.4)
    ax.text(x0, T - 47.6, "Block 1 adapts each scale (64 / 128 / 256 ch)\nto the tower width. Head weights are not shared\n"
            "across P, M and G.", fontsize=9, color=GREY, va="top", linespacing=1.4)
    seg(ax, [6, 164], [T - 65, T - 65], RULE, 1.2)


# ---------------------------------------------------------------- LOOPED ARMS
def looped_panel(T, title, subtitle, wb, wc, layers, passes, formula):
    """Draw one looped arm. `layers` are the shared layers inside F: ('A',) or ('A', 'B')."""
    section(ax, T, title, subtitle)
    YB, YC = T - 16, T - 40                  # box tower, class tower
    ax.text(32, T - 8.4, "one adapter per scale, per tower: DW 3×3 on input ch\n+ 1×1 → W + learned scale ID (id)", ha="center",
            va="bottom", fontsize=8.4, color=GREY, linespacing=1.3)
    box_ys, cls_ys = [], []
    BOX_BUS, CLS_BUS = 41.6, 43.6
    for s, dy in (("P", -20), ("M", -38), ("G", -56)):
        y, c = T + dy, SCALE[s]
        scale_input(ax, 6, s, y)
        yb, yc = y + 4.6, y - 4.4
        box_ys.append(yb); cls_ys.append(yc)
        arrow(ax, (20.3, y + 1), (26.8, yb)); arrow(ax, (20.3, y - 1), (26.8, yc))
        box(ax, 27, yb, 5, H(wb), c, alpha=.6)
        box(ax, 27, yc, 5, H(wc), c, alpha=.35)
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
        seg(ax, [bus, 46], [yt, yt])
        point_list(ax, 46, yt, 11)
        arrow(ax, (57.2, yt), (68.2, yt), SHARED, lw=1.4)
    ax.text(51.5, YB + 2.8, "flatten + concat", ha="center", fontsize=8.4, color=INK, fontweight="bold")
    ax.text(51.5, YB - 2.2, "5,376 points", ha="center", va="top", fontsize=8.4, color=INK, fontweight="bold")
    ax.text(51.5, YB - 5.2, "4096 P | 1024 M | 256 G", ha="center", va="top", fontsize=6.9, color=GREY)

    ax.add_patch(FancyBboxPatch((63, T - 56.4), 46.6, 51.4, boxstyle="round,pad=0.2,rounding_size=1.5",
                                fc="#F6F2FD", ec=SHARED, lw=1.2, ls="--", zorder=.8))
    ax.text(86.3, T - 7.0, "loop weights shared by P, M and G", ha="center", fontsize=10, color=SHARED,
            fontweight="bold")
    ax.text(86.3, T - 54.2, formula, ha="center", fontsize=9.3, color=SHARED)

    n = len(layers)
    bw = 9 if n == 2 else 15                  # block width
    x1, cx = 67.5, 99
    xs = [x1 + i * (bw + 3.4) for i in range(n)]
    end = xs[-1] + bw

    def tower(yt, ch, width_label, rad):
        h = H(ch)
        for x, layer in zip(xs, layers):
            box(ax, x, yt, bw, h, SHARED, layer if h > 4 else "", fs=12, alpha=.3, bold=True, tc=SHARED)
            if h <= 4:
                ax.text(x + bw / 2, yt + h / 2 + 1.5, layer, ha="center", fontsize=9.5, color=SHARED,
                        fontweight="bold")
        for a, b in zip(xs, xs[1:]):
            arrow(ax, (a + bw + .2, yt), (b - .2, yt), SHARED)
        arrow(ax, (end + .2, yt), (cx - 1.5, yt), SHARED)
        ax.add_patch(Circle((cx, yt), 1.3, fc="white", ec=SHARED, lw=1.5, zorder=3))
        ax.text(cx, yt, "+", ha="center", va="center", fontsize=11, color=SHARED, fontweight="bold", zorder=4)
        low = yt - h / 2 - 2
        seg(ax, [65, 65, cx], [yt, low, low], SHARED, 1.2, 1.6)
        arrow(ax, (cx, low), (cx, yt - 1.4), SHARED)
        arrow(ax, (cx + 1, yt + 1), (x1 - .6, yt + h / 2 + .3), SHARED, rad=rad, lw=2, ms=14)
        ax.text(cx + 2.4, yt + 3.2, f"×{passes}", fontsize=11, color=SHARED, fontweight="bold")
        ax.text(86.3, low - 2.4, width_label, ha="center", fontsize=8.4, color=SHARED)

    kind = " → ".join(layers)
    tower(YB, wb, f"box · {kind} · norm + SiLU + 1×1 conv · {wb} ch", .35)
    tower(YC, wc, f"class · {kind} · norm + SiLU + 1×1 conv · {wc} ch", .45)
    for yt, ch, label, n_out in ((YB, 4, "boxes", 4), (YC, 80, "classes", 80)):
        h = H(ch) + (.4 if ch == 4 else 0)
        seg(ax, [cx + 1.3, 112.8], [yt, yt], SHARED)
        box(ax, 113, yt, 3, h, SHARED, alpha=.55)
        ax.text(114.5, yt + h / 2 + 1.3, f"readout\n1×1 → {n_out}", ha="center", va="bottom", fontsize=8,
                color=SHARED, linespacing=1.2)
        seg(ax, [116.2, 119], [yt, yt])
        point_list(ax, 119, yt, 11)
        arrow(ax, (130.2, yt), (133, yt))
        ax.text(133.8, yt, f"{label}, split back\nto P, M and G", va="center", fontsize=9.6, color=INK,
                fontweight="bold", linespacing=1.35)
    ax.text(106, T - 60.8, "Adapters are specific to each scale and still run 3 times. In the loop, the only operations are 1×1 convs with "
            "per-point channel norm,\nso P, M and G points share one list and every layer is applied once per pass to all 5,376 points; "
            "no 3×3 and no statistics\nacross points or scales. Scale IDs let the loop behave differently per scale; they do not guarantee that conflicts disappear.",
            ha="center", va="top", fontsize=9, color=GREY, linespacing=1.45)
    seg(ax, [6, 164], [T - 70.4, T - 70.4], RULE, 1.2)


native_panel(316)
looped_panel(243, "LOOP_AB2 — two layers A → B per tower, 2 passes",
             "box 24 · class 96 channels  ·  box and class do not share weights  ·  22 conv applications per internal copy",
             24, 96, ("A", "B"), 2, "each pass:  h ← h + 0.1 · F(h),   F = B(A(h))")
looped_panel(170, "LOOP_A3 — one layer A per tower, 3 passes",
             "box 28 · class 108 channels  ·  box and class do not share weights  ·  20 conv applications per internal copy",
             28, 108, ("A",), 3, "each pass:  h ← h + 0.1 · F(h),   F = A(h)")


# ---------------------------------------------------------------- UNROLLED PATHS (stacked)
ax.text(85, 92, "The path of one location through the class tower", ha="center", fontsize=12.5,
        fontweight="bold", color=INK)


def path(y, label, steps, note, out_color, out_text="out"):
    ax.text(6, y + 5.2, label, fontsize=10, fontweight="bold", color=INK)
    box(ax, 6, y, 6, 4.6, SCALE["P"], "in", fs=8, alpha=.1)
    x, end = 15, 12.2
    for text, color, w, gap, top in steps:
        arrow(ax, (end, y), (x - .2, y))
        end = x + w + .2
        tc = SHARED if color == SHARED else INK
        box(ax, x, y, w, H(80) * .62, color, text, fs=8.5, alpha=.3, tc=tc, bold=color == SHARED)
        if top:
            ax.text(x + w / 2, y + 3.9, top, ha="center", fontsize=7.8, color=SHARED)
        x += w + gap
    arrow(ax, (end, y), (x - .2, y))
    box(ax, x, y, 7, 4.6, out_color, out_text, fs=8, alpha=.4, tc=SHARED if out_color == SHARED else INK)
    ax.text(x + 10, y, note, va="center", fontsize=9.3, color=GREY)


path(81, "native", [("DW 3×3 + PW · 80", SCALE["P"], 17, 3.4, ""), ("DW 3×3 + PW · 80", SCALE["P"], 17, 3.4, "")],
     "2 blocks, used once, weights of this scale only", SCALE["P"])
path(67, "loop_ab2", [("adapter · 96", SCALE["P"], 17, 3.4, ""), ("A", SHARED, 7, 3.4, "pass 1"),
                      ("B", SHARED, 7, 3.4, ""), ("A", SHARED, 7, 3.4, "pass 2"), ("B", SHARED, 7, 3.4, "")],
     "this scale's adapter, then the same A and B twice for P, M and G", SHARED, "readout")
path(53, "loop_a3", [("adapter · 108", SCALE["P"], 17, 3.4, ""), ("A", SHARED, 7, 3.4, "pass 1"),
                     ("A", SHARED, 7, 3.4, "pass 2"), ("A", SHARED, 7, 3.4, "pass 3")],
     "this scale's adapter, then the same A three times for P, M and G", SHARED, "readout")
seg(ax, [6, 164], [44.5, 44.5], RULE, 1.2)

# ---------------------------------------------------------------- NUMBERS
rows_txt = [("", "native", "loop_ab2", "loop_a3"),
            ("channel width", "box 16 · class 80", "box 24 · class 96", "box 28 · class 108"),
            ("what is reused", "nothing across scales", "A, B shared by P, M, G", "A shared by P, M, G"),
            ("applications per point", "2 blocks", "A → B twice = 4", "A three times = 3"),
            ("conv applications / internal copy", "24", "22", "20"),
            ("spatial view of the input map", "5×5", "3×3 (adapter only)", "3×3 (adapter only)"),
            ("head params, both internal branches", "309,656", "181,920", "183,504"),
            ("head Conv MACs, both internal branches", "384.6 M (measured)", "631.4 M (measured)",
             "636.9 M (measured)"),
            ("head Conv MACs, one2one branch only", "192.3 M (measured)", "315.7 M (measured)",
             "318.5 M (measured)")]
ax.add_patch(FancyBboxPatch((8, 12.0), 154, 27.2, boxstyle="round,pad=0.2,rounding_size=1",
                            fc="#F9FAFB", ec=RULE, lw=1, zorder=.5))
for i, row in enumerate(rows_txt):
    yy = 37.6 - i * 2.9
    ax.text(52, yy, row[0], ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
    for xc, txt, col in zip((70, 103, 137), row[1:], (SCALE["P"], SHARED, SHARED)):
        ax.text(xc, yy, txt, ha="center", va="center", fontsize=10, color=col if i == 0 else INK,
                fontweight="bold" if i == 0 else "normal")
ax.text(85, 9.2, "Measured per 512 px image from the release profiles (results/looped/profiles.csv): unique head parameters and Conv MACs only; "
        "norm, activation, add and decoding are not counted.\n"
        "The end-to-end YOLO26n head has two internal branches (one2many and one2one); \"one2one branch only\" is the branch-pruned "
        "inference head, not a fused-BatchNorm model.\n"
        "Conv MACs are not latency. Spatial view = 3×3 or 5×5 on the head's input feature map; conv applications count layers per internal copy, "
        "not GPU kernels.\n"
        "Head parameters fall about 41% while head compute rises about 64–66% versus native; accuracy, latency and memory are in the README.",
        ha="center", va="top", fontsize=8.4, color=GREY, linespacing=1.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
