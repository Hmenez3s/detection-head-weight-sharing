#!/usr/bin/env python3
"""Render assets/architecture.{svg,png} from one Matplotlib drawing.

Static concept figure only: no study data and no model imports. Output is
deterministic (fixed hash salt, no embedded date/host/path, glyphs as paths).

    MPLBACKEND=Agg .venv/bin/python scripts/draw_architecture.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

W, H, DPI = 2000, 1000, 100
YOFF = 30  # extra top margin: data y-range is [YOFF, H + YOFF]
OUT = Path(__file__).resolve().parent.parent / "assets"

BG, PANEL = "#0B1426", "#101B33"
INK, MUTED = "#EAF2FA", "#A9BBCF"
CYAN, SLATE = "#52D6EE", "#93A7BE"
AMBER, GREEN = "#E0AC4A", "#62CC93"

TITLE = "Architecture of the head-sharing comparison"
DESC = (
    "A frozen YOLO26n backbone and neck feed a common FP32 to FP16 to FP32 feature boundary. "
    "Its features fan out to three controlled rows with identical native adapter and predictor "
    "architectures. Only the second refiner F differs: native applies one F once; "
    "independent_k2 applies two distinct F objects in sequence, F1 then F2; shared_k2 applies "
    "the same F object twice. Each row stands for one of twelve sites: three scales, box or "
    "class, one-to-many or one-to-one. Nothing is tied across scales or branches."
)

_fits: list[tuple[object, tuple[float, float, float, float], str]] = []
_texts: list[object] = []


def px(size: float) -> float:
    """Pixel height to points at the figure DPI."""
    return size * 72 / DPI


def text(ax, x, y, s, size, color=INK, weight="normal", ha="left", va="baseline", gid=None):
    t = ax.text(x, y, s, fontsize=px(size), color=color, fontweight=weight, ha=ha, va=va,
                family="DejaVu Sans", gid=gid)
    _texts.append(t)
    return t


def box(ax, x0, y0, x1, y1, edge, fill, lw=3, radius=14, gid=None, dashed=False):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=lw, edgecolor=edge, facecolor=fill, gid=gid,
                       linestyle=(0, (6, 5)) if dashed else "solid")
    ax.add_patch(p)
    return p


def labeled_box(ax, x0, y0, x1, y1, edge, fill, head, sub, gid, head_px=30, sub_px=22, sub_color=MUTED):
    box(ax, x0, y0, x1, y1, edge, fill, gid=gid)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if sub:
        t1 = text(ax, cx, cy + 6, head, head_px, INK, "bold", "center", "bottom")
        t2 = text(ax, cx, cy - 6, sub, sub_px, sub_color, "normal", "center", "top")
        _fits.extend([(t1, (x0, y0, x1, y1), gid), (t2, (x0, y0, x1, y1), gid)])
    else:
        t1 = text(ax, cx, cy, head, head_px, INK, "bold", "center", "center")
        _fits.append((t1, (x0, y0, x1, y1), gid))


def arrow(ax, p0, p1, color, lw=3.2, gid=None, head=13):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={head / 2.2},head_width={head / 3.4}",
                                 mutation_scale=2.2, color=color, linewidth=lw, shrinkA=0, shrinkB=0, gid=gid))


def loop(ax, x_left, x_right, y_top, color, gid):
    """Return arc from F's output back to its own input: the same object applied again."""
    xr, xl, rise = x_right - 34, x_left + 34, 34
    path = MPath([(xr, y_top), (xr, y_top + rise * 1.35), (xl, y_top + rise * 1.35), (xl, y_top + 3)],
                 [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>,head_length=5,head_width=3.2",
                                 mutation_scale=2.2, color=color, linewidth=3.4, gid=gid))


def build():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(YOFF, H + YOFF)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, YOFF), W, H, facecolor=BG, edgecolor="none", gid="background"))

    # Headline frames the hypothesis as a question only.
    text(ax, 60, 962, "HEAD-GENERALIZATION STUDY  ·  YOLO26n", 22, CYAN, "bold", gid="eyebrow")
    text(ax, 60, 912, "Does tying a head's second refiner change what it learns?", 48, INK, "bold", gid="headline")
    text(ax, 60, 866, "Same frozen features and native adapter/predictor design; only the second refiner differs.",
         26, MUTED, gid="subhead")

    # Geometry
    band_h, gap, top0 = 195, 12, 836
    tops = [top0 - i * (band_h + gap) for i in range(3)]
    cy = [t - band_h + 72 for t in tops]  # block centre line per row
    mid = cy[1]
    band_x0, band_x1 = 715, 1950

    # Common trunk + Q
    box(ax, 50, mid - 110, 340, mid + 110, CYAN, "#0F2B45", gid="trunk")
    tcx = 195
    for dy, s, size, col, wt in ((50, "Frozen backbone", 28, INK, "bold"), (14, "+ neck", 32, INK, "bold"),
                                 (-30, "YOLO26n · eval mode", 22, MUTED, "normal"),
                                 (-64, "BN stats fixed", 22, MUTED, "normal")):
        t = text(ax, tcx, mid + dy, s, size, col, wt, "center", "center")
        _fits.append((t, (50, mid - 110, 340, mid + 110), "trunk"))
    arrow(ax, (340, mid), (405, mid), CYAN, gid="arrow-trunk-to-q")
    labeled_box(ax, 405, mid - 80, 655, mid + 80, CYAN, "#0F2B45", "common Q", "FP32→FP16→FP32", "boundary-q")
    text(ax, 195, mid - 150, "same for all arms", 22, CYAN, ha="center", gid="trunk-note")

    # Fan-out
    fx = 686
    ax.plot([655, fx], [mid, mid], color=CYAN, linewidth=3.2, solid_capstyle="butt", gid="fan-stem")
    ax.plot([fx, fx], [cy[2], cy[0]], color=CYAN, linewidth=3.2, solid_capstyle="butt", gid="fan-spine")
    for i, c in enumerate(cy):
        arrow(ax, (fx, c), (745, c), CYAN, gid=f"fan-arrow-{i + 1}")

    rows = [
        ("native", "native · F once", SLATE, "#18243B", "1 application", "1 unique F"),
        ("independent_k2", "independent_k2 · F₁ then F₂", AMBER, "#33290F", "2 applications", "2 unique F"),
        ("shared_k2", "shared_k2 · same F twice", GREEN, "#0F3022", "2 applications", "1 unique F"),
    ]
    for i, (key, label, col, fill, n_app, n_f) in enumerate(rows):
        t, c = tops[i], cy[i]
        box(ax, band_x0, t - band_h, band_x1, t, col, PANEL, lw=1.6, radius=18, gid=f"row-{key}")
        text(ax, 745, t - 38, label, 28, col, "bold", gid=f"row-label-{key}")
        by0, by1 = c - 46, c + 46
        labeled_box(ax, 745, by0, 975, by1, SLATE, "#18243B", "Scale adapter", "fresh init", f"adapter-{key}", 26)
        labeled_box(ax, 1460, by0, 1690, by1, SLATE, "#18243B", "Predictor", "fresh init", f"predictor-{key}", 26)
        if key == "native":
            arrow(ax, (975, c), (1117, c), col, gid=f"arrow-a-f-{key}")
            labeled_box(ax, 1117, by0, 1317, by1, col, fill, "F", "native", f"f-{key}", 32)
            arrow(ax, (1317, c), (1460, c), col, gid=f"arrow-f-p-{key}")
        elif key == "independent_k2":
            arrow(ax, (975, c), (1020, c), col, gid=f"arrow-a-f-{key}")
            labeled_box(ax, 1020, by0, 1185, by1, col, fill, "F₁", "own BN", f"f1-{key}", 32)
            arrow(ax, (1185, c), (1240, c), col, gid=f"arrow-f1-f2-{key}")
            labeled_box(ax, 1240, by0, 1405, by1, col, fill, "F₂", "own BN", f"f2-{key}", 32)
            arrow(ax, (1405, c), (1460, c), col, gid=f"arrow-f-p-{key}")
        else:
            arrow(ax, (975, c), (1117, c), col, gid=f"arrow-a-f-{key}")
            labeled_box(ax, 1117, by0, 1317, by1, col, fill, "F", "shared BN", f"f-{key}", 32)
            loop(ax, 1117, 1317, by1, col, f"loop-{key}")
            text(ax, 1217, by1 + 46, "×2", 26, col, "bold", "center", gid=f"loop-label-{key}")
            arrow(ax, (1317, c), (1460, c), col, gid=f"arrow-f-p-{key}")
        t1 = text(ax, 1716, c + 8, n_app, 26, col, "bold", va="bottom", gid=f"count-app-{key}")
        t2 = text(ax, 1716, c - 8, n_f, 26, INK, va="top", gid=f"count-f-{key}")
        _fits.extend([(t1, (1696, t - band_h, band_x1, t), f"count-{key}"),
                      (t2, (1696, t - band_h, band_x1, t), f"count-{key}")])

    # Caption
    ax.plot([60, 1940], [200, 200], color="#2A3A57", linewidth=1.6, gid="caption-rule")
    cap = [
        "Each row shows one of 12 sites: 3 scales × box/class × one-to-many/one-to-one. Only the second unit changes.",
        "Nothing is tied across scales or branches; native boxes, classes and both branches are retained.",
        "Shared F reuses Conv and BN affine + running state; running state updates twice per training forward.",
    ]
    for j, s in enumerate(cap):
        text(ax, 60, 160 - j * 40, s, 26, INK if j == 0 else MUTED, gid=f"caption-{j + 1}")
    return fig


def check_layout(fig) -> None:
    """Fail loudly on clipped text, text outside its box, or overlapping text."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.axes[0].transData.inverted()

    def ext(t):
        bb = t.get_window_extent(r)
        (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        return x0, y0, x1, y1

    for t, (bx0, by0, bx1, by1), gid in _fits:
        x0, y0, x1, y1 = ext(t)
        if x0 < bx0 + 10 or x1 > bx1 - 10 or y0 < by0 + 4 or y1 > by1 - 4:
            raise SystemExit(f"text {t.get_text()!r} does not fit in {gid}: {(x0, y0, x1, y1)}")
    boxes = [(t, ext(t)) for t in _texts]
    for t, (x0, y0, x1, y1) in boxes:
        if x0 < 30 or x1 > W - 30 or y0 < YOFF + 20 or y1 > H + YOFF - 10:
            raise SystemExit(f"text {t.get_text()!r} outside canvas margin")
    for i, (a, ea) in enumerate(boxes):
        for b, eb in boxes[i + 1:]:
            if ea[0] < eb[2] and eb[0] < ea[2] and ea[1] < eb[3] and eb[1] < ea[3]:
                raise SystemExit(f"text overlap: {a.get_text()!r} / {b.get_text()!r}")


def add_svg_accessibility(path: Path) -> None:
    s = path.read_text(encoding="utf-8")
    s = re.sub(r'width="[\d.]+pt" height="[\d.]+pt"', f'width="{W}" height="{H}"', s, count=1)
    root = re.search(r"<svg\b[^>]*>", s)
    tag = root.group(0)
    new = tag[:-1] + ' role="img" aria-labelledby="fig-title fig-desc">'
    s = s.replace(tag, new + f'\n <title id="fig-title">{TITLE}</title>\n <desc id="fig-desc">{DESC}</desc>', 1)
    path.write_text(s, encoding="utf-8")


def main() -> int:
    matplotlib.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "head-generalization-architecture-v1"})
    OUT.mkdir(exist_ok=True)
    fig = build()
    check_layout(fig)
    svg_meta = {"Creator": "scripts/draw_architecture.py", "Date": None, "Title": TITLE, "Description": DESC}
    fig.savefig(OUT / "architecture.svg", format="svg", facecolor=BG, metadata=svg_meta)
    fig.savefig(OUT / "architecture.png", format="png", dpi=DPI, facecolor=BG, metadata={"Software": None})
    add_svg_accessibility(OUT / "architecture.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
