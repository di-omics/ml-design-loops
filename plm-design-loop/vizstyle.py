"""House figure style: Manrope, pastel blue/green/purple, thin black lines, white.

Call vizstyle.apply() once before plotting. Colors are deliberately soft ("baby")
with a fine black edge on every mark for a clean, high-clarity look.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# Pastel palette.
BLUE = "#7EB6E6"
GREEN = "#8ED0A9"
PURPLE = "#B49BE0"
PINK = "#E9A6C0"
AMBER = "#EAC27E"
NEUTRAL = "#B7BdC7"
INK = "#20242A"
GRID = "#E7E9ED"
PALETTE = [BLUE, GREEN, PURPLE, PINK, AMBER]

# Pastel colormaps: diverging (pink <-> white <-> blue) and sequential (white->green).
DIVERGING = LinearSegmentedColormap.from_list("baby_div", [PINK, "#ffffff", BLUE])
SEQ_GREEN = LinearSegmentedColormap.from_list("baby_green", ["#ffffff", GREEN])


def _register_font() -> bool:
    fonts = Path(__file__).parent / "assets" / "fonts"
    ok = False
    if fonts.exists():
        for f in fonts.glob("*.ttf"):
            try:
                fm.fontManager.addfont(str(f))
                ok = True
            except Exception:
                pass
    return ok


def apply() -> None:
    family = "Manrope" if _register_font() else "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": family,
        "font.size": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK,
        "axes.labelsize": 10.5,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "lines.markeredgecolor": INK,
        "lines.markeredgewidth": 0.6,
        # fine black edge on every bar/patch
        "patch.force_edgecolor": True,
        "patch.edgecolor": INK,
        "patch.linewidth": 0.8,
        "hatch.linewidth": 0.6,
    })
