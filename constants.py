"""
Style constants for the mutation_viewer app.
Extracted from the manuscript Figure_Scripts_Final/constants.py — only items
used by app.py are included here so the app is fully self-contained.
"""

import matplotlib as mpl
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
TITLE_FS          = 10
AXIS_LABEL_FS     = 9
TICK_FS           = 8
LEGEND_FS         = 8
COLORBAR_LABEL_FS = 8
COLORBAR_TICK_FS  = 7
ANNOTATION_FS     = 8

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
ORANGE = "#DD8D6E"
BEIGE  = "#F0E8D1"
GREEN  = "#558771"
TEAL   = "#82A899"
PURPLE = "#885784"
GOLD   = "#C2AB42"


def make_linear_cmap(colors, name="custom_cmap", N=256):
    """Create a LinearSegmentedColormap from a list of hex / RGB colors."""
    return mcolors.LinearSegmentedColormap.from_list(name, colors, N=N)


# Pre-built colormaps
CMAP_ORANGE_GREEN  = make_linear_cmap([ORANGE, BEIGE, GREEN],  "orange_green")
CMAP_ORANGE_PURPLE = make_linear_cmap([ORANGE, BEIGE, PURPLE], "orange_purple")

# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------
mpl.rcParams["figure.facecolor"] = "none"
mpl.rcParams["axes.facecolor"]   = "none"
mpl.rcParams["font.family"]      = "sans-serif"
mpl.rcParams["pdf.fonttype"]     = 42
mpl.rcParams["ps.fonttype"]      = 42
