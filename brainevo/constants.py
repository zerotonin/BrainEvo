# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — constants                                            ║
# ║  « one source of truth for colours, figure rules, defaults »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Wong (2011) palette with semantic mappings for the two ends of  ║
# ║  the sensory pathway, figure export rules, numerical epsilons,   ║
# ║  and type aliases shared by the ABM and the ODE reduction.       ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from typing import TypeAlias

import numpy as np

# ┌────────────────────────────────────────────────────────────┐
# │ Type aliases  « what the arrays mean »                     │
# └────────────────────────────────────────────────────────────┘
FloatArray: TypeAlias = np.ndarray
"""1-D float array indexed by resource type `z`."""

History: TypeAlias = dict[str, np.ndarray]
"""Population-level time series, one array per logged key."""

# ┌────────────────────────────────────────────────────────────┐
# │ Numerics  « guards used in more than one module »          │
# └────────────────────────────────────────────────────────────┘
EPS: float = 1e-12
"""Smallest meaningful stock / weight; avoids division by zero."""

NAN: float = float("nan")

# ┌────────────────────────────────────────────────────────────┐
# │ Wong (2011) palette  « colourblind-safe base colours »     │
# └────────────────────────────────────────────────────────────┘
WONG: dict[str, str] = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermilion":      "#D55E00",
    "reddish_purple": "#CC79A7",
}

# semantic mapping: the two ends of the sensory pathway and the niche
COLOUR_FRONT: str = WONG["blue"]
"""Front-end (peripheral) investment `E_f`."""
COLOUR_BACK: str = WONG["orange"]
"""Back-end (central) investment `E_b`."""
COLOUR_NICHE: str = WONG["bluish_green"]
"""Niche breadth `sigma`."""
COLOUR_POPULATION: str = WONG["black"]
COLOUR_SWITCH: str = WONG["vermilion"]
"""The back-end switch-on energy `e_switch`."""
COLOUR_FRONT_ONLY: str = WONG["sky_blue"]
"""Front-only (no functional back-end) branch in bifurcation plots."""
COLOUR_MIXED: str = WONG["reddish_purple"]
"""Mixed front+back branch in bifurcation plots."""

# ┌────────────────────────────────────────────────────────────┐
# │ Figure rules  « SVG with live text, PNG, CSV companion »   │
# └────────────────────────────────────────────────────────────┘
FIGURE_DPI: int = 300
FIGURE_SINGLE: tuple[float, float] = (3.5, 2.8)
"""Single-column panel, inches."""
FIGURE_DOUBLE: tuple[float, float] = (7.2, 4.5)
"""Double-column panel, inches."""
FONT_SIZE: int = 8
RCPARAMS: dict[str, object] = {
    "svg.fonttype": "none",
    "font.size": FONT_SIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
    "savefig.dpi": FIGURE_DPI,
}
