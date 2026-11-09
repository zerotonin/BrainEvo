# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — figures                                              ║
# ║  « SVG with live text, PNG for slides, CSV for reviewers »       ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from brainevo.constants import FIGURE_DPI, RCPARAMS


def apply_style() -> None:
    """Install the project rcParams (idempotent)."""
    matplotlib.rcParams.update(RCPARAMS)


def save_figure(
    fig: Figure,
    stem: str,
    output_dir: Path,
    data: pd.DataFrame | Mapping[str, np.ndarray] | None = None,
) -> dict[str, Path]:
    """Export ``fig`` as SVG + PNG, with an optional CSV data companion.

    Args:
        fig:        Matplotlib figure to save.
        stem:       Filename stem (no extension).
        output_dir: Target directory (created if needed).
        data:       Table behind the figure, written next to it as CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": output_dir / f"{stem}.svg",
        "png": output_dir / f"{stem}.png",
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=FIGURE_DPI, bbox_inches="tight")
    if data is not None:
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(dict(data))
        paths["csv"] = output_dir / f"{stem}.csv"
        frame.to_csv(paths["csv"], index=False)
    plt.close(fig)
    return paths
