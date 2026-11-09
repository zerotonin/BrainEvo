# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm.viz                                              ║
# ║  « trajectories of investment, niche and population »            ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from brainevo.abm.cost_laws import Architecture
from brainevo.abm.energetics import ATPArchitecture
from brainevo.constants import (
    COLOUR_BACK,
    COLOUR_FRONT,
    COLOUR_NICHE,
    COLOUR_POPULATION,
    COLOUR_SWITCH,
    FIGURE_DOUBLE,
)


def plot_trajectories(hist: pd.DataFrame, arch: Architecture | ATPArchitecture) -> Figure:
    """Four stacked panels: E_f/E_b, F_f/F_b, niche breadth, population size."""
    fig, axes = plt.subplots(4, 1, figsize=(FIGURE_DOUBLE[0], FIGURE_DOUBLE[1] * 1.4), sharex=True)
    t = hist["time"] if "time" in hist else hist.index

    ax = axes[0]
    ax.plot(t, hist["Ef_mean"], color=COLOUR_FRONT, label=r"$E_f$")
    ax.plot(t, hist["Eb_mean"], color=COLOUR_BACK, label=r"$E_b$")
    ax.axhline(arch.e_switch, color=COLOUR_SWITCH, ls="--", lw=0.8, label=r"$e_\mathrm{switch}$")
    ax.set_ylabel("neural investment\n(energy units yr$^{-1}$)")
    ax.legend(frameon=False, ncol=3)

    ax = axes[1]
    ax.plot(t, hist["Ff_mean"], color=COLOUR_FRONT, label=r"$F_f$")
    ax.plot(t, hist["Fb_mean"], color=COLOUR_BACK, label=r"$F_b$")
    ax.set_ylabel("fidelity")
    ax.legend(frameon=False, ncol=2)

    ax = axes[2]
    ax.plot(t, hist["sigma"], color=COLOUR_NICHE)
    ax.set_ylabel(r"niche breadth $\sigma_\mathrm{org}$" + "\n(resource types)")

    ax = axes[3]
    ax.plot(t, hist["population_size"], color=COLOUR_POPULATION)
    ax.set_ylabel(r"population size $N$")
    ax.set_xlabel("time (years)")
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig
