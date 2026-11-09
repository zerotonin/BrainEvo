# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — ode.viz                                              ║
# ║  « landscape, valley, bifurcation diagram, hysteresis loop »     ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from brainevo.constants import (
    COLOUR_BACK,
    COLOUR_FRONT,
    COLOUR_FRONT_ONLY,
    COLOUR_MIXED,
    COLOUR_NICHE,
    COLOUR_POPULATION,
    COLOUR_SWITCH,
    FIGURE_DOUBLE,
    FIGURE_SINGLE,
    FloatArray,
)
from brainevo.ode.meanfield import MeanField


def plot_landscape(mf: MeanField, n: int = 200) -> Figure:
    """Q(E_f, E_b) as a filled contour with the singular strategies marked."""
    a = mf.arch
    Ef = np.linspace(0.0, 3.0 * a.E_f_saturation, n)
    Eb = np.linspace(0.0, a.e_switch + 2.5 * a.E_b_climb, n)
    EF, EB = np.meshgrid(Ef, Eb)
    Q = np.asarray(mf.Q(EF, EB))
    fig, ax = plt.subplots(figsize=FIGURE_SINGLE)
    cs = ax.contourf(EF, EB, Q, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=r"$Q = F / C$ (fidelity per unit cost)")
    ax.axhline(a.e_switch, color=COLOUR_SWITCH, ls="--", lw=0.8)
    for s in mf.singular_strategies():
        colour = COLOUR_MIXED if s.has_backend else COLOUR_FRONT_ONLY
        ax.plot(s.Ef, s.Eb, "o", color=colour, mec="white", ms=7, label=s.branch)
    ax.set_xlabel(r"front-end investment $E_f$ (energy units yr$^{-1}$)")
    ax.set_ylabel(r"back-end investment $E_b$ (energy units yr$^{-1}$)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def plot_valley(mfs: dict[str, MeanField]) -> Figure:
    """Q along E_b at the front-only E_f, for several labelled mean fields."""
    fig, ax = plt.subplots(figsize=FIGURE_SINGLE)
    cmap = plt.get_cmap("viridis")
    for i, (label, mf) in enumerate(mfs.items()):
        Eb, Q = mf.valley()
        ax.plot(Eb, Q / Q[0], color=cmap(i / max(len(mfs) - 1, 1)), label=label)
    ax.axhline(1.0, color=COLOUR_POPULATION, lw=0.6, ls=":")
    ax.set_xlabel(r"back-end investment $E_b$ (energy units yr$^{-1}$)")
    ax.set_ylabel(r"$Q(E_f^*, E_b)\,/\,Q(E_f^*, 0)$")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return fig


def plot_bifurcation(
    control: FloatArray,
    branches: dict[str, list[tuple[float, float]]],
    control_label: str,
) -> Figure:
    """E_b* of every singular strategy against a control parameter.

    ``branches`` maps a branch name to a list of ``(control, E_b)`` points.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SINGLE)
    colours = {"front-only": COLOUR_FRONT_ONLY, "mixed": COLOUR_MIXED, "global": COLOUR_POPULATION}
    for name, pts in branches.items():
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        style = dict(lw=0, marker="o", ms=3) if name != "global" else dict(lw=1.2, ls="--")
        ax.plot(xs, ys, color=colours.get(name, COLOUR_BACK), label=name, **style)
    ax.set_xlabel(control_label)
    ax.set_ylabel(r"singular back-end investment $E_b^*$")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_hysteresis(loop: dict[str, FloatArray], control_label: str) -> Figure:
    fig, ax = plt.subplots(figsize=FIGURE_SINGLE)
    ax.plot(loop["control"], loop["Eb_down"], color=COLOUR_FRONT_ONLY, marker="<", ms=3,
            label="ramp down (from front-only)")
    ax.plot(loop["control"], loop["Eb_up"], color=COLOUR_MIXED, marker=">", ms=3, label="ramp up")
    ax.plot(loop["control"], loop["Eb_global"], color=COLOUR_POPULATION, ls="--", lw=0.8, label="global optimum")
    ax.set_xlabel(control_label)
    ax.set_ylabel(r"evolved back-end investment $E_b^*$")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_fold_map(
    P: FloatArray,
    y: FloatArray,
    n_attractors: np.ndarray,
    deficit: np.ndarray,
    ylabel: str,
    log_y: bool = False,
) -> Figure:
    """Valley deficit over the (P, switch-parameter) plane, with the fold
    (where the front-only attractor vanishes) drawn as a contour."""
    fig, ax = plt.subplots(figsize=FIGURE_DOUBLE)
    pm = ax.pcolormesh(P, y, deficit, cmap="viridis", shading="nearest")
    fig.colorbar(pm, ax=ax, label="fitness-valley deficit (energy units yr$^{-1}$)")
    if n_attractors.min() < n_attractors.max():
        ax.contour(P, y, n_attractors, levels=[1.5], colors=[COLOUR_SWITCH], linewidths=1.2)
    ax.set_xlabel(r"back-end overhead $P$ (multiples of $E_{f,\mathrm{sat}}$)")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_pip(sigmas: FloatArray, S: np.ndarray, title: str = "") -> Figure:
    """Pairwise invasibility plot in niche breadth: shaded where the mutant invades."""
    fig, ax = plt.subplots(figsize=FIGURE_SINGLE)
    ax.contourf(sigmas, sigmas, (S > 0).astype(float), levels=[-0.5, 0.5, 1.5],
                colors=["white", COLOUR_MIXED], alpha=0.8)
    ax.contour(sigmas, sigmas, S, levels=[0.0], colors=[COLOUR_POPULATION], linewidths=0.8)
    ax.plot(sigmas, sigmas, color=COLOUR_POPULATION, lw=0.5, ls=":")
    ax.set_xlabel(r"resident niche breadth $\sigma$ (resource types)")
    ax.set_ylabel(r"mutant niche breadth $\sigma'$ (resource types)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=7)
    fig.tight_layout()
    return fig


def plot_niche_trajectory(t: FloatArray, Ef: FloatArray, Eb: FloatArray, sigma: FloatArray, N: FloatArray) -> Figure:
    """Canonical-equation trajectory of (E_f, E_b, sigma) and resident N*."""
    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_SINGLE[0], FIGURE_SINGLE[1] * 2.2), sharex=True)
    axes[0].plot(t, Ef, color=COLOUR_FRONT, label=r"$E_f$")
    axes[0].plot(t, Eb, color=COLOUR_BACK, label=r"$E_b$")
    axes[0].set_ylabel("investment\n(energy units yr$^{-1}$)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(t, sigma, color=COLOUR_NICHE)
    axes[1].set_ylabel(r"niche breadth $\sigma$")
    axes[2].plot(t, N, color=COLOUR_POPULATION)
    axes[2].set_ylabel(r"resident $N^*$")
    axes[2].set_xlabel("evolutionary time (canonical units)")
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig
