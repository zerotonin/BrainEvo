# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — analysis.viz                                         ║
# ║  « the sweep figures »                                           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Grid maps (naive establishment vs seeded retention), hazard agai║
# ║  the mean-field valley deficit, neuron count against atmospheric ║
# ║  oxygen for both lineages, one-at-a-time sensitivity strips, and ║
# ║  niche-breadth distribution through time. Wong palette; every fig║
# ║  goes out as SVG + PNG + CSV through brainevo.figures.save_figure║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from brainevo.constants import (
    COLOUR_BACK,
    COLOUR_FRONT,
    COLOUR_FRONT_ONLY,
    COLOUR_MIXED,
    COLOUR_NICHE,
    COLOUR_SWITCH,
    FIGURE_DOUBLE,
    FIGURE_SINGLE,
    WONG,
)

HEATMAP_CMAP: str = "cividis"
COLOUR_TRACHEAL: str = WONG["vermilion"]
COLOUR_CONVECTIVE: str = WONG["blue"]
MODERN_PO2_KPA: float = 21.2
CARBONIFEROUS_PO2_KPA: float = 30.0
JITTER: float = 0.08


# ┌────────────────────────────────────────────────────────────┐
# │ Grid maps  « establishment vs retention »                  │
# └────────────────────────────────────────────────────────────┘

def plot_grid_maps(cells: pd.DataFrame, x: str, y: str, title: str = "") -> Figure:
    """Three panels over the (x, y) grid: naive establishment probability,
    seeded retention probability, and mean establishment year (log)."""
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_DOUBLE[0] * 1.4, FIGURE_DOUBLE[1] * 0.75), constrained_layout=True)
    panels = (("p_establish", "P(established | naive)", 0.0, 1.0, False),
              ("p_retained", "P(retained | seeded)", 0.0, 1.0, False),
              ("t_establish_mean", "mean establishment year", None, None, True))
    for ax, (col, label, vmin, vmax, log) in zip(axes, panels, strict=True):
        pivot = cells.pivot(index=y, columns=x, values=col).sort_index(ascending=True)
        values = np.log10(pivot.to_numpy()) if log else pivot.to_numpy()
        im = ax.imshow(values, origin="lower", aspect="auto", cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(pivot.shape[1]), [f"{v:g}" for v in pivot.columns])
        ax.set_yticks(range(pivot.shape[0]), [f"{v:g}" for v in pivot.index])
        ax.set_xlabel(_pretty(x))
        ax.set_ylabel(_pretty(y))
        ax.set_title(label)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        if log:
            cb.set_label("log$_{10}$ year")
    if title:
        fig.suptitle(title)
    return fig


def plot_hazard_vs_deficit(cells: pd.DataFrame, fit_slope: float, fit_intercept: float) -> Figure:
    """Establishment hazard (events per year, log) against the mean-field valley deficit."""
    fig, ax = plt.subplots(figsize=FIGURE_SINGLE, constrained_layout=True)
    pos = cells[cells["hazard"] > 0]
    zero = cells[cells["hazard"] == 0]
    ax.errorbar(pos["valley_deficit"], pos["hazard"],
                yerr=[pos["hazard"] - pos["hazard_lo"], pos["hazard_hi"] - pos["hazard"]],
                fmt="o", color=COLOUR_MIXED, ms=3, lw=0.8, label="established in ≥1 replicate")
    if len(zero):
        ax.scatter(zero["valley_deficit"], zero["hazard_hi"], marker="v", color=COLOUR_FRONT_ONLY, s=14,
                   label="never established (upper bound)")
    if np.isfinite(fit_slope):
        xx = np.linspace(cells["valley_deficit"].min(), cells["valley_deficit"].max(), 50)
        ax.plot(xx, np.exp(fit_intercept + fit_slope * xx), color=COLOUR_SWITCH, lw=1,
                label=f"log h = {fit_intercept:.1f} {fit_slope:+.1f}·deficit")
    ax.set_yscale("log")
    ax.set_xlabel("mean-field valley deficit (energy yr$^{-1}$)")
    ax.set_ylabel("establishment hazard (yr$^{-1}$)")
    ax.legend(fontsize=6, frameon=False)
    return fig


# ┌────────────────────────────────────────────────────────────┐
# │ Oxygen  « the capped insect and the uncapped vertebrate »  │
# └────────────────────────────────────────────────────────────┘

def plot_oxygen(table: pd.DataFrame) -> Figure:
    """Effective neuron count against atmospheric pO2 for both lineages.

    ``table`` needs columns lineage, pO2, n_eff (per run), hard_cap and
    n_meanfield (per lineage × pO2).
    """
    fig, ax = plt.subplots(figsize=(FIGURE_SINGLE[0], FIGURE_SINGLE[1] + 0.9), constrained_layout=True)
    colours = {"tracheal": COLOUR_TRACHEAL, "convective": COLOUR_CONVECTIVE}
    for lineage, grp in table.groupby("lineage"):
        c = colours.get(str(lineage), COLOUR_BACK)
        per = grp.groupby("pO2").agg(cap=("hard_cap", "first"), mf=("n_meanfield", "first"), med=("n_eff", "median"))
        ax.plot(per.index, per["cap"], ls="--", lw=0.8, color=c, label=f"{lineage}: oxygen cap")
        ax.plot(per.index, per["mf"], ls="-", lw=1.0, color=c, alpha=0.6, label=f"{lineage}: mean-field optimum")
        jitter = np.random.default_rng(0).normal(0, JITTER, len(grp))
        ax.scatter(grp["pO2"] + jitter, grp["n_eff"], s=9, color=c, alpha=0.6, edgecolor="none")
        ax.plot(per.index, per["med"], "o", color=c, ms=4, mec="black", mew=0.4, label=f"{lineage}: ABM median")
    for p, lab in ((MODERN_PO2_KPA, "today"), (CARBONIFEROUS_PO2_KPA, "Carboniferous")):
        ax.axvline(p, color="0.6", lw=0.6, ls=":")
        ax.annotate(lab, (p, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=6,
                    color="0.4")
    ax.set_xlabel("atmospheric $p$O$_2$ (kPa)")
    ax.set_ylabel("back-end neurons (fraction of ceiling count)")
    ax.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    return fig


# ┌────────────────────────────────────────────────────────────┐
# │ Sensitivity  « one constant at a time »                    │
# └────────────────────────────────────────────────────────────┘

def plot_sensitivity(runs: pd.DataFrame, parameters: list[str], response: str, log_x: set[str] | None = None,
                     reference: dict[str, float] | None = None) -> Figure:
    """Strip + median of ``response`` against each swept parameter, one panel each."""
    log_x = log_x or set()
    reference = reference or {}
    n = len(parameters)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols + 0.6, 2.0 * rows), squeeze=False, constrained_layout=True)
    for ax, param in zip(axes.flat, parameters, strict=False):
        sub = runs[[param, response]].dropna()
        levels = sorted(sub[param].unique(), key=lambda v: (isinstance(v, str), v))
        positions = np.arange(len(levels))
        for i, level in enumerate(levels):
            vals = sub.loc[sub[param] == level, response].to_numpy(dtype=float)
            ax.scatter(i + np.random.default_rng(i).normal(0, JITTER, len(vals)), vals, s=8, color=COLOUR_BACK,
                       alpha=0.6, edgecolor="none")
            ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color="black", lw=1.2)
        ax.set_xticks(positions, [_level(v) for v in levels], fontsize=6)
        if param in reference:
            ref = [i for i, v in enumerate(levels) if np.isclose(float(v), reference[param])] if levels else []
            for i in ref:
                ax.axvspan(i - 0.4, i + 0.4, color=COLOUR_NICHE, alpha=0.12, lw=0)
        ax.set_xlabel(_pretty(param))
        ax.set_ylabel(_pretty(response))
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return fig


# ┌────────────────────────────────────────────────────────────┐
# │ Niche breadth  « bimodality through time »                 │
# └────────────────────────────────────────────────────────────┘

def plot_sigma_through_time(snapshots: pd.DataFrame, fits: pd.DataFrame | None = None) -> Figure:
    """Log-sigma histograms per snapshot (one ridge each) for a single run,
    annotated with dBIC and Ashman's D when ``fits`` is given."""
    times = sorted(snapshots["time"].unique())
    fig, ax = plt.subplots(figsize=(FIGURE_SINGLE[0], 0.45 * len(times) + 0.8), constrained_layout=True)
    log_s = np.log10(snapshots["sigma"].clip(lower=1e-3))
    bins = np.linspace(log_s.min(), log_s.max(), 40)
    for k, t in enumerate(times):
        vals = log_s[snapshots["time"] == t]
        hist, edges = np.histogram(vals, bins=bins, density=True)
        hist = hist / max(hist.max(), 1e-12) * 0.9
        ax.fill_between(0.5 * (edges[1:] + edges[:-1]), k, k + hist, color=COLOUR_NICHE, alpha=0.7, lw=0)
        if fits is not None and (fits["time"] == t).any():
            row = fits[fits["time"] == t].iloc[0]
            ax.text(bins[-1], k + 0.1, f"ΔBIC {row['delta_bic']:.0f}, D {row['ashman_d']:.1f}", fontsize=5.5,
                    ha="right", color=COLOUR_SWITCH if row["is_bimodal"] else "0.4")
    ax.set_yticks(range(len(times)), [str(t) for t in times])
    ax.set_ylabel("year")
    ax.set_xlabel("log$_{10}$ niche breadth σ")
    return fig


def plot_bimodal_fraction(table: pd.DataFrame, x: str) -> Figure:
    fig, ax = plt.subplots(figsize=FIGURE_SINGLE, constrained_layout=True)
    xs = np.arange(len(table))
    ax.errorbar(xs, table["fraction"], yerr=[table["fraction"] - table["ci_low"], table["ci_high"] - table["fraction"]],
                fmt="o", color=COLOUR_NICHE, ms=4, lw=0.8)
    ax.set_xticks(xs, [str(v) for v in table[x]], rotation=30, ha="right", fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of snapshots bimodal in σ")
    ax.set_xlabel(_pretty(x))
    return fig


# ┌────────────────────────────────────────────────────────────┐
# │ Taxa  « the four animals, side by side »                   │
# └────────────────────────────────────────────────────────────┘

TAXON_ORDER: tuple[str, ...] = ("arthropod", "frog", "mouse", "primate")
TRAIT_COLOURS: dict[str, str] = {"Ef": COLOUR_FRONT, "Eb": COLOUR_BACK, "Ec": COLOUR_NICHE}


def plot_taxon_trajectories(histories: dict[str, list[pd.DataFrame]],
                            predictions: dict[str, dict[str, float]] | None = None) -> Figure:
    """One column per taxon, median and interquartile band of E_f, E_b, E_c
    across replicates; dotted lines are the mean-field optima."""
    names = [n for n in TAXON_ORDER if n in histories] + [n for n in histories if n not in TAXON_ORDER]
    fig, axes = plt.subplots(1, len(names), figsize=(1.9 * len(names) + 0.6, 2.4), sharey=True, constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        runs = histories[name]
        for trait, colour in TRAIT_COLOURS.items():
            if f"{trait}_mean" not in runs[0]:
                continue
            stack = np.vstack([h[f"{trait}_mean"].to_numpy() for h in runs])
            t = runs[0]["time"].to_numpy() if "time" in runs[0] else np.arange(stack.shape[1])
            lo, med, hi = np.nanpercentile(stack, [25, 50, 75], axis=0)
            ax.fill_between(t, lo, hi, color=colour, alpha=0.25, lw=0)
            ax.plot(t, med, color=colour, lw=1.2, label=f"${trait[0]}_{trait[1]}$")
            if predictions and trait in predictions.get(name, {}):
                ax.axhline(predictions[name][trait], color=colour, lw=0.7, ls=":")
        ax.set_title(name)
        ax.set_xlabel("year")
    np.atleast_1d(axes)[0].set_ylabel("investment (energy yr$^{-1}$)")
    np.atleast_1d(axes)[0].legend(fontsize=6, frameon=False)
    return fig


def plot_taxon_endstates(runs: pd.DataFrame, response: str = "a_index") -> Figure:
    """Strip + median of an end-state index per taxon."""
    names = [n for n in TAXON_ORDER if n in set(runs["taxon"])] + sorted(set(runs["taxon"]) - set(TAXON_ORDER))
    fig, ax = plt.subplots(figsize=FIGURE_SINGLE, constrained_layout=True)
    for i, name in enumerate(names):
        vals = runs.loc[runs["taxon"] == name, response].to_numpy(dtype=float)
        ax.scatter(i + np.random.default_rng(i).normal(0, JITTER, len(vals)), vals, s=9, color=COLOUR_MIXED, alpha=0.6,
                   edgecolor="none")
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color="black", lw=1.2)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(_pretty(response))
    return fig


def _pretty(name: str) -> str:
    return {
        "cost__Fmax_f": "$F_{\\max,f}$", "atp__Fmax_f": "$F_{\\max,f}$", "cost__M": "$M$", "atp__M": "$M$",
        "cost__P": "overhead $P$", "capsule__pO2_kPa": "$p$O$_2$ (kPa)",
        "atp__min_circuit": "minimal circuit $n_{\\min}$",
        "atp__back_share": "back-end share of basal", "atp__front_share": "front-end share of basal",
        "atp__front_exponent": "front exponent α", "atp__wiring_exponent": "wiring exponent γ",
        "atp__half_saturation": "half-saturation $n_{1/2}$", "capsule__transport": "carriage cost",
        "capsule__build_years": "build cost (yr)", "capsule__tracheal_exponent": "tracheal exponent",
        "year_established": "establishment year", "Eb": "$E_b$ (energy yr$^{-1}$)",
        "n_eff": "neurons (fraction of ceiling)",
        "Fb": "$F_b$", "sigma": "niche breadth σ", "a_index": "back-end share $E_b/(E_f+E_b)$",
        "brain_share": "brain cost, fraction of basal",
    }.get(name, name.replace("__", " "))


def _level(value: object) -> str:
    return f"{value:g}" if isinstance(value, (int, float, np.floating)) else str(value)


__all__ = ["plot_bimodal_fraction", "plot_grid_maps", "plot_hazard_vs_deficit", "plot_oxygen", "plot_sensitivity",
           "plot_sigma_through_time", "plot_taxon_endstates", "plot_taxon_trajectories"]
