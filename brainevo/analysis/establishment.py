# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — analysis.establishment                               ║
# ║  « accessibility, measured as a waiting time »                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Per grid cell: establishment probability of naive founders, rete║
# ║  of seeded founders, and the constant-hazard rate of back-end    ║
# ║  establishment with right-censoring at the end of the run. The ha║
# ║  is then regressed on the mean-field valley deficit of the same c║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from brainevo.analysis.loading import architecture_from_params, load_params
from brainevo.ode.meanfield import MeanField

NAIVE: str = "naive"
SEEDED: str = "above_switch"


@dataclass(frozen=True)
class HazardFit:
    """log(hazard) = intercept + slope * deficit, least squares over cells with events."""

    slope: float
    intercept: float
    r_squared: float
    n_cells: int


def establishment_cells(runs: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    """One row per grid cell with naive and seeded statistics side by side.

    Args:
        runs: Table from :func:`load_runs`.
        axes: Override columns that define the grid, e.g. ``["cost__Fmax_f", "cost__M"]``.
    """
    rows = []
    for key, cell in runs.groupby(axes, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        naive = cell[cell["init__arch_mode"] == NAIVE]
        seeded = cell[cell["init__arch_mode"] == SEEDED]
        rate, lo, hi = exponential_hazard(naive["year_established"].to_numpy(), naive["years_run"].to_numpy())
        rows.append({
            **dict(zip(axes, key, strict=True)),
            "n_naive": len(naive), "n_seeded": len(seeded),
            "p_establish": float(naive["established"].mean()) if len(naive) else np.nan,
            "t_establish_mean": _mean_if_any(naive.loc[naive["established"], "year_established"]),
            "hazard": rate, "hazard_lo": lo, "hazard_hi": hi,
            "p_retained": float(seeded["retained"].mean()) if len(seeded) else np.nan,
            "Eb_naive": float(naive["Eb"].mean()) if len(naive) else np.nan,
            "Eb_seeded": float(seeded["Eb"].mean()) if len(seeded) else np.nan,
            "N_naive": float(naive["N"].mean()) if len(naive) else np.nan,
            "N_seeded": float(seeded["N"].mean()) if len(seeded) else np.nan,
            "example_run": str(cell["run_dir"].iloc[0]),
        })
    return pd.DataFrame(rows)


def _mean_if_any(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def exponential_hazard(year_established: np.ndarray, years_run: np.ndarray) -> tuple[float, float, float]:
    """Constant-hazard MLE with right-censoring and a 95 % interval.

    Runs that never established contribute their whole duration at risk.
    Returns ``(rate, lower, upper)`` in events per year; ``(0, 0, upper)``
    when no run established (upper from the rule of three).
    """
    year_established = np.asarray(year_established, dtype=float)
    years_run = np.asarray(years_run, dtype=float)
    if len(year_established) == 0:
        return np.nan, np.nan, np.nan
    events = year_established > 0
    exposure = float(np.where(events, year_established, years_run).sum())
    k = int(events.sum())
    if k == 0:
        return 0.0, 0.0, 3.0 / exposure
    rate = k / exposure
    half = 1.96 / np.sqrt(k)
    return rate, rate * np.exp(-half), rate * np.exp(half)


def hazard_vs_deficit(cells: pd.DataFrame) -> tuple[pd.DataFrame, HazardFit]:
    """Attach the mean-field valley deficit to each cell and fit log hazard against it."""
    deficits, qf, qm = [], [], []
    for run_dir in cells["example_run"]:
        mf = MeanField(architecture_from_params(load_params(run_dir)))
        deficits.append(mf.valley_deficit())
        qf.append(mf.best_front_only().Q)
        mixed = mf.best_mixed()
        qm.append(mixed.Q if mixed is not None else np.nan)
    out = cells.assign(valley_deficit=deficits, Q_front=qf, Q_mixed=qm)
    out["mixed_favoured"] = out["Q_mixed"] > out["Q_front"]
    fitted = out[(out["hazard"] > 0) & np.isfinite(out["valley_deficit"])]
    if len(fitted) < 3:
        return out, HazardFit(np.nan, np.nan, np.nan, len(fitted))
    x, y = fitted["valley_deficit"].to_numpy(), np.log(fitted["hazard"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - float(resid.var()) / float(y.var()) if y.var() > 0 else np.nan
    return out, HazardFit(float(slope), float(intercept), r2, len(fitted))
