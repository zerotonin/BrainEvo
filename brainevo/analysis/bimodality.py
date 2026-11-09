# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — analysis.bimodality                                  ║
# ║  « does the niche split inside a run? »                          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Two-component Gaussian mixture by EM on one-dimensional samples ║
# ║  (niche breadth sigma of every individual in a snapshot), compare║
# ║  with one Gaussian by BIC, plus Ashman's D for separation. A snap║
# ║  counts as bimodal when dBIC > 10, D > 2 and both weights > 0.1. ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from brainevo.analysis.loading import load_snapshots

DELTA_BIC_MIN: float = 10.0
ASHMAN_D_MIN: float = 2.0
MIN_WEIGHT: float = 0.1
VARIANCE_FLOOR: float = 1e-6


@dataclass(frozen=True)
class TwoComponentFit:
    weights: tuple[float, float]
    means: tuple[float, float]
    sds: tuple[float, float]
    log_likelihood: float
    delta_bic: float
    """BIC(one Gaussian) - BIC(two Gaussians); positive favours two."""
    ashman_d: float

    @property
    def is_bimodal(self) -> bool:
        return bool(self.delta_bic > DELTA_BIC_MIN and self.ashman_d > ASHMAN_D_MIN and min(self.weights) > MIN_WEIGHT)


def fit_two_component(x: np.ndarray, n_iter: int = 300, log_scale: bool = True) -> TwoComponentFit:
    """EM for a two-Gaussian mixture, initialised at the lower and upper quartiles."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if log_scale:
        x = np.log(x[x > 0])
    n = len(x)
    if n < 10:
        return TwoComponentFit((np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan), np.nan, np.nan, np.nan)
    mu = np.quantile(x, [0.25, 0.75])
    sd = np.full(2, max(x.std(), 1e-3))
    w = np.array([0.5, 0.5])
    for _ in range(n_iter):
        dens = w * _normal_pdf(x[:, None], mu, sd)
        resp = dens / np.maximum(dens.sum(axis=1, keepdims=True), 1e-300)
        nk = resp.sum(axis=0)
        w = nk / n
        mu = (resp * x[:, None]).sum(axis=0) / np.maximum(nk, 1e-12)
        var = (resp * (x[:, None] - mu) ** 2).sum(axis=0) / np.maximum(nk, 1e-12)
        sd = np.sqrt(np.maximum(var, VARIANCE_FLOOR))
    ll2 = float(np.log(np.maximum((w * _normal_pdf(x[:, None], mu, sd)).sum(axis=1), 1e-300)).sum())
    ll1 = float(np.log(np.maximum(_normal_pdf(x, x.mean(), max(x.std(), 1e-3)), 1e-300)).sum())
    bic1 = 2 * np.log(n) - 2 * ll1
    bic2 = 5 * np.log(n) - 2 * ll2
    order = np.argsort(mu)
    mu, sd, w = mu[order], sd[order], w[order]
    d = float(np.sqrt(2.0) * abs(mu[1] - mu[0]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2))
    return TwoComponentFit((float(w[0]), float(w[1])), (float(mu[0]), float(mu[1])), (float(sd[0]), float(sd[1])),
                           ll2, float(bic1 - bic2), d)


def bimodality_table(runs: pd.DataFrame, column: str = "sigma") -> pd.DataFrame:
    """Fit every snapshot of every run with snapshots; one row per (run, time)."""
    rows = []
    for _, run in runs[runs["has_snapshots"]].iterrows():
        snaps = load_snapshots(Path(run["run_dir"]))
        for time, frame in snaps.groupby("time"):
            fit = fit_two_component(frame[column].to_numpy())
            rows.append({
                "sweep": run["sweep"], "preset": run["preset"], "tag": run["tag"], "seed": run["seed"],
                "time": int(time),
                "n": len(frame), "delta_bic": fit.delta_bic, "ashman_d": fit.ashman_d,
                "w_low": fit.weights[0], "mean_low": np.exp(fit.means[0]), "mean_high": np.exp(fit.means[1]),
                "is_bimodal": fit.is_bimodal,
            })
    return pd.DataFrame(rows)


def _normal_pdf(x: np.ndarray, mu: np.ndarray | float, sd: np.ndarray | float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))
