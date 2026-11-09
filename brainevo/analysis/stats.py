# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — analysis.stats                                       ║
# ║  « re-randomisation tests via reRandomStats »                    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  All inference goes through the lab's reRandomStats toolkit: Fish║
# ║  resampling for two-sample contrasts, MultiGroupTest with BH-FDR ║
# ║  across parameter levels, two-sample proportion tests for        ║
# ║  establishment vs retention, Wilson intervals for bimodal fractio║
# ║  and the 4PL Hill fit for the overhead at which establishment hal║
# ║  Every test runs on every cell; thresholds are applied when readi║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import numpy as np
import pandas as pd
from rerandomstats import (
    BinomialStats,
    FisherResamplingTest,
    MultiGroupTest,
    MultipleBinomialTests,
    correct_pvalues,
    hill_fit,
)

from brainevo.analysis.establishment import NAIVE, SEEDED

RESAMPLES: int = 10_000
SEED: int = 20261109
MIN_GROUP: int = 3


def naive_vs_seeded(runs: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    """Per grid cell: does seeding change the end state?

    Establishment (naive) against retention (seeded) by a two-sample
    z-test on proportions, and tail E_b by Fisher resampling of the mean
    difference.  Both p-value families are BH-corrected across cells.
    """
    rows = []
    for key, cell in runs.groupby(axes, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        naive = cell[cell["init__arch_mode"] == NAIVE]
        seeded = cell[cell["init__arch_mode"] == SEEDED]
        row = dict(zip(axes, key, strict=True))
        row.update(n_naive=len(naive), n_seeded=len(seeded), p_proportion=np.nan, p_Eb=np.nan)
        if len(naive) >= MIN_GROUP and len(seeded) >= MIN_GROUP:
            k_est, k_ret = int(naive["established"].sum()), int(seeded["retained"].sum())
            if 0 < k_est + k_ret < len(naive) + len(seeded):
                row["p_proportion"] = MultipleBinomialTests(
                    (k_est, len(naive) - k_est), (k_ret, len(seeded) - k_ret), "ztest").main()
            else:  # identical outcomes everywhere: no test possible, difference is zero
                row["p_proportion"] = 1.0
            row["p_Eb"] = _fisher(naive["Eb"], seeded["Eb"])
        rows.append(row)
    table = pd.DataFrame(rows)
    for col in ("p_proportion", "p_Eb"):
        table[f"{col}_fdr"] = _fdr(table[col])
    return table


def across_levels(runs: pd.DataFrame, parameter: str, response: str) -> pd.DataFrame:
    """Pairwise Fisher resampling of ``response`` between every pair of
    ``parameter`` levels, BH-corrected (reRandomStats MultiGroupTest)."""
    data = runs[[parameter, response]].dropna()
    data = data[np.isfinite(data[response])]
    counts = data[parameter].value_counts()
    data = data[data[parameter].isin(counts[counts >= MIN_GROUP].index)]
    if data[parameter].nunique() < 2:
        return pd.DataFrame()
    groups = [_level_label(v) for v in data[parameter]]
    table = MultiGroupTest(data[response].tolist(), groups, "Fisher:meanDiff", RESAMPLES, seed=SEED).main()
    table.insert(0, "response", response)
    table.insert(0, "parameter", parameter)
    return table


def two_sample(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    """Two-sided Fisher resampling p-value for a difference in means."""
    return _fisher(a, b)


def bimodal_fraction(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Fraction of snapshots called bimodal per group with a Wilson 95 % interval."""
    rows = []
    for key, grp in table.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        k, n = int(grp["is_bimodal"].sum()), len(grp)
        ci = BinomialStats(k, n).exact_ci() if n else {"Lower CI": np.nan, "Upper CI": np.nan}  # percentages
        rows.append({**dict(zip(by, key, strict=True)), "n_snapshots": n, "n_bimodal": k,
                     "fraction": k / n if n else np.nan,
                     "ci_low": ci["Lower CI"] / 100.0, "ci_high": ci["Upper CI"] / 100.0,
                     "p_vs_zero": _p_above_chance(k, n)})
    return pd.DataFrame(rows)


def _p_above_chance(k: int, n: int, base_rate: float = 0.05) -> float:
    """One-sided binomial test that the bimodal fraction exceeds a nominal false-call rate."""
    return float(BinomialStats(k, n, alternative="greater").binomial_test(base_rate).pvalue) if n else np.nan


def half_establishment(x: np.ndarray, p_establish: np.ndarray) -> dict[str, float]:
    """4PL Hill fit of establishment probability against a control parameter.

    The fitted EC50 is the parameter value at which establishment halves,
    the quantitative version of the thesis' Fig-7 boundary; ``hill_n`` says
    how sharp that boundary is.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(p_establish, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if keep.sum() < 4 or np.ptp(y[keep]) < 0.05:
        return {}
    return {k: float(v) for k, v in hill_fit(x[keep], y[keep]).items() if np.isscalar(v)}


def _fisher(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
        return np.nan
    if np.ptp(np.concatenate([a, b])) == 0.0:
        return 1.0
    return float(FisherResamplingTest(list(a), list(b), "meanDiff", RESAMPLES, seed=SEED).main())


def _fdr(p: pd.Series) -> pd.Series:
    valid = {str(i): float(v) for i, v in p.items() if np.isfinite(v)}
    if not valid:
        return pd.Series(np.nan, index=p.index)
    results = correct_pvalues(valid, method="fdr_bh")["results"]
    return pd.Series({i: float(results[str(i)]["adjusted_p"]) if str(i) in results else np.nan for i in p.index})


def _level_label(value: object) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)
