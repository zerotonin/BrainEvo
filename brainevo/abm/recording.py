# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm.recording                                        ║
# ║  « population statistics, one row per year »                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Keeps the yearly log (traits, fidelities, demography, niche,    ║
# ║  trait correlations) and individual-level snapshots.  Traits     ║
# ║  are measured over survivors: that is the evolved state.         ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

from brainevo.constants import NAN, History

if TYPE_CHECKING:
    from brainevo.abm.organism import Organism

LOG_KEYS: tuple[str, ...] = (
    # demography
    "population_size", "births", "deaths", "deaths_starved", "deaths_senescent",
    # architecture
    "committed_frac", "above_switch_frac",
    "a_norm_mean", "a_norm_sd", "a_raw_mean", "a_raw_sd", "a_F_mean", "a_F_sd",
    "Ef_mean", "Ef_sd", "Eb_mean", "Eb_sd", "Ec_mean", "Ec_sd", "neural_spend_mean",
    "Ff_mean", "Ff_sd", "Fb_mean", "Fb_sd",
    "fidelity_mean", "fidelity_sd",
    "tuning_efficiency_mean", "tuning_efficiency_sd",
    "fidelity_realised_mean", "fidelity_realised_sd",
    # foraging and energetics
    "items_consumed_mean", "items_consumed_sd",
    "energy", "energy_pre_repro", "reproductive_frac", "age",
    # niche
    "mu", "mu_sd", "sigma", "sigma_sd",
    # architecture <-> niche hypothesis, measured every year
    "rho_sigma_Ef", "rho_sigma_Eb", "rho_sigma_a_norm", "rho_sigma_tuning",
    "rho_absmu_Ef", "rho_absmu_Eb",
    # large-effect events
    "jumps", "losses",
    # environment
    "stock_depleted", "standing_stock", "harvestable_stock",
)

SNAPSHOT_KEYS: tuple[str, ...] = (
    "Ef", "Eb", "Ec", "Ff", "Fb", "mu", "sigma", "a_norm", "a_raw", "a_F",
    "tuning_efficiency", "fidelity_realised", "above_switch", "age", "energy",
)

_TRAIT_KEYS: tuple[str, ...] = tuple(
    k for k in LOG_KEYS
    if k not in ("population_size", "births", "deaths", "deaths_starved", "deaths_senescent",
                 "stock_depleted", "standing_stock", "harvestable_stock",
                 "energy_pre_repro", "reproductive_frac")
)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, NaN-safe; NaN when either variable has no variance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return NAN
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return NAN
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def _nanstats(values: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values)
    if not finite.any():
        return NAN, NAN
    return float(np.nanmean(values)), float(np.nanstd(values))


class Recorder:
    """Yearly population log plus individual snapshots."""

    def __init__(self, e_switch: float, env_mu: float) -> None:
        self.e_switch = float(e_switch)
        self.env_mu = float(env_mu)
        self.log: dict[str, list[float]] = {key: [] for key in LOG_KEYS}
        self.snapshots: list[dict[str, float | int | bool]] = []

    # ┌────────────────────────────────────────────────────────────┐
    # │ Yearly record                                              │
    # └────────────────────────────────────────────────────────────┘

    def record(
        self,
        population: list[Organism],
        stock: np.ndarray,
        *,
        births: int = 0,
        starved: int = 0,
        senescent: int = 0,
        items_taken: float = NAN,
        items_per_organism: np.ndarray | None = None,
        energy_pre_repro: float = NAN,
        reproductive_frac: float = NAN,
        jumps: int = 0,
        losses: int = 0,
    ) -> None:
        log = self.log
        log["population_size"].append(len(population))
        log["births"].append(int(births))
        log["deaths"].append(int(starved + senescent))
        log["deaths_starved"].append(int(starved))
        log["deaths_senescent"].append(int(senescent))
        log["standing_stock"].append(float(stock.sum()))
        log["harvestable_stock"].append(float(stock[stock >= 1.0].sum()))
        log["stock_depleted"].append(float(items_taken))
        log["energy_pre_repro"].append(float(energy_pre_repro))
        log["reproductive_frac"].append(float(reproductive_frac))

        if not population:
            for key in _TRAIT_KEYS:
                log[key].append(NAN)
            return

        n = len(population)
        pull = lambda attr: np.fromiter((getattr(o, attr) for o in population), float, n)  # noqa: E731
        Ef, Eb, Ec, Ff, Fb = pull("Ef"), pull("Eb"), pull("Ec"), pull("Ff"), pull("Fb")
        a_norm, a_raw, a_F = pull("a_norm"), pull("a_raw"), pull("a_F")
        F, mu, sigma = pull("Fidelity"), pull("mu"), pull("sigma")
        tuning, realised = pull("tuning_efficiency"), pull("fidelity_realised")

        committed = np.isfinite(a_norm)
        log["committed_frac"].append(float(committed.mean()))
        log["above_switch_frac"].append(float(np.mean(Eb > self.e_switch)))
        for key, arr in (("a_norm", a_norm), ("a_raw", a_raw), ("a_F", a_F),
                         ("tuning_efficiency", tuning), ("fidelity_realised", realised)):
            m, s = _nanstats(arr)
            log[f"{key}_mean"].append(m)
            log[f"{key}_sd"].append(s)
        for key, arr in (("Ef", Ef), ("Eb", Eb), ("Ec", Ec), ("Ff", Ff), ("Fb", Fb), ("fidelity", F)):
            log[f"{key}_mean"].append(float(arr.mean()))
            log[f"{key}_sd"].append(float(arr.std()))
        log["neural_spend_mean"].append(float((Ef + Eb + Ec).mean()))
        log["jumps"].append(int(jumps))
        log["losses"].append(int(losses))

        abs_mu_offset = np.abs(mu - self.env_mu)
        log["rho_sigma_Ef"].append(spearman(sigma, Ef))
        log["rho_sigma_Eb"].append(spearman(sigma, Eb))
        log["rho_sigma_a_norm"].append(spearman(sigma, a_norm))
        log["rho_sigma_tuning"].append(spearman(sigma, tuning))
        log["rho_absmu_Ef"].append(spearman(abs_mu_offset, Ef))
        log["rho_absmu_Eb"].append(spearman(abs_mu_offset, Eb))

        items = np.empty(0) if items_per_organism is None else items_per_organism
        log["items_consumed_mean"].append(float(items.mean()) if items.size else NAN)
        log["items_consumed_sd"].append(float(items.std()) if items.size else NAN)
        log["energy"].append(float(pull("energy").mean()))
        log["age"].append(float(pull("age").mean()))
        log["mu"].append(float(mu.mean()))
        log["mu_sd"].append(float(mu.std()))
        log["sigma"].append(float(sigma.mean()))
        log["sigma_sd"].append(float(sigma.std()))

    # ┌────────────────────────────────────────────────────────────┐
    # │ Snapshots and export                                       │
    # └────────────────────────────────────────────────────────────┘

    def snapshot(self, population: list[Organism], time: int) -> None:
        """Append one row per surviving organism."""
        for i, o in enumerate(population):
            row: dict[str, float | int | bool] = {"time": time, "agent": i}
            for key in SNAPSHOT_KEYS:
                row[key] = getattr(o, key)
            self.snapshots.append(row)

    def history(self, extinct: bool, runaway: bool) -> History:
        lengths = {k: len(v) for k, v in self.log.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"ragged log: {lengths}")
        h: History = {k: np.array(v) for k, v in self.log.items()}
        h["time"] = np.arange(len(self.log["population_size"]))
        h["extinct"] = np.array(extinct)
        h["runaway"] = np.array(runaway)
        return h
