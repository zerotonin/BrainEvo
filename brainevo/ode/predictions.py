# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — ode.predictions                                      ║
# ║  « necessity table and emergent-feature tests »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Mean-field answers to the questions the free variables must pass║
# ║  the ablation table (which component each taxon phenotype needs),║
# ║  Haller's rule and brain-body allometry from the prices principle║
# ║  brain reduction against lost demand, reversibility of a reductio║
# ║  and the absolute minimal circuit implied by the smallest brains.║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from brainevo.abm.ablations import ABLATION_NAMES, ablate
from brainevo.abm.energetics import ATPArchitecture, ATPCostLawParams
from brainevo.abm.taxa import TAXA, Taxon
from brainevo.ode.canonical import relax
from brainevo.ode.meanfield import MeanField

KLEIBER_EXPONENT: float = 0.75
"""Basal metabolic rate ∝ body mass^0.75 (Kleiber 1932; White & Seymour 2003 for mammals)."""
CRANIUM_EXPONENT: float = 1.0
"""Cranial volume isometric with body mass; the capacity *relative to the ceiling count* then scales
as mass^(1 - 0.75)."""
OBSERVED_BRAIN_ALLOMETRY: tuple[float, float] = (0.67, 0.75)
"""Brain mass ∝ body mass^0.67-0.75 across mammals (Martin 1981; Burger et al. 2019 PLoS ONE)."""


@dataclass(frozen=True)
class Optimum:
    Ef: float
    Eb: float
    Ec: float
    neurons: float
    cap: float
    brain_share: float
    retina_share: float
    Q_ratio: float
    deficit: float

    @property
    def a_index(self) -> float:
        return self.Eb / (self.Ef + self.Eb) if self.Ef + self.Eb > 0 else 0.0


def optimum(arch: ATPArchitecture, basal: float | None = None) -> Optimum:
    mf = MeanField(arch, basal=arch.basal if basal is None else basal)
    front = mf.best_front_only()
    mixed = mf.best_mixed()
    if mixed is None:
        return Optimum(front.Ef, 0.0, 0.0, 0.0, float(arch.hard_cap), 0.0, front.Ef / mf.basal, 1.0,
                       mf.valley_deficit())
    Ec = float(arch.optimal_capsule(mixed.Eb))
    n = float(arch.effective_neurons(mixed.Eb, Ec)) if arch.has_capsule else float(arch.effective_neurons(mixed.Eb))
    return Optimum(mixed.Ef, mixed.Eb, Ec, n, float(arch.hard_cap), (mixed.Eb + Ec) / mf.basal, mixed.Ef / mf.basal,
                   mixed.Q / front.Q, mf.valley_deficit())


# ┌────────────────────────────────────────────────────────────┐
# │ Necessity  « ablation table »                              │
# └────────────────────────────────────────────────────────────┘

def ablation_table() -> pd.DataFrame:
    """Every taxon under the intact law and each ablation."""
    rows = []
    for taxon in TAXA.values():
        for name in ("intact", *ABLATION_NAMES):
            preset = ablate(taxon, name) if name != "intact" else None
            arch = taxon.architecture() if preset is None else preset.build_architecture()
            if not isinstance(arch, ATPArchitecture):  # thesis law
                mf = MeanField(arch)
                front, mixed = mf.best_front_only(), mf.best_mixed()
                Eb = mixed.Eb if mixed else 0.0
                rows.append({"taxon": taxon.name, "ablation": name, "Ef": mixed.Ef if mixed else front.Ef, "Eb": Eb,
                             "Ec": 0.0, "neurons": np.nan, "cap": np.nan, "brain_share": Eb / mf.basal,
                             "retina_share": (mixed.Ef if mixed else front.Ef) / mf.basal,
                             "a_index": Eb / ((mixed.Ef if mixed else front.Ef) + Eb) if Eb else 0.0,
                             "Q_ratio": (mixed.Q / front.Q) if mixed else 1.0, "deficit": mf.valley_deficit()})
                continue
            o = optimum(arch)
            rows.append({"taxon": taxon.name, "ablation": name, **o.__dict__, "a_index": o.a_index})
    return pd.DataFrame(rows)


# ┌────────────────────────────────────────────────────────────┐
# │ Emergent features  « what the free variables must reproduce »│
# └────────────────────────────────────────────────────────────┘

def scale_to_body_mass(atp: ATPCostLawParams, ratio: float) -> ATPCostLawParams:
    """Rescale a taxon's law to a body ``ratio`` times the reference.

    Prices are shares of basal, so they follow Kleiber; the minimal circuit is
    an absolute neuron number, so its *fraction* of the ceiling count falls as
    basal grows; cranial capacity is isometric with body mass, so relative to
    the ceiling count it grows as mass^(1 - 0.75).
    """
    basal_factor = ratio**KLEIBER_EXPONENT
    capsule = None if atp.capsule is None else replace(
        atp.capsule, hard_cap_ref=atp.capsule.hard_cap_ref * ratio ** (CRANIUM_EXPONENT - KLEIBER_EXPONENT))
    return replace(atp, basal=atp.basal * basal_factor, min_circuit=min(atp.min_circuit / basal_factor, 0.9),
                   capsule=capsule)


def haller_curve(taxon: Taxon, ratios: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Realised brain investment against body mass; returns the table and the
    fitted allometric exponent of absolute neuron number against body mass.

    Haller's rule (relatively larger brains in smaller animals) appears when the
    exponent is below 1; the measured brain-body exponent is 0.67-0.75.
    """
    rows = []
    for r in ratios:
        atp = scale_to_body_mass(taxon.atp, float(r))
        arch = ATPArchitecture(atp)
        o = optimum(arch)
        ceiling_count = arch.basal * atp.back_share  # energy of the ceiling count, ∝ neurons at constant price
        rows.append({"taxon": taxon.name, "mass_ratio": r, "basal": arch.basal, "neurons_rel": o.neurons,
                     "neurons_abs": o.neurons * ceiling_count, "brain_share": o.brain_share,
                     "brain_per_body": o.neurons * ceiling_count / r, "retina_share": o.retina_share, "cap": o.cap})
    table = pd.DataFrame(rows)
    ok = table["neurons_abs"] > 0
    slope = float(np.polyfit(np.log(table.loc[ok, "mass_ratio"]), np.log(table.loc[ok, "neurons_abs"]), 1)[0])
    return table, slope


def reduction_vs_demand(taxon: Taxon, M_values: np.ndarray) -> pd.DataFrame:
    """Neuron fraction retained when the back-end's return M is lowered from the
    taxon's own value; the model's prediction for domestication, cave life and
    parasitism, which remove demand without changing physiology."""
    reference = optimum(taxon.architecture())
    rows = []
    for M in M_values:
        o = optimum(ATPArchitecture(replace(taxon.atp, M=float(M))))
        rows.append({"taxon": taxon.name, "M": M, "M_rel": M / taxon.atp.M, "neurons": o.neurons,
                     "fraction_retained": o.neurons / reference.neurons if reference.neurons else np.nan,
                     "brain_share": o.brain_share, "back_end_present": o.Q_ratio > 1.0 and o.Eb > 0})
    return pd.DataFrame(rows)


def reversibility(taxon: Taxon, M_values: np.ndarray) -> pd.DataFrame:
    """Quasi-static ramp of M down and back up, following the local attractor
    with the canonical equation.  A valley shows as a loop; grounded costs
    predict the down and up branches coincide."""
    M_values = np.sort(np.asarray(M_values, dtype=float))
    start = optimum(ATPArchitecture(replace(taxon.atp, M=float(M_values[-1]))))
    x = (start.Ef, start.Eb)
    rows = []
    for direction, Ms in (("down", M_values[::-1]), ("up", M_values)):
        for M in Ms:
            arch = ATPArchitecture(replace(taxon.atp, M=float(M)))
            mf = MeanField(arch, basal=arch.basal)
            x = relax(mf, x)
            rows.append({"taxon": taxon.name, "direction": direction, "M": M, "Ef": x[0], "Eb": x[1],
                         "neurons": float(arch.effective_neurons(x[1], arch.optimal_capsule(x[1])))})
    return pd.DataFrame(rows)


SMALLEST_BRAINS: dict[str, tuple[float, str]] = {
    "Megaphragma mymaripenne (parasitoid wasp)": (7_400, "Polilov 2012 Arthropod Struct Dev 41:29"),
    "Caenorhabditis elegans": (302, "White et al. 1986 Phil Trans R Soc B 314:1"),
    "Drosophila melanogaster adult brain": (1.0e5, "Raji & Potter 2021 PLoS ONE 16:e0250381"),
}
INSECT_CEILING_NEURONS: tuple[float, str] = (9.6e5, "honeybee brain ~960,000 neurons (Menzel & Giurfa 2001)")


def minimal_circuit_bounds(hard_cap_ref: float = 0.35) -> pd.DataFrame:
    """Absolute minimal circuit implied by the smallest functional brains.

    The insect ceiling count is the largest insect brain divided by the
    arthropod cap fraction; each small brain then gives an upper bound on
    n_min as a fraction of that ceiling.
    """
    ceiling = INSECT_CEILING_NEURONS[0] / hard_cap_ref
    rows = [{"brain": "insect ceiling count (implied)", "neurons": ceiling, "n_min_upper_bound": np.nan,
             "source": INSECT_CEILING_NEURONS[1]}]
    for name, (n, source) in SMALLEST_BRAINS.items():
        rows.append({"brain": name, "neurons": n, "n_min_upper_bound": n / ceiling, "source": source})
    return pd.DataFrame(rows)
