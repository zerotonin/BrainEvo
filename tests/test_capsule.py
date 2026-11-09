from __future__ import annotations

import numpy as np
import pytest

from brainevo.abm import (
    ATPArchitecture,
    ATPCostLawParams,
    CapsuleLaw,
    Environment,
    EnvironmentParams,
    InitialisationParams,
    LifeHistoryParams,
    MutationParams,
    Population,
)
from brainevo.ode import MeanField


def _arch(**capsule: object) -> ATPArchitecture:
    return ATPArchitecture(ATPCostLawParams(Fmax_f=1.2, M=2.5, capsule=CapsuleLaw(**capsule)))  # type: ignore[arg-type]


def test_tracheal_cap_falls_with_oxygen_faster_than_convective() -> None:
    low, modern = 15.0, 21.2
    def ratio(lineage: str) -> float:
        return CapsuleLaw(lineage=lineage, pO2_kPa=low).hard_cap / CapsuleLaw(lineage=lineage, pO2_kPa=modern).hard_cap

    trach, conv = ratio("tracheal"), ratio("convective")
    assert trach < 0.7
    assert 0.9 < conv < 1.0
    assert CapsuleLaw(lineage="tracheal", pO2_kPa=31.0).hard_cap > 1.5  # Carboniferous giants


def test_capsule_cost_is_a_minor_share_of_neural_cost() -> None:
    a = _arch(lineage="convective")
    assert 0.2 < a.capsule_cost(1.0) / a.cost_of_neurons(1.0) < 0.8
    assert a.capacity_for_cost(a.capsule_cost(0.37)) == pytest.approx(0.37, rel=1e-6)


def test_uncapsuled_neurons_return_nothing() -> None:
    a = _arch(lineage="convective")
    Eb = a.cost_of_neurons(0.5)
    assert a.b_capability(Eb, 0.0) == 0.0
    assert a.b_capability(Eb, a.optimal_capsule(Eb)) > 0.0
    # oversize capsule buys nothing extra
    fit = a.b_capability(Eb, a.optimal_capsule(Eb))
    assert a.b_capability(Eb, 3 * a.optimal_capsule(Eb)) == pytest.approx(fit, rel=0.02)


def test_hard_cap_limits_fidelity_for_low_oxygen_insect() -> None:
    hypoxic = _arch(lineage="tracheal", pO2_kPa=12.0)
    modern = _arch(lineage="tracheal", pO2_kPa=21.2)
    assert hypoxic.max_back_fidelity < 0.8 * modern.max_back_fidelity
    assert not np.isfinite(hypoxic.E_b_saturation)


def test_frog_test_transport_cost_shrinks_brain_then_capsule() -> None:
    """Raising the cost of carrying the head must reduce the optimal back-end
    and the capsule that houses it."""
    cheap = MeanField(_arch(lineage="convective", transport=0.1)).best_mixed()
    dear = MeanField(_arch(lineage="convective", transport=2.0)).best_mixed()
    assert cheap is not None and dear is not None
    assert dear.Eb < cheap.Eb
    a_cheap, a_dear = _arch(lineage="convective", transport=0.1), _arch(lineage="convective", transport=2.0)
    assert a_dear.neurons_for_cost(dear.Eb) < a_cheap.neurons_for_cost(cheap.Eb)


def test_lower_demand_shrinks_brain() -> None:
    rich = MeanField(ATPArchitecture(ATPCostLawParams(Fmax_f=1.2, M=2.5, capsule=CapsuleLaw()))).best_mixed()
    poor = MeanField(ATPArchitecture(ATPCostLawParams(Fmax_f=0.2, M=1.5, capsule=CapsuleLaw()))).best_mixed()
    assert rich is not None and poor is not None
    assert poor.Eb < rich.Eb


def test_abm_coevolves_capsule_with_backend() -> None:
    arch = _arch(lineage="convective")
    env = Environment(EnvironmentParams())
    pop = Population(env, arch, LifeHistoryParams(uniform_foraging=True), MutationParams(niche_mutation_rate=0.0),
                     InitialisationParams(n_founders=200, arch_mode="naive"), rng=5)
    hist, snaps = pop.run(300, snapshot_times={300})
    assert hist["population_size"][-1] > 0
    assert hist["Ec_mean"][-1] > 0.0 and hist["Eb_mean"][-1] > 0.0
    # founders with naive architecture start with no capsule
    assert hist["Ec_mean"][0] == 0.0
    assert all("Ec" in row for row in snaps)


def test_thesis_law_ignores_capsule_argument() -> None:
    from brainevo.abm import Architecture, CostLawParams

    a = Architecture(CostLawParams())
    assert a.b_capability(5.0) == a.b_capability(5.0, 0.3)
    assert a.has_capsule is False
