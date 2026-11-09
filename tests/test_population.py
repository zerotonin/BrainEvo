from __future__ import annotations

import numpy as np

from brainevo.abm import (
    Architecture,
    Environment,
    InitialisationParams,
    LifeHistoryParams,
    MutationParams,
    Population,
)
from brainevo.abm.presets import EXP2_TUNING, PRESETS


def _small_population(seed: int = 3, **init: object) -> Population:
    arch = Architecture(EXP2_TUNING.cost)
    env = Environment(EXP2_TUNING.environment)
    return Population(env, arch, LifeHistoryParams(uniform_foraging=False), MutationParams(),
                      InitialisationParams(n_founders=120, **init), rng=seed)  # type: ignore[arg-type]


def test_runs_and_logs_every_year() -> None:
    pop = _small_population()
    hist, snaps = pop.run(30, snapshot_times={10, 30})
    assert len(hist["population_size"]) == 31  # founders + 30 years
    assert hist["population_size"][-1] > 0
    assert {row["time"] for row in snaps} == {10, 30}


def test_same_seed_same_history() -> None:
    h1, _ = _small_population(seed=11).run(25)
    h2, _ = _small_population(seed=11).run(25)
    for key in ("population_size", "Ef_mean", "Eb_mean", "sigma"):
        assert np.array_equal(h1[key], h2[key], equal_nan=True)


def test_naive_founders_have_no_investment() -> None:
    pop = _small_population(arch_mode="naive")
    assert all(o.Ef == 0.0 and o.Eb == 0.0 for o in pop.population)


def test_above_switch_founders_are_above_switch() -> None:
    pop = _small_population(arch_mode="above_switch")
    assert all(o.above_switch for o in pop.population)


def test_split_founders_match_fraction() -> None:
    pop = _small_population(arch_mode="split", backend_fraction=0.5)
    frac = np.mean([o.above_switch for o in pop.population])
    assert 0.3 < frac < 0.7


def test_mutation_never_makes_negative_investment() -> None:
    pop = _small_population()
    for _ in range(2000):
        _, _, Ef, Eb = pop.mutate(50.0, 15.0, 0.0, 0.0)
        assert Ef >= 0.0 and Eb >= 0.0


def test_large_effect_gain_lands_past_switch() -> None:
    pop = _small_population()
    for _ in range(50):
        assert pop._backend_gain() > pop.e_switch


def test_cull_removes_starved_and_old() -> None:
    pop = _small_population()
    pop.population[0].energy = -1.0
    pop.population[1].age = pop.life.max_age
    n = len(pop.population)
    starved, senescent = pop.cull()
    assert starved >= 1 and senescent >= 1
    assert len(pop.population) == n - starved - senescent


def test_presets_override_and_roundtrip() -> None:
    p = PRESETS["exp1-cost-sweep"].with_overrides(cost__M=4.0, cost__P=1.0)
    assert p.cost.M == 4.0 and p.cost.P == 1.0
    assert p.to_dict()["cost"]["M"] == 4.0
