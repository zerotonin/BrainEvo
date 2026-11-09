"""The refactor must reproduce the thesis code exactly for a shared seed.

Both implementations consume the random stream in the same order, so every
logged statistic must agree to floating-point precision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from brainevo.abm import (
    Architecture,
    CostLawParams,
    Environment,
    EnvironmentParams,
    InitialisationParams,
    LifeHistoryParams,
    MutationParams,
    Population,
)

LEGACY = Path(__file__).resolve().parents[1] / "_legacy"
pytestmark = pytest.mark.skipif(not (LEGACY / "population.py").exists(), reason="legacy sources absent")

COST = dict(Fmax_f=0.8, k=3, SNR=14.44, M=3.0, P=3.75, back_rate=0.8)
YEARS = 40
SEED = 2026


def _legacy_run(uniform: bool, tuned: bool):
    sys.path.insert(0, str(LEGACY.parent))
    import importlib

    arch_mod = importlib.import_module("_legacy.architecture")
    env_mod = importlib.import_module("_legacy.environment")
    pop_mod = importlib.import_module("_legacy.population")
    arch = arch_mod.Architecture(**COST, tuning_width=3.75 if tuned else None)
    env = env_mod.Environment(n_resources=100, env_sigma=15.0, capacity_peak=400.0, regrowth_rate=1.0,
                              energy_per_item=1.0, refuge_frac=0.1)
    pop = pop_mod.Population(env, arch, initial_population=150, initial_energy=10.0, offspring_energy=5.0,
                             repro_scale=60.0, repro_cost=8.0, basal_metabolism=8.0, max_age=20,
                             E_mutation_sd=0.03, mu_mutation_sd=0.5, sigma_mutation_sd=0.1,
                             niche_mutation_rate=0.1, arch_mutation_rate=0.1, duplication_rate=0.01,
                             loss_rate=0.001, foraging_rounds=32, sigma_init=15.0, naive_ns=True,
                             uniform_foraging=uniform, rng=SEED)
    hist, _ = pop.run(years=YEARS)
    return hist


def _new_run(uniform: bool, tuned: bool):
    arch = Architecture(CostLawParams(**COST, tuning_width=3.75 if tuned else None))
    env = Environment(EnvironmentParams(n_resources=100, env_sigma=15.0, capacity_peak=400.0, refuge_frac=0.1))
    pop = Population(
        env, arch,
        LifeHistoryParams(uniform_foraging=uniform),
        MutationParams(duplication_rate=0.01),
        InitialisationParams(n_founders=150, niche_mode="fixed", sigma_init=15.0, arch_mode="naive"),
        rng=SEED,
    )
    hist, _ = pop.run(YEARS)
    return hist


@pytest.mark.parametrize("uniform,tuned", [(True, False), (False, True)])
def test_history_matches_legacy(uniform: bool, tuned: bool) -> None:
    old = _legacy_run(uniform, tuned)
    new = _new_run(uniform, tuned)
    for key in ("population_size", "births", "deaths_starved", "Ef_mean", "Eb_mean", "Ff_mean", "Fb_mean",
                "sigma", "mu", "a_norm_mean", "items_consumed_mean", "jumps", "losses", "standing_stock"):
        assert np.allclose(old[key], new[key], equal_nan=True, rtol=1e-10, atol=1e-10), key
