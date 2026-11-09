from __future__ import annotations

import pytest

from brainevo.abm import Architecture, CostLawParams, Environment, EnvironmentParams


@pytest.fixture
def arch() -> Architecture:
    return Architecture(CostLawParams(Fmax_f=0.8, M=3.0, P=3.75, back_rate=0.8))


@pytest.fixture
def tuned_arch() -> Architecture:
    return Architecture(CostLawParams(Fmax_f=0.8, M=3.0, P=3.75, back_rate=0.8, tuning_width=3.75))


@pytest.fixture
def env() -> Environment:
    return Environment(EnvironmentParams(n_resources=100, env_sigma=15.0, capacity_peak=400.0, refuge_frac=0.1))
