from __future__ import annotations

import numpy as np
import pytest

from brainevo.abm import Environment, EnvironmentParams


def test_capacity_is_gaussian_and_centred(env: Environment) -> None:
    assert env.capacity.argmax() == env.env_mu
    assert env.capacity.max() == pytest.approx(env.capacity_peak)


def test_preference_distribution_normalises(env: Environment) -> None:
    p = env.preference_distribution(50.0, 5.0)
    assert p is not None
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 50


def test_off_axis_niche_is_none(env: Environment) -> None:
    assert env.preference_distribution(-500.0, 0.1) is None


def test_take_respects_refuge(env: Environment) -> None:
    z = env.env_mu
    env.stock[z] = env.refuge[z] + 0.5
    assert env.take(z) is False
    env.stock[z] = env.refuge[z] + 1.0
    assert env.take(z) is True


def test_regrowth_is_logistic_and_bounded(env: Environment) -> None:
    env.stock[:] = 0.5 * env.capacity
    before = env.stock.copy()
    env.regrow()
    assert np.all(env.stock >= before)
    assert np.all(env.stock <= env.capacity + 1e-9)


def test_global_volatility_moves_all_types_together() -> None:
    env = Environment(EnvironmentParams(volatility=0.25, volatility_corr_len=float("inf")))
    env.fluctuate(np.random.default_rng(0))
    ratio = env.capacity / env.capacity_base
    assert np.allclose(ratio, ratio[0])


def test_volatility_is_mean_preserving() -> None:
    env = Environment(EnvironmentParams(volatility=0.25, volatility_corr_len=0.0))
    rng = np.random.default_rng(1)
    ratios = []
    for _ in range(4000):
        env.fluctuate(rng)
        ratios.append(env.capacity[env.env_mu] / env.capacity_base[env.env_mu])
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.02)


def test_smoothed_kernel_preserves_variance() -> None:
    env = Environment(EnvironmentParams(volatility=0.3, volatility_corr_len=3.0))
    assert env.shock_mode == "smoothed"
    assert env._shock_kernel is not None
    assert (env._shock_kernel ** 2).sum() == pytest.approx(1.0)
