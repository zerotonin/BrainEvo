from __future__ import annotations

import numpy as np
import pytest

from brainevo.abm import Architecture, CostLawParams
from brainevo.ode import NicheMeanField, Strategy


@pytest.fixture(scope="module")
def model() -> NicheMeanField:
    return NicheMeanField(Architecture(CostLawParams(Fmax_f=0.8, M=3.0, P=3.75, back_rate=0.8, tuning_width=3.75)))


def test_resident_equilibrium_is_viable_and_balanced(model: NicheMeanField) -> None:
    s = Strategy(0.3, 0.0, float(model.env.env_mu), 15.0)
    eq = model.resident_equilibrium(s)
    assert eq.viable and eq.N > 10
    assert model.invasion_fitness(s, eq) == pytest.approx(0.0, abs=1e-4)
    assert np.all(eq.stock >= model.env.refuge - 1e-9)
    assert np.all((0.0 <= eq.success) & (eq.success <= 1.0))


def test_stocks_are_depleted_most_where_the_resident_forages(model: NicheMeanField) -> None:
    mu = float(model.env.env_mu)
    eq = model.resident_equilibrium(Strategy(0.3, 0.0, mu, 5.0))
    centre = int(mu)
    assert eq.success[centre] < eq.success[0]


def test_specialist_resident_opens_room_for_broader_mutants(model: NicheMeanField) -> None:
    """Frequency dependence: in a narrow resident's depleted centre, a broader mutant
    finds untouched flanks."""
    mu = float(model.env.env_mu)
    narrow = Strategy(0.3, 0.0, mu, 4.0)
    eq = model.resident_equilibrium(narrow)
    assert model.invasion_fitness(Strategy(0.3, 0.0, mu, 8.0), eq) > 0.0


def test_gradient_vanishes_in_Eb_dead_zone_and_points_down(model: NicheMeanField) -> None:
    g = model.selection_gradient(Strategy(0.3, 0.0, float(model.env.env_mu), 15.0))
    assert g[1] == pytest.approx(-1.0, rel=1e-3)  # pure cost below the switch


def test_pip_diagonal_is_neutral(model: NicheMeanField) -> None:
    sigmas = np.array([6.0, 12.0, 24.0])
    S = model.pip_sigma(Strategy(0.3, 0.0, float(model.env.env_mu), 15.0), sigmas)
    assert np.allclose(np.diag(S), 0.0, atol=1e-4)


def test_classification_runs(model: NicheMeanField) -> None:
    cls = model.classify(Strategy(0.3, 0.0, float(model.env.env_mu), 15.0), free=(0, 2))
    assert cls.kind in {"CSS", "branching point", "Garden of Eden", "repellor"}
    assert cls.jacobian.shape == (3, 3)
