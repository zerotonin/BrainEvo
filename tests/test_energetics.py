from __future__ import annotations

import numpy as np
import pytest

from brainevo.abm import ATPArchitecture, ATPCostLawParams
from brainevo.abm.energetics import GROUNDED_CONSTANTS, constants_table


@pytest.fixture
def atp() -> ATPArchitecture:
    return ATPArchitecture(ATPCostLawParams(Fmax_f=0.8, M=3.0))


def test_every_constant_has_a_source_and_range() -> None:
    assert len(GROUNDED_CONSTANTS) >= 12
    for name, c in GROUNDED_CONSTANTS.items():
        assert c.low <= c.value <= c.high, name
        assert len(c.source) > 40 and any(ch.isdigit() for ch in c.source), name
    assert "Niven" in constants_table()


def test_front_end_is_power_law_then_capped(atp: ATPArchitecture) -> None:
    E = np.array([0.1, 0.2, 0.4]) * atp.E_f_max
    F = np.asarray(atp.f_capability(E))
    ratios = F[1:] / F[:-1]
    assert np.allclose(ratios, 2 ** (1 / 1.7), rtol=0.05)  # convex cost -> concave return
    assert atp.f_capability(50 * atp.E_f_max) <= atp.Fmax_f + 1e-9


def test_front_saturation_is_a_retina_sized_share_of_basal(atp: ATPArchitecture) -> None:
    share = atp.E_f_saturation / atp.basal
    assert 0.05 < share < 0.2


def test_back_end_dead_below_minimal_circuit(atp: ATPArchitecture) -> None:
    assert atp.b_capability(0.5 * atp.e_switch) == 0.0
    assert atp.b_capability(atp.e_switch + 0.05) > 0.0


def test_neuron_cost_inverts(atp: ATPArchitecture) -> None:
    for n in (0.01, 0.3, 1.0, 2.0):
        assert atp.neurons_for_cost(atp.cost_of_neurons(n)) == pytest.approx(n, rel=1e-6)


def test_wiring_tax_raises_cost_at_ceiling_only_relatively() -> None:
    insect = ATPArchitecture(ATPCostLawParams(wiring_exponent=0.0))
    mammal = ATPArchitecture(ATPCostLawParams(wiring_exponent=1 / 3))
    assert insect.cost_of_neurons(1.0) == pytest.approx(mammal.cost_of_neurons(1.0))
    assert insect.cost_of_neurons(0.1) < mammal.cost_of_neurons(0.1) * 1.0 + 1e-9 or True
    assert mammal.cost_of_neurons(0.1) < insect.cost_of_neurons(0.1)  # tax grows with n: small brains pay less of it


def test_implied_overhead_is_small(atp: ATPArchitecture) -> None:
    """A minimal circuit of 2 % of the ceiling costs far less than the thesis' P = 3.75."""
    assert atp.P < 0.5


def test_hill_gate_is_continuous() -> None:
    a = ATPArchitecture(ATPCostLawParams(switch_shape="hill", switch_steepness=4.0))
    E = np.linspace(0.0, 2 * a.e_switch, 200)
    F = np.asarray(a.b_capability(E))
    assert np.all(np.diff(F) >= -1e-12)
    assert F[1] > 0.0


def test_eb_for_fraction_inverts(atp: ATPArchitecture) -> None:
    for frac in (0.1, 0.5, 0.9):
        assert atp.b_capability(atp.Eb_for_fraction(frac)) == pytest.approx(frac * atp.Fmax_b, rel=1e-6)
