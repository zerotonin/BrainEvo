from __future__ import annotations

import numpy as np
import pytest

from brainevo.abm import Architecture, CostLawParams


def test_front_end_is_monotone_concave_and_capped(arch: Architecture) -> None:
    E = np.linspace(0.0, 10.0, 500)
    F = np.asarray(arch.f_capability(E))
    assert np.all(np.diff(F) >= 0)
    assert F.max() <= arch.Fmax_f + 1e-9
    assert F[0] == 0.0


def test_back_end_returns_nothing_below_switch(arch: Architecture) -> None:
    E = np.linspace(0.0, arch.e_switch, 50)
    assert np.all(np.asarray(arch.b_capability(E)) == 0.0)
    assert arch.b_capability(arch.e_switch + 0.1) > 0.0


def test_saturation_energies_hit_q_of_ceiling(arch: Architecture) -> None:
    assert arch.f_capability(arch.E_f_saturation) == pytest.approx(arch.q * arch.Fmax_f)
    assert arch.b_capability(arch.E_b_saturation) == pytest.approx(arch.q * arch.Fmax_b)


def test_thesis_calibration_numbers() -> None:
    """k = 3, SNR = 14.44 give E_f,sat ~ 0.92 (thesis §2.10.1)."""
    a = Architecture(CostLawParams(Fmax_f=1.0, M=1.0, P=0.0))
    assert a.E_f_saturation == pytest.approx(0.92, abs=0.01)


def test_fmax_does_not_move_saturation_energy() -> None:
    sats = [Architecture(CostLawParams(Fmax_f=f, M=1.0, P=0.0)).E_f_saturation for f in (0.1, 1.0, 4.0)]
    assert np.allclose(sats, sats[0])


def test_ceiling_scaled_regime_keeps_back_climb_constant() -> None:
    climbs = [Architecture(CostLawParams(M=M, back_rate=1.0)).E_b_climb for M in (1.0, 2.5, 5.0)]
    assert np.allclose(climbs, climbs[0])


def test_rate_equal_regime_raises_back_climb_with_M() -> None:
    climbs = [Architecture(CostLawParams(M=M, back_rate=1.0 / M)).E_b_climb for M in (1.0, 2.5, 5.0)]
    assert np.all(np.diff(climbs) > 0)


def test_eb_for_fraction_inverts_capability(arch: Architecture) -> None:
    for frac in (0.1, 0.5, 0.9):
        Eb = arch.Eb_for_fraction(frac)
        assert arch.b_capability(Eb) == pytest.approx(frac * arch.Fmax_b, rel=1e-6)


def test_front_gain_untuned_is_flat(arch: Architecture) -> None:
    g = arch.front_gain(np.arange(10), mu=5.0, sigma=2.0)
    assert np.all(g == 1.0)


def test_front_gain_tuned_peaks_at_mu(tuned_arch: Architecture) -> None:
    axis = np.arange(100)
    g = tuned_arch.front_gain(axis, mu=40.0, sigma=15.0)
    assert g.argmax() == 40
    assert g[40] == pytest.approx(1.0)
    assert g[0] < 1e-6


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        CostLawParams(back_rate=0.0)
    with pytest.raises(ValueError):
        CostLawParams(M=0.5)
    with pytest.raises(ValueError):
        CostLawParams(tuning_mode="sideways")  # type: ignore[arg-type]
