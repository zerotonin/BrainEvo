from __future__ import annotations

import numpy as np
import pytest

from brainevo.abm import Architecture, CostLawParams
from brainevo.ode import MeanField, integrate_canonical, ramp_hysteresis


def _mf(P: float = 2.0, w: float = 0.0, Fmax: float = 1.2, M: float = 2.5, **kw: object) -> MeanField:
    return MeanField(Architecture(CostLawParams(Fmax_f=Fmax, M=M, P=P, back_rate=0.8, switch_width=w, **kw)))  # type: ignore[arg-type]


def test_hard_switch_has_two_attractors() -> None:
    strategies = _mf().singular_strategies()
    branches = {s.branch for s in strategies}
    assert branches == {"front-only", "mixed"}


def test_singular_condition_marginal_equals_average() -> None:
    """At a singular strategy F'(E) = Q at each invested end."""
    mf = _mf()
    s = mf.best_mixed()
    assert s is not None
    h = 1e-5
    Ff_prime = (mf.arch.f_capability(s.Ef + h) - mf.arch.f_capability(s.Ef - h)) / (2 * h)
    Fb_prime = (mf.back_fidelity(s.Eb + h) - mf.back_fidelity(s.Eb - h)) / (2 * h)
    assert Ff_prime == pytest.approx(s.Q, rel=1e-3)
    assert Fb_prime == pytest.approx(s.Q, rel=1e-3)


def test_invasion_fitness_sign_follows_Q() -> None:
    mf = _mf()
    front, mixed = mf.best_front_only(), mf.best_mixed()
    assert mixed is not None
    s = mf.invasion_fitness((mixed.Ef, mixed.Eb), (front.Ef, front.Eb))
    assert (s > 0) == (mixed.Q > front.Q)
    assert mf.invasion_fitness((front.Ef, front.Eb), (front.Ef, front.Eb)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "Fmax,M,Eb_abm,Ef_abm",
    [(1.2, 1.0, 2.585, 0.504), (1.2, 2.5, 2.859, 0.528), (2.7, 1.0, 2.653, 0.600),
     (0.5, 2.5, 2.669, 0.478), (0.5, 4.5, 2.823, 0.372)],
)
def test_mixed_optimum_matches_seeded_abm_runs(Fmax: float, M: float, Eb_abm: float, Ef_abm: float) -> None:
    """ABM populations seeded above the switch (P = 2, back_rate = 0.8,
    2000 years, tail-200 means) settle where the mean field says.

    E_b agrees to a few percent.  E_f is systematically ~10-20 % below the
    prediction because Q is nearly flat in E_f once a back-end exists, so
    E_f is still creeping upward at 2000 years (compare thesis Fig 9C)."""
    s = _mf(Fmax=Fmax, M=M).best_mixed()
    assert s is not None
    assert s.Eb == pytest.approx(Eb_abm, rel=0.08)
    assert s.Ef == pytest.approx(Ef_abm, rel=0.30)
    assert s.Ef >= Ef_abm * 0.95


def test_smooth_switch_removes_front_only_attractor_for_wide_w() -> None:
    assert len(_mf(P=0.5, w=0.0).singular_strategies()) == 2
    assert len(_mf(P=0.5, w=2.0).singular_strategies()) == 1


def test_canonical_flow_stays_on_local_branch() -> None:
    mf = _mf()
    _, X = integrate_canonical(mf, (0.1, 0.0), t_max=300.0)
    assert X[-1, 1] == pytest.approx(0.0, abs=1e-9)
    mixed = mf.best_mixed()
    assert mixed is not None
    _, X = integrate_canonical(mf, (0.1, mf.arch.e_switch + 0.5), t_max=600.0)
    assert X[-1, 1] == pytest.approx(mixed.Eb, rel=0.05)


def test_hysteresis_loop_with_smooth_switch() -> None:
    loop = ramp_hysteresis(lambda P: _mf(P=P, w=0.3), np.linspace(0.0, 4.0, 17))
    # front-only persists on the way down until the fold (~P = 1.4), then jumps to mixed
    assert loop["Eb_down"][-1] == pytest.approx(0.0, abs=1e-6)
    assert loop["Eb_down"][0] > 0.5
    # once mixed, the way up stays mixed: the loop is open
    assert np.all(loop["Eb_up"] > 0.5)
    assert np.any(loop["Eb_down"] < loop["Eb_up"] - 0.5)


def test_hill_gate_keeps_front_only_attractor_at_any_steepness() -> None:
    """A saturation gate has zero slope at E_b = 0, so the front-only state
    stays locally stable however gentle the flank; only the valley deficit shrinks."""
    deficits = []
    for n in (0.5, 2.0, 8.0, 64.0):
        mf = _mf(P=2.0, switch_shape="hill", switch_steepness=n)
        assert {s.branch for s in mf.singular_strategies()} == {"front-only", "mixed"}
        deficits.append(mf.valley_deficit())
    assert deficits == sorted(deficits)


def test_ramp_width_zero_is_thesis_switch(arch: Architecture) -> None:
    smooth = Architecture(CostLawParams(Fmax_f=0.8, M=3.0, P=3.75, back_rate=0.8, switch_width=0.0))
    E = np.linspace(0.0, 8.0, 200)
    assert np.array_equal(np.asarray(arch.b_capability(E)), np.asarray(smooth.b_capability(E)))
    assert arch.is_hard_switch
