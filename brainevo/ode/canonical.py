# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — ode.canonical                                        ║
# ║  « the canonical equation and quasi-static ramps »               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Dieckmann & Law (1996): dx/dt = ½ μ σ² N*(x) ∇s(x'|x)|x'=x.     ║
# ║  With Q an optimisation principle the flow is a gradient ascent  ║
# ║  on log Q up to a positive factor, so every trajectory ends at a ║
# ║  local maximum of Q.  Ramping P (or w) forward and back traces   ║
# ║  the hysteresis loop of the evolutionary attractor.              ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from brainevo.constants import FloatArray
from brainevo.ode.meanfield import MeanField

MeanFieldFactory = Callable[[float], MeanField]


def integrate_canonical(
    mf: MeanField,
    x0: tuple[float, float],
    t_max: float = 200.0,
    rate: float = 1.0,
    n_points: int = 400,
) -> tuple[FloatArray, FloatArray]:
    """Integrate the canonical equation from ``x0``.

    The mutational variance and population size are absorbed into
    ``rate``; they rescale time, not the attractor.  Traits are clipped
    at zero (E_f, E_b >= 0), making E_b = 0 an absorbing boundary when
    the gradient points outward.

    Returns:
        (t, X) with X of shape (n_points, 2) holding (E_f, E_b).
    """

    def rhs(_: float, x: FloatArray) -> FloatArray:
        Ef, Eb = max(x[0], 0.0), max(x[1], 0.0)
        gEf, gEb = mf.selection_gradient(Ef, Eb)
        if Eb <= 0.0 and gEb < 0.0:
            gEb = 0.0
        if Ef <= 0.0 and gEf < 0.0:
            gEf = 0.0
        return rate * np.array([gEf, gEb])

    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(rhs, (0.0, t_max), np.array(x0, dtype=float), t_eval=t_eval, method="LSODA", rtol=1e-8, atol=1e-10)
    X = np.clip(sol.y.T, 0.0, None)
    return sol.t, X


def relax(mf: MeanField, x0: tuple[float, float], t_max: float = 400.0) -> tuple[float, float]:
    """Follow the local attractor from ``x0`` until the gradient vanishes."""
    _, X = integrate_canonical(mf, x0, t_max=t_max, n_points=50)
    return float(X[-1, 0]), float(X[-1, 1])


def ramp_hysteresis(
    factory: MeanFieldFactory,
    control_values: FloatArray,
    x_start: tuple[float, float] | None = None,
) -> dict[str, FloatArray]:
    """Quasi-static ramp of one control parameter, downward then upward.

    Starts front-only at the largest control value (the thesis' naive
    founders at high overhead), ramps the control down until the
    front-only attractor vanishes at the fold, then ramps back up and
    watches whether the mixed state is retained.

    Args:
        factory:        Maps a control value to a :class:`MeanField`.
        control_values: Increasing grid of the control parameter.
        x_start:        Initial traits; defaults to the front-only
                        optimum at the largest control value.

    Returns:
        Dict with ``control``, ``Eb_down``/``Ef_down`` (descending ramp),
        ``Eb_up``/``Ef_up`` (ascending ramp) and ``Eb_global``/``Ef_global``
        (the global optimum, the Maxwell reference).
    """
    control_values = np.asarray(control_values, dtype=float)
    mf_top = factory(float(control_values[-1]))
    x = x_start if x_start is not None else (mf_top.best_front_only().Ef, 0.0)

    down: list[tuple[float, float]] = []
    for c in control_values[::-1]:
        x = relax(factory(float(c)), x)
        down.append(x)
    down = down[::-1]
    up: list[tuple[float, float]] = []
    for c in control_values:
        x = relax(factory(float(c)), x)
        up.append(x)
    glob = [factory(float(c)).global_optimum() for c in control_values]

    return {
        "control": control_values,
        "Ef_down": np.array([x[0] for x in down]),
        "Eb_down": np.array([x[1] for x in down]),
        "Ef_up": np.array([x[0] for x in up]),
        "Eb_up": np.array([x[1] for x in up]),
        "Eb_global": np.array([s.Eb for s in glob]),
        "Ef_global": np.array([s.Ef for s in glob]),
    }
