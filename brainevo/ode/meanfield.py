# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — ode.meanfield                                        ║
# ║  « invasion fitness collapses to fidelity per unit cost »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Resident x = (E_f, E_b) at demographic equilibrium fixes the    ║
# ║  hit probability p*(x) so that per-capita surplus equals the     ║
# ║  reproductive drain c:                                           ║
# ║                                                                  ║
# ║      R·e·p*(x)·F(x) = b + c + E_f + E_b  =:  C(x)                ║
# ║                                                                  ║
# ║  A mutant x' gains R·e·p*(x)·F(x') - C(x'), so it invades iff    ║
# ║  F(x')/C(x') > F(x)/C(x).  Q = F/C is an optimisation principle  ║
# ║  (Metz, Mylius & Diekmann 2008): a one-dimensional, monotone     ║
# ║  environmental feedback.  Singular strategies satisfy            ║
# ║      F_f'(E_f) = Q(x)   and   F_b'(E_b) = Q(x)   (or E_b = 0).   ║
# ║  The hard switch makes E_b = 0 a boundary attractor for every    ║
# ║  parameter value; a smooth switch of width w removes it at a     ║
# ║  fold, which is where the genuine bifurcation lives.             ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from brainevo.abm.cost_laws import Architecture, CostLawParams
from brainevo.abm.energetics import ATPArchitecture
from brainevo.constants import FloatArray

DEFAULT_BASAL: float = 8.0
"""Basal metabolic cost b (thesis Table 7)."""
DEFAULT_REPRO_DRAIN: float = 1.35
"""Per-capita yearly surplus that demography absorbs at equilibrium
(reproduction minus the energy injected at birth plus the energy carried
off by the dead).  Measured from the ABM as items * F - b - E_f - E_b over
six seeded runs at P = 2: 1.18-1.59, mean 1.35."""


@dataclass(frozen=True)
class SingularStrategy:
    """A local maximum of Q, i.e. a convergence-stable singular point."""

    Ef: float
    Eb: float
    Q: float
    branch: str
    """``"front-only"`` (E_b = 0) or ``"mixed"`` (functional back-end)."""

    @property
    def has_backend(self) -> bool:
        return self.branch == "mixed"


class MeanField:
    """Fitness-per-cost landscape of the FEBE model.

    Args:
        arch:         The cost laws.
        basal:        Basal metabolic cost b.
        repro_drain:  Reproductive drain c (energy per capita per year).

    The switch shape (hard kink, softplus ramp of width w, or Hill gate of
    steepness n) is part of ``arch`` (:class:`CostLawParams`).
    """

    def __init__(self, arch: Architecture | ATPArchitecture, basal: float = DEFAULT_BASAL,
                 repro_drain: float = DEFAULT_REPRO_DRAIN) -> None:
        self.arch = arch
        self.basal = float(basal)
        self.repro_drain = float(repro_drain)

    @classmethod
    def from_params(cls, params: CostLawParams, **kwargs: float) -> MeanField:
        return cls(Architecture(params), **kwargs)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Fidelity, cost, Q                                          │
    # └────────────────────────────────────────────────────────────┘

    def back_fidelity(self, Eb: FloatArray | float) -> FloatArray | float:
        """F_b with, for capsule laws, the capsule a well-adapted organism
        pairs with E_b (the smallest that houses its neurons)."""
        if getattr(self.arch, "has_capsule", False):
            return self.arch.b_capability(Eb, self.arch.optimal_capsule(Eb))  # type: ignore[union-attr]
        return self.arch.b_capability(Eb)

    def capsule_cost(self, Eb: FloatArray | float) -> FloatArray | float:
        if getattr(self.arch, "has_capsule", False):
            return self.arch.optimal_capsule(Eb)  # type: ignore[union-attr]
        return 0.0

    def fidelity(self, Ef: FloatArray | float, Eb: FloatArray | float) -> FloatArray | float:
        return 1.0 + self.arch.f_capability(Ef) + self.back_fidelity(Eb)

    def cost(self, Ef: FloatArray | float, Eb: FloatArray | float) -> FloatArray | float:
        """C = b + c + E_f + E_b (+ the matching capsule for capsule laws)."""
        return (self.basal + self.repro_drain + np.asarray(Ef, dtype=float) + np.asarray(Eb, dtype=float)
                + np.asarray(self.capsule_cost(Eb), dtype=float))

    def Q(self, Ef: FloatArray | float, Eb: FloatArray | float) -> FloatArray | float:
        """Fidelity per unit cost: the quantity evolution maximises."""
        return self.fidelity(Ef, Eb) / self.cost(Ef, Eb)

    def invasion_fitness(self, mutant: tuple[float, float], resident: tuple[float, float]) -> float:
        """Per-capita yearly surplus of a rare mutant in the resident's
        environment, s(x'|x) = C(x)·F(x')/F(x) - C(x')."""
        Fr = float(self.fidelity(*resident))
        Cr = float(self.cost(*resident))
        return Cr * float(self.fidelity(*mutant)) / Fr - float(self.cost(*mutant))

    def selection_gradient(self, Ef: float, Eb: float, h: float = 1e-6) -> tuple[float, float]:
        """∂s/∂x' at x' = x, which equals C(x)·∇log Q(x)."""
        q0 = np.log(float(self.Q(Ef, Eb)))
        gEf = (np.log(float(self.Q(Ef + h, Eb))) - q0) / h
        gEb = (np.log(float(self.Q(Ef, Eb + h))) - q0) / h
        C = float(self.cost(Ef, Eb))
        return C * gEf, C * gEb

    # ┌────────────────────────────────────────────────────────────┐
    # │ Singular strategies                                        │
    # └────────────────────────────────────────────────────────────┘

    def best_front_only(self) -> SingularStrategy:
        """Maximise Q along E_b = 0."""
        res = minimize_scalar(lambda Ef: -float(self.Q(Ef, 0.0)), bounds=(0.0, 20.0), method="bounded")
        return SingularStrategy(float(res.x), 0.0, -float(res.fun), "front-only")

    def best_mixed(self) -> SingularStrategy | None:
        """Maximise Q with a functional back-end; ``None`` if every start
        collapses onto E_b = 0."""
        a = self.arch
        upper = a.e_switch + 10.0 * a.E_b_climb
        Ef0 = self.best_front_only().Ef
        starts = [(Ef0, a.e_switch + f * a.E_b_climb) for f in (0.05, 0.3, 0.8, 1.5)]
        if not a.is_hard_switch:
            starts += [(Ef0, 0.25 * a.e_switch + 0.05), (Ef0, 0.5 * a.e_switch + 0.05)]
        best: SingularStrategy | None = None
        for x0 in starts:
            res = minimize(lambda x: -float(self.Q(x[0], x[1])), x0=np.array(x0),
                           bounds=[(0.0, 20.0), (0.0, upper)], method="L-BFGS-B")
            Eb = float(res.x[1])
            if Eb <= 1e-6 or Eb >= upper - 1e-6:
                continue  # on a bound: not an interior maximum
            if a.is_hard_switch and Eb <= a.e_switch + 1e-9:
                continue  # inside the dead zone of a hard switch
            cand = SingularStrategy(float(res.x[0]), Eb, -float(res.fun), "mixed")
            if best is None or cand.Q > best.Q:
                best = cand
        return best

    def singular_strategies(self) -> list[SingularStrategy]:
        """All convergence-stable singular strategies (local maxima of Q)."""
        out: list[SingularStrategy] = []
        if self._front_only_is_attractor():
            out.append(self.best_front_only())
        mixed = self.best_mixed()
        if mixed is not None:
            out.append(mixed)
        if not out:  # smooth switch, single interior optimum very close to 0
            out.append(self.best_front_only())
        return out

    def global_optimum(self) -> SingularStrategy:
        return max(self.singular_strategies(), key=lambda s: s.Q)

    def _front_only_is_attractor(self) -> bool:
        """E_b = 0 attracts iff ∂Q/∂E_b < 0 there (F_b'(0) < Q)."""
        Ef = self.best_front_only().Ef
        _, gEb = self.selection_gradient(Ef, 0.0)
        return gEb < 0.0

    # ┌────────────────────────────────────────────────────────────┐
    # │ Derived quantities for the paper                           │
    # └────────────────────────────────────────────────────────────┘

    def equilibrium_hit_probability(self, s: SingularStrategy, rounds: int = 32, energy_per_item: float = 1.0) -> float:
        """p* = C/(R·e·F) at the resident equilibrium."""
        return float(self.cost(s.Ef, s.Eb)) / (rounds * energy_per_item * float(self.fidelity(s.Ef, s.Eb)))

    def valley(self, n: int = 400) -> tuple[FloatArray, FloatArray]:
        """Q along E_b at the front-only optimum's E_f: the fitness valley a
        gain mutant must cross.  Returns (E_b grid, Q)."""
        a = self.arch
        Ef = self.best_front_only().Ef
        Eb = np.linspace(0.0, a.e_switch + 2.0 * a.E_b_climb, n)
        return Eb, np.asarray(self.Q(Ef, Eb), dtype=float)

    def valley_depth(self) -> float:
        """Q(front-only) minus the minimum of Q along E_b; zero when monotone."""
        _, Q = self.valley()
        return float(Q[0] - Q.min())

    def valley_deficit(self) -> float:
        """Largest invasion-fitness deficit (energy per year) of any
        intermediate along the valley relative to the front-only resident:
        max_Eb [-s((E_f*, E_b) | front-only)].  Comparable with the energy
        effect of a single mutational step; zero when no valley exists."""
        Ef = self.best_front_only().Ef
        Eb, Q = self.valley()
        Eb = Eb[: int(np.argmax(Q)) + 1]  # only the climb up to the mixed optimum counts
        s = np.array([self.invasion_fitness((Ef, e), (Ef, 0.0)) for e in Eb])
        return float(max(-s.min(), 0.0))

    def break_even_Eb(self) -> float:
        """Smallest E_b > e_switch at which Q returns to its E_b = 0 value:
        the width of the valley a gain mutant's descendants must climb."""
        Eb, Q = self.valley()
        above = np.where((Eb > self.arch.e_switch) & (Q >= Q[0]))[0]
        return float(Eb[above[0]]) if above.size else float("inf")
