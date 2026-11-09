# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — ode.niche                                            ║
# ║  « vector feedback: where branching becomes possible »           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  With niche-based foraging the environment is the whole stock    ║
# ║  vector, so a mutant's fitness depends on the resident's         ║
# ║  encounter distribution: frequency dependence.  This module      ║
# ║  computes the resident ecological equilibrium (stocks S_z*,      ║
# ║  population N*), the invasion fitness of a mutant in that        ║
# ║  environment, the selection gradient in (E_f, E_b, log sigma),   ║
# ║  the canonical-equation flow, and the Geritz et al. (1998)       ║
# ║  classification of the singular strategy (CSS / branching /      ║
# ║  repellor) from the Jacobian and the mutant Hessian.             ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from brainevo.abm.cost_laws import Architecture
from brainevo.abm.environment import Environment, EnvironmentParams
from brainevo.abm.population import LifeHistoryParams, MutationParams
from brainevo.constants import EPS, FloatArray
from brainevo.ode.meanfield import DEFAULT_REPRO_DRAIN

N_MAX: float = 1e6
"""Upper bracket for the resident population size."""


@dataclass(frozen=True)
class Strategy:
    """Resident or mutant trait vector."""

    Ef: float
    Eb: float
    mu: float
    sigma: float

    def as_array(self) -> FloatArray:
        """Evolving coordinates: (E_f, E_b, log sigma); mu is held fixed."""
        return np.array([self.Ef, self.Eb, np.log(self.sigma)])

    def with_array(self, x: FloatArray) -> Strategy:
        return Strategy(max(float(x[0]), 0.0), max(float(x[1]), 0.0), self.mu, float(np.exp(x[2])))


@dataclass(frozen=True)
class ResidentEquilibrium:
    """Ecological equilibrium set by a monomorphic resident."""

    strategy: Strategy
    N: float
    stock: FloatArray
    success: FloatArray
    """Per-type probability h_z that an attempt on type z finds an item."""
    viable: bool
    resident_surplus: float = 0.0
    """Realised per-capita surplus of the resident at N (zero at a root)."""


@dataclass(frozen=True)
class SingularClassification:
    """Geritz et al. (1998) classification in the evolving coordinates."""

    strategy: Strategy
    gradient: FloatArray
    jacobian: FloatArray
    hessian: FloatArray
    convergence_stable: bool
    evolutionarily_stable: bool
    jacobian_eigenvalues: FloatArray = field(repr=False)
    hessian_eigenvalues: FloatArray = field(repr=False)

    @property
    def kind(self) -> str:
        if self.convergence_stable and self.evolutionarily_stable:
            return "CSS"
        if self.convergence_stable and not self.evolutionarily_stable:
            return "branching point"
        if not self.convergence_stable and self.evolutionarily_stable:
            return "Garden of Eden"
        return "repellor"


class NicheMeanField:
    """Deterministic skeleton of FEBE with niche-based foraging.

    Within a year the N residents make R attempts each, distributed over
    types by the encounter distribution pi_z(mu, sigma); attempts succeed
    while the stock is above its refuge.  Stationarity of each stock under
    harvest-then-logistic-regrowth fixes S_z*, and the population grows
    until per-capita surplus equals the demographic drain c.
    """

    def __init__(
        self,
        arch: Architecture,
        env: EnvironmentParams | None = None,
        life: LifeHistoryParams | None = None,
        mutation: MutationParams | None = None,
        repro_drain: float = DEFAULT_REPRO_DRAIN,
    ) -> None:
        self.arch = arch
        self.env = Environment(env or EnvironmentParams())
        self.life = life or LifeHistoryParams(uniform_foraging=False)
        self.mutation = mutation or MutationParams()
        self.repro_drain = float(repro_drain)
        self.axis = self.env.resource_axis.astype(float)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Strategy-level quantities                                  │
    # └────────────────────────────────────────────────────────────┘

    def encounter(self, s: Strategy) -> FloatArray:
        p = self.env.preference_distribution(s.mu, s.sigma)
        if p is None:
            raise ValueError(f"niche off-axis: {s}")
        return p

    def item_fidelity(self, s: Strategy) -> FloatArray:
        """F(z; x) = 1 + F_f g(z) + F_b for every resource type."""
        a = self.arch
        return 1.0 + float(a.f_capability(s.Ef)) * a.front_gain(self.axis, s.mu, s.sigma) + float(a.b_capability(s.Eb))

    def cost(self, s: Strategy) -> float:
        return self.life.basal_metabolism + self.repro_drain + s.Ef + s.Eb

    # ┌────────────────────────────────────────────────────────────┐
    # │ Resident ecological equilibrium                            │
    # └────────────────────────────────────────────────────────────┘

    def _stationary_stock(self, attempts: float, z: int) -> float:
        """Stock at the start of the year that harvest-then-regrowth returns to.

        A fixed yearly quota A on a logistic stock has a stable upper fixed
        point only while A is below the maximum sustainable yield rK/4;
        the ABM starts every type at K, so that branch is the one realised.
        Above it the type collapses to its refuge and yields only the
        regrowth from the refuge, r * refuge * (1 - f).  The yield is
        therefore discontinuous in A: overexploitation collapse.
        """
        K, refuge, r = float(self.env.capacity[z]), float(self.env.refuge[z]), self.env.regrowth_rate
        if attempts <= 0.0:
            return K
        discriminant = 1.0 - 4.0 * attempts / (r * K)
        if discriminant >= 0.0:
            post_harvest = 0.5 * K * (1.0 + np.sqrt(discriminant))
            return post_harvest + attempts
        return refuge + r * refuge * (1.0 - refuge / K)

    def _success(self, N: float, pi: FloatArray) -> tuple[FloatArray, FloatArray]:
        attempts = N * self.life.foraging_rounds * pi
        stock = np.array([self._stationary_stock(float(a), z) for z, a in enumerate(attempts)])
        available = np.maximum(stock - self.env.refuge, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(attempts > 0.0, np.minimum(available / np.maximum(attempts, EPS), 1.0), 1.0)
        return stock, h

    def intake(self, s: Strategy, success: FloatArray) -> float:
        """Expected yearly energy intake of strategy ``s`` in an environment
        with per-type success probabilities ``success``."""
        pi = self.encounter(s)
        return self.life.foraging_rounds * self.env.energy_per_item * float(pi @ (success * self.item_fidelity(s)))

    def resident_equilibrium(self, s: Strategy) -> ResidentEquilibrium:
        """Solve intake(N) = cost for the resident population size.

        Because per-type yield collapses discontinuously at the maximum
        sustainable yield, the surplus may jump across zero without a
        root; the bracket then closes on that jump, which is where the ABM
        population hovers.  Fitness is therefore always measured relative
        to the resident's realised surplus (see :meth:`invasion_fitness`).
        """
        pi = self.encounter(s)
        C = self.cost(s)

        def surplus(N: float) -> float:
            _, h = self._success(N, pi)
            return self.intake(s, h) - C

        if surplus(0.0) <= 0.0:
            stock, h = self._success(0.0, pi)
            return ResidentEquilibrium(s, 0.0, stock, h, viable=False, resident_surplus=surplus(0.0))
        hi = 10.0
        while surplus(hi) > 0.0 and hi < N_MAX:
            hi *= 4.0
        if hi >= N_MAX:
            raise RuntimeError(f"resident surplus never absorbed below N = {N_MAX:g}: {s}")
        N = float(brentq(surplus, 0.0, hi, xtol=1e-6))
        stock, h = self._success(N, pi)
        return ResidentEquilibrium(s, N, stock, h, viable=True, resident_surplus=self.intake(s, h) - C)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Invasion fitness and selection gradient                    │
    # └────────────────────────────────────────────────────────────┘

    def invasion_fitness(self, mutant: Strategy, resident: Strategy | ResidentEquilibrium) -> float:
        """Yearly surplus of a rare mutant in the resident's environment,
        relative to the resident's own realised surplus (zero at a true
        ecological root, slightly off at an overexploitation jump)."""
        eq = resident if isinstance(resident, ResidentEquilibrium) else self.resident_equilibrium(resident)
        return self.intake(mutant, eq.success) - self.cost(mutant) - eq.resident_surplus

    def selection_gradient(self, resident: Strategy, h: float = 1e-5) -> FloatArray:
        """d s(x'|x) / d x' at x' = x, in coordinates (E_f, E_b, log sigma).

        Central differences, except one-sided at the E_f = 0 or E_b = 0
        boundary so the hard switch's dead zone is respected.
        """
        eq = self.resident_equilibrium(resident)
        x0 = resident.as_array()
        grad = np.zeros(3)
        for i in range(3):
            step = np.zeros(3)
            step[i] = h
            if i < 2 and x0[i] < h:
                grad[i] = (self.invasion_fitness(resident.with_array(x0 + step), eq)
                           - self.invasion_fitness(resident, eq)) / h
            else:
                grad[i] = (self.invasion_fitness(resident.with_array(x0 + step), eq)
                           - self.invasion_fitness(resident.with_array(x0 - step), eq)) / (2 * h)
        return grad

    def mutant_hessian(self, resident: Strategy, h: float = 1e-3) -> FloatArray:
        """Second derivatives of s(x'|x) in the mutant at x' = x."""
        eq = self.resident_equilibrium(resident)
        x0 = resident.as_array()
        f = lambda d: self.invasion_fitness(resident.with_array(x0 + d), eq)  # noqa: E731
        H = np.zeros((3, 3))
        f0 = f(np.zeros(3))
        for i in range(3):
            ei = np.eye(3)[i] * h
            H[i, i] = (f(ei) - 2 * f0 + f(-ei)) / h**2
            for j in range(i + 1, 3):
                ej = np.eye(3)[j] * h
                H[i, j] = H[j, i] = (f(ei + ej) - f(ei - ej) - f(-ei + ej) + f(-ei - ej)) / (4 * h**2)
        return H

    # ┌────────────────────────────────────────────────────────────┐
    # │ Canonical equation and classification                      │
    # └────────────────────────────────────────────────────────────┘

    def mutational_rates(self) -> FloatArray:
        """Per-trait factor mu_i sigma_i^2 of the canonical equation.

        E_f and E_b mutate additively (sd ``E_mutation_sd``); sigma mutates
        log-normally, so its coordinate is log sigma with sd ``sigma_mutation_sd``.
        """
        m = self.mutation
        return np.array([m.arch_mutation_rate * m.E_mutation_sd**2,
                         m.arch_mutation_rate * m.E_mutation_sd**2,
                         m.niche_mutation_rate * m.sigma_mutation_sd**2])

    def integrate_canonical(
        self, x0: Strategy, t_max: float = 50.0, n_points: int = 200, rate: float = 1.0,
    ) -> tuple[FloatArray, list[Strategy], FloatArray]:
        """dx/dt = ½ mu sigma² N*(x) grad s.  Returns (t, strategies, N*)."""
        rates = self.mutational_rates()

        def rhs(_: float, x: FloatArray) -> FloatArray:
            s = x0.with_array(x)
            eq = self.resident_equilibrium(s)
            if not eq.viable:
                return np.zeros(3)
            g = self.selection_gradient(s)
            if s.Ef <= 0.0 and g[0] < 0.0:
                g[0] = 0.0
            if s.Eb <= 0.0 and g[1] < 0.0:
                g[1] = 0.0
            return rate * 0.5 * rates * eq.N * g

        t_eval = np.linspace(0.0, t_max, n_points)
        sol = solve_ivp(rhs, (0.0, t_max), x0.as_array(), t_eval=t_eval, method="RK45", rtol=1e-5, atol=1e-7)
        strategies = [x0.with_array(col) for col in sol.y.T]
        N = np.array([self.resident_equilibrium(s).N for s in strategies])
        return sol.t, strategies, N

    def classify(self, s: Strategy, free: tuple[int, ...] = (0, 1, 2)) -> SingularClassification:
        """Classify ``s`` as a singular strategy in the ``free`` coordinates.

        Coordinates pinned at zero by a hard switch (E_b = 0 with negative
        gradient) should be excluded from ``free``; the default treats all
        three as evolving.
        """
        idx = list(free)
        g = self.selection_gradient(s)
        x0 = s.as_array()
        h = 1e-3
        J = np.zeros((3, 3))
        for j in range(3):
            step = np.zeros(3)
            step[j] = h
            J[:, j] = (self.selection_gradient(s.with_array(x0 + step))
                       - self.selection_gradient(s.with_array(x0 - step))) / (2 * h)
        H = self.mutant_hessian(s)
        Jf, Hf = J[np.ix_(idx, idx)], H[np.ix_(idx, idx)]
        jev = np.linalg.eigvals(Jf).real
        hev = np.linalg.eigvalsh(Hf)
        return SingularClassification(
            s, g, J, H,
            convergence_stable=bool(np.all(jev < 0.0)),
            evolutionarily_stable=bool(np.all(hev < 0.0)),
            jacobian_eigenvalues=jev, hessian_eigenvalues=hev,
        )

    # ┌────────────────────────────────────────────────────────────┐
    # │ Pairwise invasibility plot in niche breadth                │
    # └────────────────────────────────────────────────────────────┘

    def pip_sigma(
        self, template: Strategy, sigmas: FloatArray, architecture_of: str = "fixed",
    ) -> np.ndarray:
        """Sign of s(sigma' | sigma) over a grid, at fixed (E_f, E_b).

        ``architecture_of="fixed"`` keeps the template's investments for
        resident and mutant; the PIP then isolates niche-breadth selection
        in the environment the resident shapes.
        """
        sigmas = np.asarray(sigmas, dtype=float)
        S = np.zeros((len(sigmas), len(sigmas)))
        for j, sr in enumerate(sigmas):
            resident = Strategy(template.Ef, template.Eb, template.mu, float(sr))
            eq = self.resident_equilibrium(resident)
            for i, sm in enumerate(sigmas):
                S[i, j] = self.invasion_fitness(Strategy(template.Ef, template.Eb, template.mu, float(sm)), eq)
        return S
