# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm.organism                                         ║
# ║  « one agent: four heritable traits and a running energy »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Heritable: E_f, E_b (neural investment), mu, sigma (niche).     ║
# ║  Derived at birth: fidelities, per-type item fidelity through    ║
# ║  the matched filter, encounter distribution, architecture        ║
# ║  indices.  State: age, energy, this year's harvest.              ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from brainevo.abm.cost_laws import Architecture
from brainevo.abm.environment import Environment
from brainevo.constants import NAN, FloatArray


class Organism:
    """One agent of the FEBE model."""

    __slots__ = (
        "environment", "architecture", "mu", "sigma", "Ef", "Eb",
        "Ec", "Ff", "Fb", "fidelity_floor", "Fidelity", "a_norm", "a_raw", "a_F",
        "search_probs", "front_gain", "item_fidelity", "tuning_efficiency",
        "expected_fidelity", "basal_metabolism", "items_consumed",
        "fidelity_harvest", "fidelity_realised", "age", "energy",
    )

    def __init__(
        self,
        environment: Environment,
        architecture: Architecture,
        Ef: float,
        Eb: float,
        mu: float,
        sigma: float,
        Ec: float = 0.0,
        fidelity_floor: float = 1.0,
        basal_metabolism: float = 8.0,
        age: int = 0,
        initial_energy: float = 10.0,
    ) -> None:
        self.environment = environment
        self.architecture = architecture

        # heritable traits
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.Ef = float(Ef)
        self.Eb = float(Eb)
        self.Ec = float(Ec)  # capsule investment; stays 0 for laws without a capsule

        # fidelities (thesis eqs 3-7)
        self.Fb = float(architecture.b_capability(self.Eb, self.Ec))
        self.Ff = float(architecture.f_capability(self.Ef))
        self.fidelity_floor = float(fidelity_floor)
        self.Fidelity = self.fidelity_floor + self.Fb + self.Ff

        self.a_norm = self.architecture_index_normalised()
        self.a_raw = self.architecture_index_raw()
        self.a_F = self.architecture_index_fidelity()

        # niche: encounter distribution over resource types
        self.search_probs: FloatArray | None = environment.preference_distribution(self.mu, self.sigma)

        # matched filter: share of F_f realised on each type; F_b pays on all
        self.front_gain: FloatArray = architecture.front_gain(environment.resource_axis, self.mu, self.sigma)
        self.item_fidelity: FloatArray = self.fidelity_floor + self.Fb + self.Ff * self.front_gain

        if self.search_probs is None:
            self.tuning_efficiency = NAN
            self.expected_fidelity = NAN
        else:
            self.tuning_efficiency = float(self.search_probs @ self.front_gain)
            self.expected_fidelity = self.fidelity_floor + self.Fb + self.Ff * self.tuning_efficiency

        # state
        self.basal_metabolism = float(basal_metabolism)
        self.items_consumed = 0
        self.fidelity_harvest = 0.0
        self.fidelity_realised = NAN
        self.age = int(age)
        self.energy = float(initial_energy)

    def __repr__(self) -> str:
        return (f"Organism(Ef={self.Ef:.3f}, Eb={self.Eb:.3f}, mu={self.mu:.1f}, "
                f"sigma={self.sigma:.2f}, age={self.age}, energy={self.energy:.1f})")

    # ┌────────────────────────────────────────────────────────────┐
    # │ Architecture indices  « thesis eqs 12-14 »                 │
    # └────────────────────────────────────────────────────────────┘

    def architecture_index_normalised(self, eps: float = 1e-12) -> float:
        """a_norm in [-1, 1]: each end scaled by its own saturation
        investment, back-end counted from the switch (eq 12)."""
        arch = self.architecture
        Ef = self.Ef / arch.E_f_saturation
        Eb = max(self.Eb - arch.e_switch, 0.0) / arch.E_b_climb
        total = Ef + Eb
        return (Eb - Ef) / total if total > eps else NAN

    def architecture_index_raw(self, eps: float = 1e-12) -> float:
        """Raw spending contrast (E_b - E_f) / (E_b + E_f)."""
        total = self.Ef + self.Eb
        return (self.Eb - self.Ef) / total if total > eps else NAN

    def architecture_index_fidelity(self, eps: float = 1e-12) -> float:
        """Fidelity contrast with each end scaled by its ceiling."""
        arch = self.architecture
        Ff = self.Ff / arch.Fmax_f
        Fb = self.Fb / arch.Fmax_b
        total = Ff + Fb
        return (Fb - Ff) / total if total > eps else NAN

    @property
    def above_switch(self) -> bool:
        return self.Eb > self.architecture.e_switch
