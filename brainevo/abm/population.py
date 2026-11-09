# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm.population                                       ║
# ║  « forage, reproduce, cull, regrow: one year of evolution »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Life cycle of Dodd (2026) §2.6-2.9.  One year = R foraging      ║
# ║  rounds, each a shuffled roll per organism against the standing  ║
# ║  stock; energy settles once a year; reproduction is graded by    ║
# ║  surplus with uniparental inheritance; small-effect mutations    ║
# ║  plus rare large-effect gain/loss of the back-end.               ║
# ║                                                                  ║
# ║  The random-number call order is identical to the thesis code,   ║
# ║  so a shared seed reproduces the original runs exactly.          ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from brainevo.abm.cost_laws import Architecture
from brainevo.abm.environment import Environment
from brainevo.abm.organism import Organism
from brainevo.abm.recording import Recorder
from brainevo.constants import NAN, History

NicheMode = Literal["fixed", "environment", "random"]
ArchMode = Literal["naive", "random", "above_switch", "fixed_Eb", "split"]
ProgressCallback = Callable[["Population", int, int], None]


@dataclass(frozen=True)
class LifeHistoryParams:
    """Energetics, reproduction and mortality (thesis Table 7)."""

    initial_energy: float = 10.0
    offspring_energy: float = 5.0
    repro_scale: float = 60.0
    repro_cost: float = 8.0
    basal_metabolism: float = 8.0
    max_age: int = 20
    foraging_rounds: int = 32
    uniform_foraging: bool = True
    sigma_min: float = 1.5
    sigma_max: float = 60.0
    runaway_guard: int = 100_000


@dataclass(frozen=True)
class MutationParams:
    """Small-effect and large-effect mutation (thesis Table 8)."""

    arch_mutation_rate: float = 0.1
    E_mutation_sd: float = 0.03
    niche_mutation_rate: float = 0.1
    mu_mutation_sd: float = 0.5
    sigma_mutation_sd: float = 0.1
    duplication_rate: float = 0.001
    loss_rate: float = 0.001
    jump_frac: float = 0.01
    jump_sigma: float = 0.5


@dataclass(frozen=True)
class InitialisationParams:
    """Founder population.

    Attributes:
        n_founders:       Number of founders.
        niche_mode:       ``"fixed"`` (mu at the axis centre, sigma =
                          ``sigma_init``), ``"environment"`` (mu, sigma of
                          the carrying-capacity curve) or ``"random"``.
        sigma_init:       Founder niche breadth for ``"fixed"``.
        arch_mode:        ``"naive"`` (E_f = E_b = 0), ``"random"``,
                          ``"above_switch"`` (minimal functional back-end),
                          ``"fixed_Eb"`` (E_f = 0, E_b = ``initial_Eb``) or
                          ``"split"`` (a fraction ``backend_fraction`` of
                          founders carry a minimal back-end).
        initial_Eb:       Founder back-end investment for ``"fixed_Eb"``.
        backend_fraction: Fraction with a back-end for ``"split"``.
        initial_Ec:       Founder capsule investment (capsule laws only);
                          ``None`` gives each founder the capsule its
                          back-end needs.
    """

    n_founders: int = 500
    niche_mode: NicheMode = "fixed"
    sigma_init: float = 15.0
    arch_mode: ArchMode = "naive"
    initial_Eb: float = 0.0
    backend_fraction: float = 0.5
    initial_Ec: float | None = None


class Population:
    """Evolving population of :class:`Organism` on an :class:`Environment`."""

    def __init__(
        self,
        environment: Environment,
        architecture: Architecture,
        life: LifeHistoryParams | None = None,
        mutation: MutationParams | None = None,
        init: InitialisationParams | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        self.environment = environment
        self.architecture = architecture
        self.life = life or LifeHistoryParams()
        self.mutation = mutation or MutationParams()
        self.init = init or InitialisationParams()
        self.rng = np.random.default_rng(rng)

        self.e_switch = architecture.e_switch
        self.has_capsule = bool(getattr(architecture, "has_capsule", False))
        self.sim_time = 0
        self.extinct = False
        self.runaway = False
        self.jumps_this_year = 0
        self.losses_this_year = 0
        self._reset_stashes()

        self.recorder = Recorder(self.e_switch, environment.env_mu)
        self.population: list[Organism] = []
        self._initialise_population()
        self._record()

    def __repr__(self) -> str:
        return f"Population(N={len(self.population)}, t={self.sim_time}, {self.architecture!r})"

    # ┌────────────────────────────────────────────────────────────┐
    # │ Setup                                                      │
    # └────────────────────────────────────────────────────────────┘

    def _reset_stashes(self) -> None:
        self._items_per_organism = np.empty(0)
        self._items_taken = NAN
        self._energy_pre_repro = NAN
        self._reproductive_frac = NAN

    def _initialise_population(self) -> None:
        env, arch, ini, rng = self.environment, self.architecture, self.init, self.rng
        for _ in range(int(ini.n_founders)):
            if ini.niche_mode == "fixed":
                mu, sigma = float(env.env_mu), float(ini.sigma_init)
            elif ini.niche_mode == "environment":
                mu, sigma = float(env.env_mu), env.env_sigma
            else:
                mu = rng.uniform(0.0, env.n_resources - 1.0)
                sigma = rng.uniform(2.0, 10.0)

            if ini.arch_mode == "fixed_Eb":
                Ef, Eb = 0.0, float(ini.initial_Eb)
            elif ini.arch_mode == "split":
                Ef = 0.0
                Eb = arch.e_switch + 0.2 * arch.E_b_climb if rng.random() < ini.backend_fraction else 0.0
            elif ini.arch_mode == "above_switch":
                Ef = 0.0
                Eb = arch.e_switch + (0.2 * arch.E_b_climb) * np.exp(rng.normal(0.0, 0.05))
            elif ini.arch_mode == "naive":
                Ef, Eb = 0.0, 0.0
            else:
                Ef = rng.uniform(0.0, 1.2 * arch.E_f_saturation)
                Eb = rng.uniform(0.0, 1.2 * arch.E_b_saturation)

            age = int(rng.integers(0, self.life.max_age))
            Ec = self._founder_capsule(Eb)
            self.population.append(self._birth(mu, sigma, Ef, Eb, Ec, age=age))

    def _founder_capsule(self, Eb: float) -> float:
        if not self.has_capsule:
            return 0.0
        if self.init.initial_Ec is not None:
            return float(self.init.initial_Ec)
        return float(self.architecture.optimal_capsule(Eb))  # type: ignore[union-attr]

    def _birth(self, mu: float, sigma: float, Ef: float, Eb: float, Ec: float = 0.0,
               age: int = 0, energy: float | None = None) -> Organism:
        mu = float(np.clip(mu, 0.0, self.environment.n_resources - 1.0))
        sigma = float(np.clip(sigma, self.life.sigma_min, self.life.sigma_max))
        return Organism(
            self.environment, self.architecture, Ef=Ef, Eb=Eb, mu=mu, sigma=sigma, Ec=Ec, age=age,
            basal_metabolism=self.life.basal_metabolism,
            initial_energy=self.life.initial_energy if energy is None else energy,
        )

    # ┌────────────────────────────────────────────────────────────┐
    # │ Run loop                                                   │
    # └────────────────────────────────────────────────────────────┘

    def run(
        self,
        years: int,
        snapshot_times: set[int] | None = None,
        progress_every: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[History, list[dict[str, float | int | bool]]]:
        """Simulate ``years`` of evolution.

        Returns:
            The yearly history and the individual-level snapshot rows.
        """
        snapshot_times = set() if snapshot_times is None else set(snapshot_times)
        for _ in range(int(years)):
            if not self.step_year():
                break
            if self.sim_time in snapshot_times:
                self.recorder.snapshot(self.population, self.sim_time)
            if progress_every and on_progress and self.sim_time % progress_every == 0:
                on_progress(self, self.sim_time, int(years))
        return self.history(), self.recorder.snapshots

    def step_year(self) -> bool:
        """Fluctuate, forage, reproduce, cull, regrow, record. False if finished."""
        if not self.population:
            self.extinct = True
            return False
        self.environment.fluctuate(self.rng)
        self.forage_year()
        self.jumps_this_year = 0
        self.losses_this_year = 0
        births = self.reproduce()
        starved, senescent = self.cull()
        self.environment.regrow()
        self.sim_time += 1
        self._record(births, starved, senescent)
        return self._check_status()

    def history(self) -> History:
        return self.recorder.history(self.extinct, self.runaway)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Foraging and energetics  « thesis eqs 8-9 »                │
    # └────────────────────────────────────────────────────────────┘

    def forage_year(self) -> None:
        if not self.population:
            self._reset_stashes()
            return
        env, rng, pop = self.environment, self.rng, self.population
        items_taken = 0
        if self.life.uniform_foraging:
            for _ in range(self.life.foraging_rounds):
                order = rng.permutation(len(pop))
                picks = rng.integers(0, env.n_resources, size=len(pop))
                for slot, idx in zip(order, picks, strict=True):
                    idx = int(idx)
                    if env.take(idx):
                        self._eat(pop[slot], idx)
                        items_taken += 1
        else:
            search = [o.search_probs for o in pop]
            for _ in range(self.life.foraging_rounds):
                for slot in rng.permutation(len(pop)):
                    p = search[slot]
                    if p is None:
                        continue
                    idx = env.draw(p, rng)
                    if env.take(idx):
                        self._eat(pop[slot], idx)
                        items_taken += 1

        self._items_per_organism = np.fromiter((o.items_consumed for o in pop), dtype=float, count=len(pop))
        self._items_taken = float(items_taken)

        for org in pop:
            org.energy += org.fidelity_harvest * env.energy_per_item
            org.energy -= org.basal_metabolism + org.Ef + org.Eb + org.Ec
            org.fidelity_realised = (org.fidelity_harvest / org.items_consumed
                                     if org.items_consumed else NAN)
            org.items_consumed = 0
            org.fidelity_harvest = 0.0
        self._energy_pre_repro = float(np.mean([o.energy for o in pop]))

    @staticmethod
    def _eat(org: Organism, resource: int) -> None:
        org.items_consumed += 1
        org.fidelity_harvest += org.item_fidelity[resource]

    # ┌────────────────────────────────────────────────────────────┐
    # │ Reproduction and mutation  « thesis §2.7, §2.9 »           │
    # └────────────────────────────────────────────────────────────┘

    def reproduce(self) -> int:
        candidates = self._eligible()
        self._reproductive_frac = len(candidates) / len(self.population) if self.population else NAN
        self.rng.shuffle(candidates)  # type: ignore[arg-type]
        offspring: list[Organism] = []
        for i in range(0, len(candidates) - 1, 2):
            p1, p2 = candidates[i], candidates[i + 1]
            p1.energy -= self.life.repro_cost
            p2.energy -= self.life.repro_cost
            niche_parent = p1 if self.rng.random() < 0.5 else p2
            Ef = self._inherit(p1.Ef, p2.Ef)
            Eb = self._inherit(p1.Eb, p2.Eb)
            mu, sigma, Ef, Eb = self.mutate(niche_parent.mu, niche_parent.sigma, Ef, Eb)
            # the capsule draws random numbers only when the cost law has one,
            # so the thesis law's random stream is untouched
            Ec = self._inherit_and_mutate_capsule(p1.Ec, p2.Ec) if self.has_capsule else 0.0
            offspring.append(self._birth(mu, sigma, Ef, Eb, Ec, age=0, energy=self.life.offspring_energy))
        self.population.extend(offspring)
        return len(offspring)

    def _eligible(self) -> list[Organism]:
        """Bernoulli entry to the mating pool, graded by surplus (eq 10)."""
        p = np.clip([o.energy / self.life.repro_scale for o in self.population], 0.0, 1.0)
        draws = self.rng.random(len(p))
        return [o for o, keep in zip(self.population, draws < p, strict=True) if keep]

    def _inherit(self, trait1: float, trait2: float) -> float:
        """Uniparental: the offspring takes one parent's value at random."""
        return trait2 if self.rng.random() < 0.5 else trait1

    def mutate(self, mu: float, sigma: float, Ef: float, Eb: float) -> tuple[float, float, float, float]:
        m, rng = self.mutation, self.rng
        if rng.random() < m.arch_mutation_rate:
            Ef = max(Ef + rng.normal(0.0, m.E_mutation_sd), 0.0)
        if rng.random() < m.arch_mutation_rate:
            Eb = max(Eb + rng.normal(0.0, m.E_mutation_sd), 0.0)

        if Eb < self.e_switch:
            if rng.random() < m.duplication_rate:
                Eb = self._backend_gain()
                self.jumps_this_year += 1
        elif rng.random() < m.loss_rate:
            Eb = 0.0
            self.losses_this_year += 1

        if rng.random() < m.niche_mutation_rate:
            mu += rng.normal(0.0, m.mu_mutation_sd)
        if rng.random() < m.niche_mutation_rate:
            sigma *= np.exp(rng.normal(0.0, m.sigma_mutation_sd))
        return mu, sigma, Ef, Eb

    def _inherit_and_mutate_capsule(self, Ec1: float, Ec2: float) -> float:
        m, rng = self.mutation, self.rng
        Ec = self._inherit(Ec1, Ec2)
        if rng.random() < m.arch_mutation_rate:
            Ec = max(Ec + rng.normal(0.0, m.E_mutation_sd), 0.0)
        return Ec

    def _backend_gain(self) -> float:
        """Large-effect gain: a minimal functional back-end just past the switch."""
        m = self.mutation
        frac = m.jump_frac * float(np.exp(self.rng.normal(0.0, m.jump_sigma)))
        return self.architecture.Eb_for_fraction(min(frac, 0.95))

    # ┌────────────────────────────────────────────────────────────┐
    # │ Mortality  « thesis §2.8 »                                 │
    # └────────────────────────────────────────────────────────────┘

    def cull(self) -> tuple[int, int]:
        """Age everyone; remove the starved and the senescent."""
        survivors: list[Organism] = []
        starved = senescent = 0
        for o in self.population:
            o.age += 1
            if o.energy <= 0.0:
                starved += 1
            elif o.age > self.life.max_age:
                senescent += 1
            else:
                survivors.append(o)
        self.population = survivors
        return starved, senescent

    # ┌────────────────────────────────────────────────────────────┐
    # │ Bookkeeping                                                │
    # └────────────────────────────────────────────────────────────┘

    def _record(self, births: int = 0, starved: int = 0, senescent: int = 0) -> None:
        self.recorder.record(
            self.population, self.environment.stock,
            births=births, starved=starved, senescent=senescent,
            items_taken=self._items_taken, items_per_organism=self._items_per_organism,
            energy_pre_repro=self._energy_pre_repro, reproductive_frac=self._reproductive_frac,
            jumps=self.jumps_this_year, losses=self.losses_this_year,
        )

    def _check_status(self) -> bool:
        if not self.population:
            self.extinct = True
            return False
        if len(self.population) > self.life.runaway_guard:
            self.runaway = True
            return False
        return True
