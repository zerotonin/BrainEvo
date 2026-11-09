# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm.presets                                          ║
# ║  « the thesis experiments as named parameter sets »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Dodd (2026) §2.10: Experiment 1 (cost-law sweep, uniform        ║
# ║  foraging), Experiment 2 (matched-filter tuning, niche evolves), ║
# ║  Experiment 3 (resource volatility).  Each preset bundles the    ║
# ║  five parameter dataclasses; sweeps override single fields.      ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from brainevo.abm.cost_laws import Architecture, CostLawParams
from brainevo.abm.energetics import ATPArchitecture, ATPCostLawParams, CapsuleLaw
from brainevo.abm.environment import EnvironmentParams
from brainevo.abm.population import InitialisationParams, LifeHistoryParams, MutationParams


@dataclass(frozen=True)
class Preset:
    """A complete, named model configuration."""

    name: str
    description: str
    cost: CostLawParams
    environment: EnvironmentParams
    life: LifeHistoryParams
    mutation: MutationParams
    init: InitialisationParams
    years: int
    atp: ATPCostLawParams | None = None
    """When set, the grounded ATP law replaces ``cost`` (which is then ignored)."""

    def build_architecture(self) -> Architecture | ATPArchitecture:
        return ATPArchitecture(self.atp) if self.atp is not None else Architecture(self.cost)

    def with_overrides(self, **fields: object) -> Preset:
        """Return a copy with dotted overrides, e.g. ``cost__M=2.5``."""
        groups: dict[str, dict[str, object]] = {}
        for key, value in fields.items():
            group, _, field = key.partition("__")
            if not field:
                raise KeyError(f"override must be '<group>__<field>', got {key!r}")
            groups.setdefault(group, {})[field] = value
        updates = {}
        for g, kv in groups.items():
            if g == "capsule":  # nested under atp
                if self.atp is None or self.atp.capsule is None:
                    raise KeyError(f"preset {self.name!r} has no capsule law to override")
                atp = updates.get("atp", self.atp)
                updates["atp"] = replace(atp, capsule=replace(atp.capsule, **kv))
                continue
            current = updates.get(g, getattr(self, g))
            if current is None:
                raise KeyError(f"preset {self.name!r} has no {g!r} parameters to override")
            updates[g] = replace(current, **kv)
        return replace(self, **updates)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Preset:
        """Rebuild a preset from ``to_dict`` output (e.g. a run's params.json)."""
        atp_data = data.get("atp")
        atp = None
        if atp_data is not None:
            atp_data = dict(atp_data)  # type: ignore[arg-type]
            capsule = atp_data.pop("capsule", None)
            atp = ATPCostLawParams(**atp_data, capsule=None if capsule is None else CapsuleLaw(**capsule))
        return cls(
            name=str(data["name"]), description=str(data.get("description", "")), years=int(data["years"]),  # type: ignore[arg-type]
            cost=CostLawParams(**data["cost"]), environment=EnvironmentParams(**data["environment"]),  # type: ignore[arg-type]
            life=LifeHistoryParams(**data["life"]), mutation=MutationParams(**data["mutation"]),  # type: ignore[arg-type]
            init=InitialisationParams(**data["init"]), atp=atp,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "description": self.description, "years": self.years,
            "cost": asdict(self.cost), "environment": asdict(self.environment),
            "life": asdict(self.life), "mutation": asdict(self.mutation), "init": asdict(self.init),
            "atp": None if self.atp is None else asdict(self.atp),
        }


_ENV = EnvironmentParams()
_LIFE_UNIFORM = LifeHistoryParams(uniform_foraging=True)
_LIFE_NICHE = LifeHistoryParams(uniform_foraging=False)
_MUT_ARCH_ONLY = MutationParams(niche_mutation_rate=0.0)
_MUT_FULL = MutationParams()

EXP1_COST_SWEEP = Preset(
    name="exp1-cost-sweep",
    description="Cost-law sweep: niche fixed and uniform foraging, only E_f and E_b evolve "
                "(thesis §2.10.2; vary cost__Fmax_f, cost__M, cost__P, cost__back_rate).",
    cost=CostLawParams(Fmax_f=1.2, M=2.5, P=2.0, back_rate=0.8),
    environment=_ENV, life=_LIFE_UNIFORM, mutation=_MUT_ARCH_ONLY,
    init=InitialisationParams(arch_mode="naive", niche_mode="fixed"), years=6000,
)

EXP1_OVERHEAD_SWEEP = EXP1_COST_SWEEP.with_overrides(cost__Fmax_f=1.2, cost__M=2.5)
EXP1_OVERHEAD_SWEEP = replace(EXP1_OVERHEAD_SWEEP, name="exp1-overhead-sweep", years=4000,
                              description="Overhead sweep at Fmax=1.2, M=2.5; vary cost__P (thesis Fig 8).")

EXP2_TUNING = Preset(
    name="exp2-tuning",
    description="Matched-filter front-end, absolute tuning width w; niche position and breadth evolve "
                "(thesis §2.10.3, Figs 11-12).",
    cost=CostLawParams(Fmax_f=0.8, M=3.0, P=3.75, back_rate=0.8, tuning_width=3.75, tuning_mode="absolute"),
    environment=_ENV, life=_LIFE_NICHE, mutation=_MUT_FULL,
    init=InitialisationParams(arch_mode="naive", niche_mode="fixed", sigma_init=15.0), years=4000,
)

EXP2_RELATIVE_CONTROL = replace(
    EXP2_TUNING, name="exp2-relative-control",
    description="Control: tuning width scales with niche breadth, so a broad niche is never penalised.",
    cost=replace(EXP2_TUNING.cost, tuning_mode="relative", tuning_width=0.25),
)

EXP3_VOLATILITY = replace(
    EXP2_TUNING, name="exp3-volatility",
    description="Experiment 2 architecture under year-to-year capacity volatility v=0.25 "
                "with one shock shared by all types (L=inf) (thesis §2.10.4, Fig 13).",
    environment=replace(_ENV, volatility=0.25, volatility_corr_len=float("inf")),
)

EXP4_ATP_UNIFORM = replace(
    EXP1_COST_SWEEP, name="exp4-atp-uniform",
    description="Grounded ATP cost law (energetics.py), niche fixed, uniform foraging; sweep atp__Fmax_f, atp__M, "
                "atp__min_circuit, atp__back_share, atp__wiring_exponent.",
    atp=ATPCostLawParams(Fmax_f=1.2, M=2.5),
)

EXP4_ATP_TUNING = replace(
    EXP2_TUNING, name="exp4-atp-tuning",
    description="Grounded ATP cost law with the matched-filter front-end; niche evolves.",
    atp=ATPCostLawParams(Fmax_f=0.8, M=3.0, tuning_width=3.75),
)

EXP5_CAPSULE_CONVECTIVE = replace(
    EXP4_ATP_UNIFORM, name="exp5-capsule-convective",
    description="Grounded law with an evolving capsule (third trait E_c), vertebrate-type convective oxygen "
                "supply; sweep capsule__pO2_kPa, capsule__transport, capsule__build_years, atp__Fmax_f.",
    atp=replace(EXP4_ATP_UNIFORM.atp, capsule=CapsuleLaw(lineage="convective")),
)

EXP5_CAPSULE_TRACHEAL = replace(
    EXP5_CAPSULE_CONVECTIVE, name="exp5-capsule-tracheal",
    description="As exp5-capsule-convective but insect-type tracheal supply (hard cap ∝ pO2^1.5) and isometric "
                "neuropil (no wiring tax).",
    atp=replace(EXP4_ATP_UNIFORM.atp, wiring_exponent=0.0, capsule=CapsuleLaw(lineage="tracheal")),
)

PRESETS: dict[str, Preset] = {
    p.name: p for p in (EXP1_COST_SWEEP, EXP1_OVERHEAD_SWEEP, EXP2_TUNING, EXP2_RELATIVE_CONTROL, EXP3_VOLATILITY,
                        EXP4_ATP_UNIFORM, EXP4_ATP_TUNING, EXP5_CAPSULE_CONVECTIVE, EXP5_CAPSULE_TRACHEAL)
}


def all_presets() -> dict[str, Preset]:
    """Experiment presets plus the taxon test cases (which live in :mod:`brainevo.abm.taxa`)."""
    from brainevo.abm.taxa import taxon_presets  # local import: taxa builds on the presets above

    return {**PRESETS, **taxon_presets()}
