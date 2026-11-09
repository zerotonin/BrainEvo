# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — ablations                                            ║
# ║  « necessity: remove one part, watch what breaks »               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Each ablation switches off one component of the grounded model. ║
# ║  Run on the taxon cases it shows which component each phenotype  ║
# ║  needs: the capsule for the grade split, the oxygen cap for the  ║
# ║  Carboniferous, saturation for the frog's loss, the minimal circu║
# ║  for any valley, and grounded prices for the brain-share ordering║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from brainevo.abm.cost_laws import CostLawParams
from brainevo.abm.energetics import ATPCostLawParams, CapsuleLaw
from brainevo.abm.population import InitialisationParams
from brainevo.abm.presets import Preset
from brainevo.abm.taxa import TAXA, Taxon, taxon_preset

Ablation = Callable[[ATPCostLawParams], ATPCostLawParams]

NO_SATURATION_HALF: float = 100.0
NO_MIN_CIRCUIT: float = 1e-4
NO_OXYGEN_CAP: float = 1e3
THESIS_OVERHEAD: float = 3.75

ABLATIONS: dict[str, Ablation] = {
    "no-capsule": lambda a: replace(a, capsule=None),
    "no-oxygen-cap": lambda a: replace(a, capsule=_uncapped(a)),
    "no-saturation": lambda a: replace(a, half_saturation=NO_SATURATION_HALF),
    "no-min-circuit": lambda a: replace(a, min_circuit=NO_MIN_CIRCUIT),
}
"""Name -> transformation of the grounded law.  ``thesis-prices`` is separate: it swaps the law."""


def _uncapped(a: ATPCostLawParams) -> CapsuleLaw | None:
    return None if a.capsule is None else replace(a.capsule, hard_cap_ref=NO_OXYGEN_CAP)

ABLATION_NAMES: tuple[str, ...] = (*ABLATIONS, "thesis-prices")


def ablate(taxon: Taxon, ablation: str) -> Preset:
    """The taxon's preset with one component removed."""
    base = taxon_preset(taxon)
    if ablation == "thesis-prices":
        cost = CostLawParams(Fmax_f=taxon.atp.Fmax_f, M=max(taxon.atp.M, 1.0), P=THESIS_OVERHEAD)
        return replace(base, name=f"{base.name}--thesis-prices", atp=None, cost=cost,
                       description=f"{taxon.name} under the thesis Shannon law (P = {THESIS_OVERHEAD}) instead of ATP")
    return replace(base, name=f"{base.name}--{ablation}", atp=ABLATIONS[ablation](taxon.atp),
                   description=f"{taxon.name} with {ablation.replace('-', ' ')}")


def seeded_from(taxon: Taxon, ancestor: Taxon, years: int = 6000, ablation: str | None = None) -> Preset:
    """``taxon``'s law (optionally ablated) with founders at ``ancestor``'s mean-field optimum."""
    from brainevo.ode.meanfield import MeanField  # local import: ode depends on abm

    mixed = MeanField(ancestor.architecture()).best_mixed()
    if mixed is None:
        raise RuntimeError(f"{ancestor.name} has no mixed optimum to seed from")
    preset = taxon_preset(taxon, years) if ablation is None else ablate(taxon, ablation)
    arch = preset.build_architecture()
    capsule = float(arch.optimal_capsule(mixed.Eb)) if getattr(arch, "has_capsule", False) else 0.0
    init = InitialisationParams(arch_mode="fixed_Eb", initial_Eb=float(mixed.Eb), initial_Ec=capsule)
    return replace(preset, name=f"{preset.name}-from-{ancestor.name}", init=init,
                   description=f"{preset.description}; seeded at the {ancestor.name} optimum (E_b = {mixed.Eb:.3f})")


def ablation_presets() -> dict[str, Preset]:
    """Every taxon x ablation, the frog's loss under each ablation, the demand-loss
    and reversibility seeds."""
    presets: dict[str, Preset] = {}
    for taxon in TAXA.values():
        for name in ABLATION_NAMES:
            p = ablate(taxon, name)
            presets[p.name] = p
    frog, mouse = TAXA["frog"], TAXA["mouse"]
    for name in ABLATION_NAMES:
        p = seeded_from(frog, mouse, ablation=name)
        presets[p.name] = p
    seeded_mouse = seeded_from(mouse, mouse)  # demand-loss sweep: lower atp__M from here
    presets[seeded_mouse.name] = seeded_mouse
    regain = seeded_from(mouse, frog)  # reversibility: a reduced lineage under the ancestral law
    presets[regain.name] = regain
    return presets
