# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — taxa                                                 ║
# ║  « four animals the model must get right »                       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Qualitative test cases for the grounded law: an arthropod (high ║
# ║  front-end, small back-end), a mouse (small but strong back-end),║
# ║  frog (secondary brain reduction from a brainier ancestor) and a ║
# ║  primate (massive back-end). Every physiological number carries i║
# ║  source; the two niche numbers (Fmax_f, M) are declared assumptio║
# ║  and swept. The model passes when the ordering of back-end share ║
# ║  the frog's loss come out without tuning.                        ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import dataclass, replace

from brainevo.abm.energetics import ATPArchitecture, ATPCostLawParams, CapsuleLaw
from brainevo.abm.population import InitialisationParams
from brainevo.abm.presets import EXP4_ATP_UNIFORM, Preset


@dataclass(frozen=True)
class Taxon:
    """A grounded parameter set plus the phenotype the literature says to expect.

    Attributes:
        name:        Short identifier used in preset names.
        atp:         Grounded cost law (physiology + niche demand).
        expectation: One sentence on what the model must reproduce.
        sources:     Mapping decision -> citation, printed into the methods table.
        ancestor:    Taxon whose optimum seeds the secondary-loss run.
    """

    name: str
    atp: ATPCostLawParams
    expectation: str
    sources: dict[str, str]
    ancestor: str | None = None

    def architecture(self) -> ATPArchitecture:
        return ATPArchitecture(self.atp)


# ┌────────────────────────────────────────────────────────────┐
# │ Physiology  « shares of basal metabolism, one lineage each »│
# └────────────────────────────────────────────────────────────┘

ARTHROPOD = Taxon(
    name="arthropod",
    atp=ATPCostLawParams(
        Fmax_f=1.4, M=1.5, front_share=0.10, back_share=0.08, wiring_exponent=0.0, min_circuit=0.02,
        half_saturation=0.25,
        capsule=CapsuleLaw(lineage="tracheal", hard_cap_ref=0.35, transport=0.10, build_years=0.5, lifespan_years=20.0),
    ),
    expectation="Front-heavy: saturated compound-eye front-end, back-end pinned at the capsule/oxygen cap.",
    sources={
        "front_share 0.10": ("blowfly retina ~8-10 % of resting metabolism (Howard, Blakeslee & Laughlin 1987; Niven "
                             "& Laughlin 2008)"),
        "back_share 0.08": ("price, not outcome: cost of the reference neuron count, identical for every lineage "
                            "because cost per neuron is constant across species (Herculano-Houzel 2011); the "
                            "realised brain share is a model output compared with Mink et al. 1981"),
        "wiring_exponent 0": "insect neuropil scales isometrically, no white-matter tax (Polilov & Makarova 2020)",
        "hard_cap_ref 0.35": ("exoskeletal capsule fixed between moults; brain volume bounded by the head capsule "
                              "(Polilov 2015; Chittka & Niven 2009)"),
        "lineage tracheal": "diffusive oxygen supply, cap scales with pO2 (Harrison, Kaiser & VandenBrooks 2010)",
        "transport 0.10": "flight cost dominated by the thorax; head mass a small fraction (Dudley 2000)",
        "Fmax_f 1.4, M 1.5 (niche)": ("assumption: a fast sensorimotor niche rewards bandwidth and offers little "
                                      "scope for integration; swept"),
    },
)

MOUSE = Taxon(
    name="mouse",
    atp=ATPCostLawParams(
        Fmax_f=0.6, M=3.0, front_share=0.08, back_share=0.08, wiring_exponent=1.0 / 3.0, min_circuit=0.02,
        half_saturation=0.25,
        capsule=CapsuleLaw(lineage="convective", hard_cap_ref=0.5, transport=0.25,
                           build_years=1.0, lifespan_years=20.0),
    ),
    expectation=("Small but strong back-end: neuron count at the small cranial cap, clearly above the switch, "
                 "modest front-end."),
    sources={
        "back_share 0.08": ("price, not outcome: cost of the reference neuron count, identical for every lineage "
                            "because cost per neuron is constant across species (Herculano-Houzel 2011); the "
                            "realised brain share is a model output compared with Mink et al. 1981"),
        "front_share 0.08": ("price of a full-daylight retina, identical across vertebrates (Howard, Blakeslee & "
                             "Laughlin 1987 via Niven & Laughlin 2008); the realised retina share is an output"),
        "wiring_exponent 1/3": "white matter scales as grey matter^(4/3) in mammals (Zhang & Sejnowski 2000)",
        "hard_cap_ref 0.5": "small cranium, ~7x10^7 neurons (Herculano-Houzel, Mota & Lent 2006)",
        "half_saturation 0.25": ("default: pooled information half-saturates at a quarter of the ceiling count "
                                 "(unmeasured)"),
        "build_years 1": "brain growth complete within weeks (Workman et al. 2013)",
        "Fmax_f 0.6, M 3 (niche)": ("assumption: nocturnal, low-bandwidth senses, integrative olfactory and whisker "
                                    "niche (Hughes 1977); swept"),
    },
)

FROG = Taxon(
    name="frog",
    atp=ATPCostLawParams(
        Fmax_f=0.8, M=1.2, front_share=0.08, back_share=0.08, wiring_exponent=1.0 / 3.0, min_circuit=0.02,
        half_saturation=0.05,
        capsule=CapsuleLaw(lineage="convective", hard_cap_ref=0.5, transport=1.0, build_years=1.0, lifespan_years=20.0),
    ),
    expectation=("Secondary reduction: seeded with the ancestral back-end, neuron count falls below the cap and "
                 "the capsule follows."),
    sources={
        "back_share 0.08": ("price, not outcome: cost of the reference neuron count, identical for every lineage "
                            "because cost per neuron is constant across species (Herculano-Houzel 2011); the "
                            "realised brain share is a model output compared with Mink et al. 1981"),
        "front_share 0.08": ("price of a full-daylight retina, identical across vertebrates (Howard, Blakeslee & "
                             "Laughlin 1987 via Niven & Laughlin 2008); the realised retina share is an output"),
        "half_saturation 0.05": ("the loss driver: prey capture is solved by a handful of retinal and tectal "
                                 "detector classes, so information from additional neurons saturates early (Lettvin "
                                 "et al. 1959; Ewert 1987). Swept 0.02-0.25; the reduction needs n_half below ~0.1"),
        "transport 1.0": ("saltatory locomotion: cost rises with carried mass and the head sits at the end of a "
                          "lever (Taylor et al. 1980; Nauwelaerts & Aerts 2006); secondary to n_half for the loss"),
        "hard_cap_ref 0.5": "same cranial grade as the ancestor, so any reduction is a choice, not a cap",
        "Fmax_f 0.8, M 1.2 (niche)": ("assumption: sit-and-wait prey capture needs motion detection and little "
                                      "integration; swept"),
        "ancestor = mouse-like": ("amniote-grade back-end as the pre-reduction state; relative brain size is "
                                  "smallest in Lissamphibia among tetrapods (Jerison 1973; Taylor, Nol & Boire 1995; "
                                  "Roth, Blanke & Wake 1994)"),
    },
    ancestor="mouse",
)

PRIMATE = Taxon(
    name="primate",
    atp=ATPCostLawParams(
        Fmax_f=1.2, M=4.0, front_share=0.08, back_share=0.08, wiring_exponent=1.0 / 3.0, min_circuit=0.02,
        half_saturation=1.0,
        capsule=CapsuleLaw(lineage="convective", hard_cap_ref=1.5, transport=0.25,
                           build_years=4.0, lifespan_years=20.0),
    ),
    expectation=("Massive back-end: neuron count at a large cranial cap, big capsule, front-end also high (diurnal "
                 "vision)."),
    sources={
        "back_share 0.08": ("price, not outcome: cost of the reference neuron count, identical for every lineage "
                            "because cost per neuron is constant across species (Herculano-Houzel 2011); the "
                            "realised brain share is a model output compared with Mink et al. 1981"),
        "front_share 0.08": ("price of a full-daylight retina, identical across vertebrates (Howard, Blakeslee & "
                             "Laughlin 1987 via Niven & Laughlin 2008); the realised retina share is an output"),
        "half_saturation 1.0": ("social and foraging cognition keep paying for more neurons (Dunbar 1998; DeCasien, "
                                "Williams & Higham 2017); information-limiting correlations weak at the circuit "
                                "scale"),
        "hard_cap_ref 1.5": ("larger cranium with linear neuron scaling (Herculano-Houzel 2009); realised brain "
                             "share should land at 15-25 % of RMR (Mink et al. 1981; Aiello & Wheeler 1995)"),
        "build_years 4": "human brain glucose use peaks at 66 % of RMR in childhood (Kuzawa et al. 2014)",
        "Fmax_f 1.2, M 4 (niche)": ("assumption: patchy ephemeral fruit and social cognition reward integration "
                                    "(Clutton-Brock & Harvey 1980); swept"),
    },
)

TAXA: dict[str, Taxon] = {t.name: t for t in (ARTHROPOD, MOUSE, FROG, PRIMATE)}


# ┌────────────────────────────────────────────────────────────┐
# │ Presets  « naive founders, and the secondary-loss run »    │
# └────────────────────────────────────────────────────────────┘

def taxon_preset(taxon: Taxon, years: int = 6000) -> Preset:
    """Naive founders evolve under the taxon's grounded law."""
    return replace(EXP4_ATP_UNIFORM, name=f"taxon-{taxon.name}", description=taxon.expectation,
                   atp=taxon.atp, years=years)


def secondary_loss_preset(taxon: Taxon, years: int = 6000) -> Preset:
    """Founders carry the ancestor's mean-field optimum, then evolve under ``taxon``'s law."""
    if taxon.ancestor is None:
        raise ValueError(f"{taxon.name} has no ancestor defined")
    from brainevo.ode.meanfield import MeanField  # local import: ode depends on abm, not the reverse

    ancestor = TAXA[taxon.ancestor]
    mixed = MeanField(ancestor.architecture()).best_mixed()
    if mixed is None:
        raise RuntimeError(f"ancestor {ancestor.name} has no mixed optimum to seed from")
    arch = taxon.architecture()
    init = InitialisationParams(arch_mode="fixed_Eb", initial_Eb=float(mixed.Eb),
                                initial_Ec=float(arch.optimal_capsule(mixed.Eb)))
    return replace(EXP4_ATP_UNIFORM, name=f"taxon-{taxon.name}-from-{ancestor.name}",
                   description=f"{taxon.expectation} Seeded at the {ancestor.name} optimum (E_b={mixed.Eb:.3f}).",
                   atp=taxon.atp, init=init, years=years)


def taxon_presets() -> dict[str, Preset]:
    presets = {f"taxon-{t.name}": taxon_preset(t) for t in TAXA.values()}
    for t in TAXA.values():
        if t.ancestor is not None:
            p = secondary_loss_preset(t)
            presets[p.name] = p
    return presets


def sources_table() -> str:
    """Markdown table of every taxon decision and its source."""
    lines = ["| taxon | decision | source |", "|---|---|---|"]
    for t in TAXA.values():
        for decision, source in t.sources.items():
            lines.append(f"| {t.name} | {decision} | {source} |")
    return "\n".join(lines)
