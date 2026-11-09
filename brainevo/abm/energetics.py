# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm.energetics                                       ║
# ║  « every constant in ATP, every decision with a citation »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Grounded alternative to the phenomenological Shannon cost law   ║
# ║  of cost_laws.py.  Costs are fractions of the organism's basal   ║
# ║  metabolic rate (the model's b = 8 energy units per year), so    ║
# ║  that the literature's "retina = 8 % of resting metabolism" or   ║
# ║  "brain = 2-20 % of RMR" map onto the model without a per-       ║
# ║  species ATP budget.  Shapes are the measured scalings:          ║
# ║    front-end  cost ∝ (information)^1.7      Niven et al. 2007    ║
# ║    back-end   cost ∝ N · (1 + wiring tax),  Herculano-Houzel     ║
# ║               capability saturating in N    2011; Zohary 1994;   ║
# ║               with a minimal-circuit gate    Moreno-Bote 2014    ║
# ║  Each GroundedConstant carries value, range and source so the    ║
# ║  sensitivity sweep and the methods table write themselves.       ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from brainevo.abm.cost_laws import SwitchShape, TuningMode, matched_filter_gain, soft_cap
from brainevo.constants import FloatArray


@dataclass(frozen=True)
class GroundedConstant:
    """A number with provenance.

    Attributes:
        value:   Default used by the model.
        low/high: Plausible range for sensitivity analysis.
        unit:    Physical unit.
        source:  Citation and the sentence or figure the number comes from.
    """

    value: float
    low: float
    high: float
    unit: str
    source: str


# ┌────────────────────────────────────────────────────────────┐
# │ Front-end  « photoreceptors as the measured sensory channel »│
# └────────────────────────────────────────────────────────────┘

FRONT_COST_EXPONENT = GroundedConstant(
    1.7, 1.5, 1.7, "dimensionless",
    "Niven, Anderson & Laughlin 2007 PLoS Biol 5:e116, Results: across four fly species the total "
    "photoreceptor cost increases as (information rate)^1.7; the dark (fixed) cost as (rate)^1.47.",
)
FRONT_FIXED_FRACTION = GroundedConstant(
    0.20, 0.20, 0.25, "fraction of peak consumption",
    "Niven et al. 2007, Abstract and Discussion: fixed cost ~20 % of a photoreceptor's maximum "
    "consumption, 'between a fifth and a quarter' across species.",
)
PHOTORECEPTOR_PEAK_RATE = GroundedConstant(
    955.0, 197.0, 1130.0, "bit s^-1 per cell",
    "Niven et al. 2007: D. melanogaster 197, D. virilis 512, Calliphora 955, Sarcophaga 1130 bit/s at "
    "the brightest intensity.",
)
PHOTORECEPTOR_PEAK_ATP = GroundedConstant(
    7e9, 5.7e8, 7e9, "ATP s^-1 per cell",
    "Niven et al. 2007: fully light-adapted Calliphora R1-6 photoreceptor 7e9 ATP/s; Laughlin, "
    "de Ruyter van Steveninck & Anderson 1998 Nat Neurosci 1:36 give the same figure.",
)
ATP_PER_BIT_GRADED = GroundedConstant(
    1e6, 1e6, 1e7, "ATP bit^-1",
    "Laughlin et al. 1998: 1e6-1e7 ATP per bit for graded signals in a photoreceptor or interneuron, "
    "against ~1e4 ATP per bit at a chemical synapse.",
)
RETINA_SHARE_OF_RMR = GroundedConstant(
    0.08, 0.08, 0.10, "fraction of resting metabolic rate",
    "Howard, Blakeslee & Laughlin 1987 Proc R Soc B 231:415 via Niven & Laughlin 2008 J Exp Biol "
    "211:1792 (~8 % for Calliphora); Harris, Jolivet & Attwell 2012 Neuron 75:762 quote ~10 %.",
)

# ┌────────────────────────────────────────────────────────────┐
# │ Back-end  « central neurons: linear cost, saturating gain »│
# └────────────────────────────────────────────────────────────┘

GLUCOSE_PER_NEURON = GroundedConstant(
    5.79e-9, 4.93e-9, 7.05e-9, "umol glucose min^-1 neuron^-1 (whole brain)",
    "Herculano-Houzel 2011 PLoS ONE 6:e17514, Table 1: mouse 5.20, rat 6.10, squirrel 7.05, macaque "
    "4.93, baboon 6.00, human 5.44 (x1e-9); 'remarkably constant, varying only by 40 %'. Cortex is "
    "~1.5e-8, cerebellum ~0.9e-9.",
)
ATP_PER_GLUCOSE = GroundedConstant(
    31.0, 30.0, 32.0, "ATP per glucose",
    "Rich 2003 Biochem Soc Trans 31:1095: modern P/O ratios give ~31 ATP per glucose oxidised.",
)
ATP_PER_NEURON = GroundedConstant(
    1.8e9, 1.5e9, 2.2e9, "ATP s^-1 neuron^-1",
    "Derived: 5.79e-9 umol/min x 6.022e17 molecules/umol / 60 s x 31 ATP/glucose = 1.8e9. Consistent "
    "with Attwell & Laughlin 2001 J Cereb Blood Flow Metab 21:1133 (8000 synapses x 1.6e5 ATP/s at "
    "4 Hz ~ 1.3e9 ATP/s for synaptic signalling alone, via Harris & Attwell 2012 J Neurosci 32:356).",
)
NON_SIGNALLING_FRACTION = GroundedConstant(
    0.25, 0.25, 0.45, "fraction of neural energy",
    "Attwell & Laughlin 2001 assume 25 % housekeeping; Engl & Attwell 2015 J Physiol 593:3417 'just "
    "under half'; Engl, Jolivet, Hall & Attwell 2016 JCBFM measure ~45 % in rat slices.",
)
WIRING_TAX_EXPONENT = GroundedConstant(
    1.0 / 3.0, 0.0, 1.0 / 3.0, "dimensionless",
    "Zhang & Sejnowski 2000 PNAS 97:5621: white matter ∝ grey matter^(4/3) across mammals, so wire "
    "per neuron grows as N^(1/3). Polilov & Makarova 2020 Sci Rep 10:4573: insect neuropil:rind is "
    "isometric (exponent ~1.0), so the insect tax exponent is 0.",
)
BRAIN_SHARE_OF_RMR = GroundedConstant(
    0.08, 0.02, 0.20, "fraction of resting metabolic rate",
    "Mink, Blumenschine & Adams 1981 Am J Physiol 241:R203: vertebrate brains 2-8 % of RMR; Aiello & "
    "Wheeler 1995 Curr Anthropol 36:199: human ~20 %, primates ~8-9 %.",
)
MIN_CIRCUIT_FRACTION = GroundedConstant(
    0.02, 0.003, 0.10, "fraction of the ceiling neuron count",
    "Weakest link. Chittka & Niven 2009 Curr Biol 19:R995: network analyses put the minimum for "
    "counting at a few hundred neurons; Caron et al. 2013 Nature 497:113: ~2000 Kenyon cells form the "
    "fly's associative centre, ~2 % of ~1e5 brain neurons (Raji & Potter 2021 eLife 10:e65376).",
)
HALF_SATURATION_FRACTION = GroundedConstant(
    0.25, 0.1, 1.0, "fraction of the ceiling neuron count",
    "Shape only: pooled information under information-limiting correlations rises as N/(N + N_half) "
    "(Zohary, Shadlen & Newsome 1994 Nature 370:140; Moreno-Bote et al. 2014 Nat Neurosci 17:1410). "
    "N_half is not measured for any central circuit; swept.",
)

# ┌────────────────────────────────────────────────────────────┐
# │ Capsule  « the box, its supply, and carrying it around »   │
# └────────────────────────────────────────────────────────────┘

SUPPORT_TISSUE_FRACTION = GroundedConstant(
    0.10, 0.05, 0.15, "fraction of neural running cost per unit capacity",
    "Maintenance of the non-neuronal box: glia ~5-10 % of brain energy (Attwell & Laughlin 2001; Harris, "
    "Jolivet & Attwell 2012), capillary length per neuron conserved (Karbowski 2011 PLoS ONE 6:e26709), bone "
    "2.3 vs brain 1008 kJ kg^-1 d^-1 (Elia 1992, in Energy Metabolism: Tissue Determinants, Raven Press).",
)
TRANSPORT_COST_FRACTION = GroundedConstant(
    0.25, 0.10, 0.50, "fraction of neural running cost per unit capacity",
    "Carrying the head: metabolic rate rises in proportion to carried load (Taylor, Heglund, McMahon & "
    "Looney 1980 J Exp Biol 86:9); field metabolic rate ~3x basal (Nagy 2005 J Exp Biol 208:1621); brain "
    "plus capsule ~1-2 % of body mass (Mink et al. 1981). (3-1) x 0.01 x basal / (0.08 basal) ~ 0.25.",
)
CAPSULE_BUILD_YEARS = GroundedConstant(
    2.0, 1.0, 4.0, "years of running cost, amortised over the lifespan",
    "Developmental cost of building central tissue: human brain glucose use peaks at ~66 % of resting "
    "metabolism in childhood (Kuzawa et al. 2014 PNAS 111:13010); integrated over development this is a "
    "few adult-years of running cost. Not measured for any invertebrate.",
)
MODERN_PO2 = GroundedConstant(
    21.2, 15.0, 35.0, "kPa",
    "0.209 x 101.3 kPa today; Phanerozoic range ~15-35 kPa with a Carboniferous-Permian peak (Berner 2006 "
    "Geochim Cosmochim Acta 70:5653, GEOCARBSULF).",
)
TRACHEAL_CAP_EXPONENT = GroundedConstant(
    1.5, 1.0, 3.0, "dimensionless",
    "Tracheal (diffusive) supply caps insect tissue size with atmospheric pO2: Carboniferous insects "
    "1.6-1.7x modern linear size at ~1.5x pO2 (Harrison, Kaiser & VandenBrooks 2010 Proc R Soc B 277:1937); "
    "tracheal investment grows hypermetrically with size (Kaiser et al. 2007 PNAS 104:13198); brain hypoxia "
    "tracks distance to tracheoles (Baccino-Calace et al. 2020 Biol Open 9:bio052183). CONTESTED: Snelling "
    "et al. 2026 Nature argue tracheal supply does not limit body size; the exponent is therefore swept "
    "down to the convective case.",
)
HAEMOGLOBIN_P50 = GroundedConstant(
    3.6, 3.3, 4.0, "kPa",
    "Half-saturation of haemoglobin, standard conditions (Severinghaus 1979 J Appl Physiol 46:599).",
)
HAEMOGLOBIN_HILL = GroundedConstant(
    2.7, 2.5, 2.9, "dimensionless",
    "Hill coefficient of the oxygen dissociation curve (Severinghaus 1979).",
)
ALVEOLAR_FRACTION = GroundedConstant(
    0.62, 0.55, 0.70, "fraction of atmospheric pO2 reaching the exchange surface",
    "Alveolar gas equation: ~13.3 kPa alveolar at 21.2 kPa atmospheric (West, Respiratory Physiology).",
)
CONVECTIVE_HEADROOM = GroundedConstant(
    1.4, 1.2, 2.0, "dimensionless",
    "Vertebrate brains run near their oxidative ceiling: OEF ~0.38, oxidative-rate headroom ~1.4x "
    "(Herculano-Houzel & Rothman 2022 Front Integr Neurosci 16:818685).",
)

GROUNDED_CONSTANTS: dict[str, GroundedConstant] = {
    k: v for k, v in globals().items() if isinstance(v, GroundedConstant)
}


# ┌────────────────────────────────────────────────────────────┐
# │ Parameters and the cost law                                │
# └────────────────────────────────────────────────────────────┘

Lineage = Literal["tracheal", "convective"]


@dataclass(frozen=True)
class CapsuleLaw:
    """Cost of housing central tissue, and the lineage's hard oxygen cap.

    Capacity ``c`` is measured like the neuron count, relative to the
    ceiling the back-end energy share buys (``c = 1`` houses ``n = 1``).

    Attributes:
        lineage:        ``"tracheal"`` (diffusive supply, insect) or
                        ``"convective"`` (vascular supply, vertebrate).
        pO2_kPa:        Atmospheric partial pressure of oxygen.
        maintenance:    Running cost of support tissue per unit capacity, as
                        a fraction of neural running cost (SUPPORT_TISSUE_FRACTION).
        transport:      Cost of carrying the capsule (TRANSPORT_COST_FRACTION).
        build_years:    Construction cost in years of running cost
                        (CAPSULE_BUILD_YEARS), amortised over ``lifespan_years``.
        lifespan_years: Amortisation window (the model's max age).
        hard_cap_ref:   Capacity the lineage can supply at MODERN_PO2.
        tracheal_exponent: Cap ∝ (pO2 / modern)^exponent for the tracheal lineage.
    """

    lineage: Lineage = "convective"
    pO2_kPa: float = MODERN_PO2.value
    maintenance: float = SUPPORT_TISSUE_FRACTION.value
    transport: float = TRANSPORT_COST_FRACTION.value
    build_years: float = CAPSULE_BUILD_YEARS.value
    lifespan_years: float = 20.0
    hard_cap_ref: float = 1.0
    tracheal_exponent: float = TRACHEAL_CAP_EXPONENT.value

    def __post_init__(self) -> None:
        if self.lineage not in ("tracheal", "convective"):
            raise ValueError("lineage must be 'tracheal' or 'convective'")
        for name in ("pO2_kPa", "lifespan_years", "hard_cap_ref"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        for name in ("maintenance", "transport", "build_years", "tracheal_exponent"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def cost_fraction(self) -> float:
        """Yearly capsule cost per unit capacity, relative to neural running cost."""
        return self.maintenance + self.transport + self.build_years / self.lifespan_years

    @property
    def hard_cap(self) -> float:
        """Capacity the oxygen supply can sustain at ``pO2_kPa``.

        Tracheal: Fick diffusion, cap ∝ pO2^exponent.  Convective: cap
        follows arterial haemoglobin saturation, which barely moves above
        ~12 kPa, so vertebrates are insensitive to atmospheric oxygen until
        it falls far.
        """
        ratio = self.pO2_kPa / MODERN_PO2.value
        if self.lineage == "tracheal":
            return self.hard_cap_ref * ratio**self.tracheal_exponent
        return self.hard_cap_ref * _saturation(self.pO2_kPa) / _saturation(MODERN_PO2.value)


def _saturation(pO2_atm: float) -> float:
    p = ALVEOLAR_FRACTION.value * pO2_atm
    n, p50 = HAEMOGLOBIN_HILL.value, HAEMOGLOBIN_P50.value
    return p**n / (p**n + p50**n)


@dataclass(frozen=True)
class ATPCostLawParams:
    """Grounded cost law in units of the organism's basal metabolic rate.

    Attributes:
        basal:            Basal metabolic rate in model energy units per year
                          (the thesis' b = 8); all shares are relative to it.
        Fmax_f:           Front-end fidelity ceiling (sweep variable, as in the thesis).
        M:                Back-end ceiling relative to the front-end's.
        front_share:      Energy at which the front-end reaches its design
                          ceiling, as a share of basal (RETINA_SHARE_OF_RMR).
        front_exponent:   Cost ∝ information^alpha (FRONT_COST_EXPONENT).
        back_share:       Energy at the back-end ceiling neuron count
                          (BRAIN_SHARE_OF_RMR).
        wiring_exponent:  Wire tax per neuron ∝ n^gamma (WIRING_TAX_EXPONENT).
        wiring_weight:    Wire tax at the ceiling relative to neuron cost.
        min_circuit:      Minimal functional circuit as a fraction of the
                          ceiling neuron count (MIN_CIRCUIT_FRACTION).
        half_saturation:  N_half as a fraction of the ceiling count.
        switch_shape / switch_steepness: how capability switches on at the
                          minimal circuit: ``"ramp"`` (hard) or ``"hill"``.
        sharpness, q:     Soft-cap exponent and saturation fraction, as in
                          :class:`brainevo.abm.cost_laws.CostLawParams`.
        tuning_*:         Matched-filter front-end, as in the thesis law.
        capsule:          When set, capacity becomes a third heritable trait
                          E_c with this cost law and hard cap; when ``None``
                          the neuron count is soft-capped at the ceiling share.
    """

    basal: float = 8.0
    Fmax_f: float = 0.8
    M: float = 3.0
    front_share: float = RETINA_SHARE_OF_RMR.value
    front_exponent: float = FRONT_COST_EXPONENT.value
    back_share: float = BRAIN_SHARE_OF_RMR.value
    wiring_exponent: float = WIRING_TAX_EXPONENT.value
    wiring_weight: float = 0.5
    min_circuit: float = MIN_CIRCUIT_FRACTION.value
    half_saturation: float = HALF_SATURATION_FRACTION.value
    switch_shape: SwitchShape = "ramp"
    switch_steepness: float = float("inf")
    sharpness: float = 5.0
    q: float = 0.95
    tuning_width: float | None = None
    tuning_mode: TuningMode = "absolute"
    tuning_floor: float = 0.0
    capsule: CapsuleLaw | None = None

    def __post_init__(self) -> None:
        for name in ("basal", "Fmax_f", "front_share", "front_exponent", "back_share", "half_saturation"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.M <= 0.0:
            raise ValueError("M must be > 0")
        if not 0.0 <= self.min_circuit < 1.0:
            raise ValueError("min_circuit must lie in [0, 1)")
        if self.wiring_exponent < 0.0 or self.wiring_weight < 0.0:
            raise ValueError("wiring tax must be non-negative")


class ATPArchitecture:
    """Grounded cost law with the same interface as :class:`Architecture`.

    Front-end::

        i_f(E) = (E / E_f,max)^(1/alpha),   E_f,max = front_share * basal
        F_f    = softcap(Fmax_f * i_f, Fmax_f)

    Back-end, with n = N / N_ceiling the neuron count relative to the
    ceiling the back-end share can buy::

        E_b(n) = E_b,max * n * (1 + w n^gamma) / (1 + w)      (cost, inverted numerically)
        g(n)   = gate(n - n_min) * (n - n_min) / (n - n_min + n_half)
        F_b    = Fmax_b * g(softcap(n, 1)) / g(1)                   (ceiling acts on n)

    The thesis quantities are read off:  e_switch = E_b(n_min),
    P = e_switch / E_f,sat.
    """

    def __init__(self, params: ATPCostLawParams | None = None, **overrides: object) -> None:
        self.params = params if params is not None else ATPCostLawParams(**overrides)  # type: ignore[arg-type]
        p = self.params
        self.basal = float(p.basal)
        self.Fmax_f = float(p.Fmax_f)
        self.M = float(p.M)
        self.Fmax_b = self.M * self.Fmax_f
        self.sharpness = float(p.sharpness)
        self.q = float(p.q)
        self.alpha = float(p.front_exponent)
        self.E_f_max = float(p.front_share) * self.basal
        self.E_b_max = float(p.back_share) * self.basal
        self.gamma = float(p.wiring_exponent)
        self.w_wire = float(p.wiring_weight)
        self.n_min = float(p.min_circuit)
        self.n_half = float(p.half_saturation)
        self.switch_shape: SwitchShape = p.switch_shape
        self.switch_steepness = float(p.switch_steepness)
        self.tuning_mode: TuningMode = p.tuning_mode
        self.tuning_width = None if p.tuning_width is None else float(p.tuning_width)
        self.tuning_floor = float(p.tuning_floor)
        self.tuned = self.tuning_width is not None
        self.capsule = p.capsule
        self.has_capsule = p.capsule is not None
        self.hard_cap = p.capsule.hard_cap if p.capsule is not None else 1.0
        self._g_ceiling = float(self._gain(np.array(1.0)))
        self.e_switch = self.cost_of_neurons(self.n_min) + self.capsule_cost(self.n_min)
        self.P = self.e_switch / self.E_f_saturation

    def __repr__(self) -> str:
        cap = ""
        if self.capsule is not None:
            cap = f", capsule={self.capsule.lineage} pO2={self.capsule.pO2_kPa:g} cap={self.hard_cap:.2f}"
        return (f"ATPArchitecture(Fmax_f={self.Fmax_f:g}, M={self.M:g}, front_share={self.E_f_max / self.basal:g}, "
                f"back_share={self.E_b_max / self.basal:g}, n_min={self.n_min:g} -> P={self.P:.2f}{cap})")

    @property
    def is_hard_switch(self) -> bool:
        return self.switch_shape == "ramp" or not np.isfinite(self.switch_steepness)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Front-end                                                  │
    # └────────────────────────────────────────────────────────────┘

    def f_capability(self, E_front: FloatArray | float) -> FloatArray | float:
        """F_f(E): measured convex cost inverted, soft-capped at Fmax_f."""
        E = np.maximum(np.asarray(E_front, dtype=float), 0.0)
        raw = self.Fmax_f * (E / self.E_f_max) ** (1.0 / self.alpha)
        return soft_cap(raw, self.Fmax_f, self.sharpness)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Back-end                                                   │
    # └────────────────────────────────────────────────────────────┘

    def cost_of_neurons(self, n: FloatArray | float) -> FloatArray | float:
        """Energy of a back-end of relative size n (per-neuron cost plus wire tax)."""
        n = np.maximum(np.asarray(n, dtype=float), 0.0)
        return self.E_b_max * n * (1.0 + self.w_wire * n**self.gamma) / (1.0 + self.w_wire)

    def neurons_for_cost(self, E_back: FloatArray | float) -> FloatArray | float:
        """Invert :meth:`cost_of_neurons` (monotone)."""
        E = np.atleast_1d(np.asarray(E_back, dtype=float))
        out = np.empty_like(E)
        for i, e in enumerate(E):
            if e <= 0.0:
                out[i] = 0.0
                continue
            hi = max(1.0, e / self.E_b_max * (1.0 + self.w_wire)) + 1.0
            out[i] = brentq(lambda n, e=e: float(self.cost_of_neurons(n)) - e, 0.0, hi, xtol=1e-12)
        return out if np.ndim(E_back) else float(out[0])

    def _gain(self, n: FloatArray) -> FloatArray:
        """Saturating pooled information above the minimal circuit."""
        excess = n - self.n_min
        if self.switch_shape == "ramp" or not np.isfinite(self.switch_steepness):
            active = np.maximum(excess, 0.0)
            gate = np.ones_like(n)
        else:
            active = np.maximum(n, 0.0)
            k = self.switch_steepness
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (np.maximum(n, 0.0) / max(self.n_min, 1e-12)) ** k
            gate = ratio / (1.0 + ratio) if self.n_min > 0.0 else np.ones_like(n)
        return gate * active / (active + self.n_half)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Capsule  « capacity is bought, and capped by oxygen »      │
    # └────────────────────────────────────────────────────────────┘

    def capsule_cost(self, capacity: FloatArray | float) -> FloatArray | float:
        """Yearly cost of housing ``capacity`` (same volume law as the neurons)."""
        if self.capsule is None:
            return np.zeros_like(np.asarray(capacity, dtype=float)) if np.ndim(capacity) else 0.0
        return self.capsule.cost_fraction * self.cost_of_neurons(capacity)

    def capacity_for_cost(self, E_capsule: FloatArray | float) -> FloatArray | float:
        """Invert :meth:`capsule_cost`."""
        if self.capsule is None:
            return np.full_like(np.asarray(E_capsule, dtype=float), np.inf) if np.ndim(E_capsule) else float("inf")
        return self.neurons_for_cost(np.asarray(E_capsule, dtype=float) / self.capsule.cost_fraction)

    def capsule_for_neurons(self, n: FloatArray | float) -> FloatArray | float:
        """Smallest capsule investment that houses ``n`` without capping it."""
        return self.capsule_cost(np.minimum(n, self.hard_cap))

    def optimal_capsule(self, E_back: FloatArray | float) -> FloatArray | float:
        """Capsule investment a well-adapted organism pairs with ``E_back``:
        exactly the capacity its neurons need, never more (excess capacity
        costs and returns nothing) and never above the oxygen cap."""
        return self.capsule_for_neurons(self.neurons_for_cost(E_back))

    def effective_neurons(self, E_back: FloatArray | float, E_capsule: FloatArray | float | None = None) -> FloatArray:
        """Neurons bought by E_b, limited by the box.

        Without a capsule law the count is soft-capped at the ceiling share.
        With one the limit is hard: a capsule is a volume and the oxygen cap
        is a supply rate, so neurons beyond min(capacity, hard cap) are
        simply not there.  A hard min also keeps the exactly fitting capsule
        lossless, which is what :meth:`optimal_capsule` assumes.
        """
        n = np.asarray(self.neurons_for_cost(E_back), dtype=float)
        if not self.has_capsule:
            return soft_cap(n, 1.0, self.sharpness)
        capacity = np.asarray(self.capacity_for_cost(0.0 if E_capsule is None else E_capsule), dtype=float)
        return np.minimum(n, np.minimum(capacity, self.hard_cap))

    def b_capability(
        self, E_back: FloatArray | float, E_capsule: FloatArray | float | None = None
    ) -> FloatArray | float:
        """F_b(E_b, E_c): neurons bought by E_b, housed by E_c, gated at the
        minimal circuit, saturating.

        Without a capsule law the neuron count is soft-capped at the ceiling
        share (Herculano-Houzel & Rothman 2022; Polilov & Makarova 2020).
        With one, the cap is min(capacity bought by E_c, oxygen hard cap).
        """
        n_eff = self.effective_neurons(E_back, E_capsule)
        out = self.Fmax_b * self._gain(n_eff) / self._g_ceiling
        return out if np.ndim(E_back) or (E_capsule is not None and np.ndim(E_capsule)) else float(out)

    def capability(self, E_front: FloatArray | float, E_back: FloatArray | float) -> FloatArray | float:
        return self.f_capability(E_front) + self.b_capability(E_back)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Derived energies  « same names as the thesis law »         │
    # └────────────────────────────────────────────────────────────┘

    def _energy_for(self, fn, target: float, lo: float, hi: float) -> float:
        return float(brentq(lambda E: float(fn(E)) - target, lo, hi, xtol=1e-10))

    @property
    def E_f_saturation(self) -> float:
        """Front-end investment at which F_f reaches q * Fmax_f."""
        return self._energy_for(self.f_capability, self.q * self.Fmax_f, 0.0, 50.0 * self.E_f_max)

    @property
    def max_back_fidelity(self) -> float:
        """F_b reachable under the oxygen cap (equals Fmax_b when uncapped)."""
        return self._b_with_optimal_capsule(self.cost_of_neurons(1e3))

    def _b_with_optimal_capsule(self, E_back: float) -> float:
        return float(self.b_capability(E_back, self.optimal_capsule(E_back)))

    @property
    def E_b_saturation(self) -> float:
        """Back-end investment (including the minimal circuit) at q * Fmax_b,
        with the capsule that fits.  Infinite if the oxygen cap sits below
        the saturation count."""
        target = self.q * self.Fmax_b
        hi = self.cost_of_neurons(1e3)
        if self._b_with_optimal_capsule(hi) < target:
            return float("inf")
        return self._energy_for(self._b_with_optimal_capsule, target, 0.0, hi)

    @property
    def E_b_climb(self) -> float:
        sat = self.E_b_saturation
        if not np.isfinite(sat):  # capped lineage: use the energy at the cap instead
            sat = self.cost_of_neurons(self.hard_cap) + self.capsule_cost(self.hard_cap)
        return max(sat - self.e_switch, 1e-9)

    def Eb_for_fraction(self, fraction: float) -> float:
        """Back-end investment buying ``fraction`` of Fmax_b."""
        target = float(np.clip(fraction, 1e-9, 0.999)) * self.Fmax_b
        hi = self.cost_of_neurons(1e3)
        if self._b_with_optimal_capsule(hi) < target:
            raise ValueError(f"fraction {fraction:g} of Fmax_b is above the oxygen cap of this lineage")
        return self._energy_for(self._b_with_optimal_capsule, target, 0.0, hi)

    def front_gain(self, resource_axis: FloatArray, mu: float, sigma: float) -> FloatArray:
        return matched_filter_gain(resource_axis, mu, sigma, self.tuning_width, self.tuning_mode, self.tuning_floor)


def constants_table() -> str:
    """Markdown table of every grounded constant, for the methods section."""
    rows = ["| constant | value | range | unit | source |", "|---|---|---|---|---|"]
    for name, c in GROUNDED_CONSTANTS.items():
        rows.append(f"| {name} | {c.value:g} | {c.low:g}-{c.high:g} | {c.unit} | {c.source} |")
    return "\n".join(rows)
