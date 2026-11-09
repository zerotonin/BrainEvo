# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm.cost_laws                                        ║
# ║  « energy in, fidelity out: the two cost laws »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Front- and back-end convert metabolic investment E into         ║
# ║  fidelity F through a Shannon-type log channel with a soft cap:  ║
# ║                                                                  ║
# ║      F(E) = softcap[ B · log2(1 + SNR · E) ],   B = Fmax / k     ║
# ║                                                                  ║
# ║  The back-end differs in ceiling (M), rate (back_rate) and a     ║
# ║  fixed overhead e_switch = P · E_f,sat below which it returns    ║
# ║  nothing.  Thesis eqs (3)-(7); Dodd (2026) Table 4.              ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from brainevo.constants import FloatArray

TuningMode = Literal["absolute", "relative"]
SwitchShape = Literal["ramp", "hill"]


def soft_cap(raw: FloatArray | float, ceiling: float, sharpness: float) -> FloatArray | float:
    """Bend ``raw`` smoothly onto ``ceiling``: raw / (1 + (raw/ceiling)^s)^(1/s)."""
    return raw / (1.0 + (raw / ceiling) ** sharpness) ** (1.0 / sharpness)


def matched_filter_gain(
    resource_axis: FloatArray, mu: float, sigma: float,
    tuning_width: float | None, tuning_mode: TuningMode, tuning_floor: float,
) -> FloatArray:
    """Fraction of F_f realised on each resource type for a front-end tuned
    at ``mu`` (Wehner 1987 matched filter); all ones when untuned."""
    axis = np.asarray(resource_axis, dtype=float)
    if tuning_width is None:
        return np.ones_like(axis)
    w = tuning_width * float(sigma) if tuning_mode == "relative" else tuning_width
    w = max(float(w), 1e-9)
    g = np.exp(-0.5 * ((axis - float(mu)) / w) ** 2)
    return tuning_floor + (1.0 - tuning_floor) * g


@dataclass(frozen=True)
class CostLawParams:
    """Parameters of the two cost laws (thesis Table 4 and Table 6).

    Attributes:
        Fmax_f:       Front-end fidelity ceiling. Scales both ceilings.
        M:            Back-end ceiling relative to the front-end's (>= 1).
        P:            Back-end overhead as a multiple of the front-end
                      saturation energy; sets ``e_switch = P * E_f_sat``.
        k:            Rate constant shared by both ends; ``B = Fmax / k``.
        SNR:          Gain per unit investment energy inside the log.
        back_rate:    Back-end rate factor in (0, 1]. ``1`` is the
                      ceiling-scaled regime, ``1/M`` the rate-equal regime.
        sharpness:    Soft-cap exponent ``s`` (how abruptly F bends onto
                      its ceiling).
        q:            Fraction of the ceiling counted as "saturated".
        tuning_width: Matched-filter width ``w`` of the front-end, in
                      resource types. ``None`` gives an untuned front-end.
        tuning_mode:  ``"absolute"`` keeps ``w`` fixed; ``"relative"``
                      scales it with the organism's niche breadth.
        tuning_floor: Fraction of F_f still realised far off-tuning.
        switch_shape: How the back-end switches on.  ``"ramp"`` (thesis):
                      the overhead is deducted and the channel acts on
                      ``E_b - e_switch``; ``"hill"``: a Hill saturation
                      gate ``E_b^n / (E_b^n + e_switch^n)`` multiplies the
                      channel acting on the full ``E_b``.
        switch_width: Ramp only.  ``0`` is the hard kink of the thesis;
                      ``w > 0`` replaces ``max(E_b - e_switch, 0)`` by the
                      softplus ``w log(1 + exp((E_b - e_switch)/w))``.
        switch_steepness: Hill only.  Flank steepness ``n``; ``inf`` is a
                      step at ``e_switch``.
    """

    Fmax_f: float = 0.8
    M: float = 3.0
    P: float = 3.75
    k: float = 3.0
    SNR: float = 14.44
    back_rate: float = 1.0
    sharpness: float = 5.0
    q: float = 0.95
    tuning_width: float | None = None
    tuning_mode: TuningMode = "absolute"
    tuning_floor: float = 0.0
    switch_shape: SwitchShape = "ramp"
    switch_width: float = 0.0
    switch_steepness: float = float("inf")

    def __post_init__(self) -> None:
        if self.switch_shape not in ("ramp", "hill"):
            raise ValueError("switch_shape must be 'ramp' or 'hill'")
        if self.switch_width < 0.0:
            raise ValueError("switch_width must be >= 0")
        if self.switch_steepness <= 0.0:
            raise ValueError("switch_steepness must be > 0")
        if self.back_rate <= 0.0:
            raise ValueError("back_rate must be > 0")
        if self.M < 1.0:
            raise ValueError("M (back-end ceiling ratio) must be >= 1")
        if self.tuning_mode not in ("absolute", "relative"):
            raise ValueError("tuning_mode must be 'absolute' or 'relative'")
        if self.tuning_width is not None and self.tuning_width <= 0.0:
            raise ValueError("tuning_width must be > 0 (or None for untuned)")
        if not 0.0 <= self.tuning_floor <= 1.0:
            raise ValueError("tuning_floor must lie in [0, 1]")


class Architecture:
    """Nervous-system capability bought by metabolic spending, split
    into a front-end and a back-end with independent investments.

    Energy is in arbitrary units (the basal metabolic cost sets the
    scale); fidelity is unitless and multiplies energy per item.
    """

    def __init__(self, params: CostLawParams | None = None, **overrides: object) -> None:
        self.params = params if params is not None else CostLawParams(**overrides)  # type: ignore[arg-type]
        p = self.params
        self.Fmax_f = float(p.Fmax_f)
        self.M = float(p.M)
        self.Fmax_b = self.M * self.Fmax_f
        self.k = float(p.k)
        self.SNR = float(p.SNR)
        self.back_rate = float(p.back_rate)
        self.P = float(p.P)
        self.q = float(p.q)
        self.sharpness = float(p.sharpness)

        self.B_front = self.Fmax_f / self.k
        self.B_back = self.back_rate * self.Fmax_b / self.k

        self.e_switch = self.P * self.E_f_saturation
        self.tuning_mode: TuningMode = p.tuning_mode
        self.tuning_width = None if p.tuning_width is None else float(p.tuning_width)
        self.tuning_floor = float(p.tuning_floor)
        self.tuned = self.tuning_width is not None
        self.switch_shape: SwitchShape = p.switch_shape
        self.switch_width = float(p.switch_width)
        self.switch_steepness = float(p.switch_steepness)

    @property
    def is_hard_switch(self) -> bool:
        """True when F_b has the thesis' kink (or a step) at e_switch."""
        if self.switch_shape == "ramp":
            return self.switch_width <= 0.0
        return not np.isfinite(self.switch_steepness)

    def __repr__(self) -> str:
        return (f"Architecture(Fmax_f={self.Fmax_f:g}, M={self.M:g}, P={self.P:g}, "
                f"k={self.k:g}, SNR={self.SNR:g}, back_rate={self.back_rate:g}, "
                f"tuned={self.tuned})")

    # ┌────────────────────────────────────────────────────────────┐
    # │ Capability curves  « eqs 3, 5, 6 »                         │
    # └────────────────────────────────────────────────────────────┘

    def f_capability(self, E_front: FloatArray | float) -> FloatArray | float:
        """Front-end fidelity F_f(E_f), soft-capped at Fmax_f."""
        E = np.maximum(np.asarray(E_front, dtype=float), 0.0)
        raw = self.B_front * np.log1p(E * self.SNR) / np.log(2.0)
        return self._soft_cap(raw, self.Fmax_f)

    has_capsule: bool = False

    def b_capability(
        self, E_back: FloatArray | float, E_capsule: FloatArray | float | None = None
    ) -> FloatArray | float:
        """Back-end fidelity F_b(E_b): nothing below e_switch, log channel above.
        ``E_capsule`` is accepted for interface parity and ignored.

        The thesis form (``switch_shape="ramp"``, ``switch_width=0``) is the
        kink ``max(E_b - e_switch, 0)``; the smooth variants are described
        on :class:`CostLawParams`.
        """
        E = np.asarray(E_back, dtype=float)
        if self.switch_shape == "ramp":
            if self.switch_width <= 0.0:
                effective = np.maximum(E - self.e_switch, 0.0)
            else:
                effective = self.switch_width * np.logaddexp(0.0, (E - self.e_switch) / self.switch_width)
            raw = self.B_back * np.log1p(effective * self.SNR) / np.log(2.0)
        else:
            raw = self.B_back * np.log1p(np.maximum(E, 0.0) * self.SNR) / np.log(2.0) * self._hill_gate(E)
        return self._soft_cap(raw, self.Fmax_b)

    def _hill_gate(self, E: FloatArray) -> FloatArray:
        if self.e_switch <= 0.0:
            return np.ones_like(E)
        if not np.isfinite(self.switch_steepness):
            return (E > self.e_switch).astype(float)
        n = self.switch_steepness
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (np.maximum(E, 0.0) / self.e_switch) ** n
        return ratio / (1.0 + ratio)

    def capability(self, E_front: FloatArray | float, E_back: FloatArray | float) -> FloatArray | float:
        """Total fidelity, the additive (substitutable) combination."""
        return self.f_capability(E_front) + self.b_capability(E_back)

    def _soft_cap(self, raw: FloatArray | float, ceiling: float) -> FloatArray | float:
        return soft_cap(raw, ceiling, self.sharpness)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Inverse curves  « energy needed for a given fidelity »     │
    # └────────────────────────────────────────────────────────────┘

    def _E_front(self, raw: float) -> float:
        return (2.0 ** (raw / self.B_front) - 1.0) / self.SNR

    def _E_back(self, raw: float) -> float:
        return (2.0 ** (raw / self.B_back) - 1.0) / self.SNR

    def _raw_for_fraction(self, fraction: float, ceiling: float) -> float:
        """Raw (pre-cap) value whose soft-capped image is fraction * ceiling."""
        s = self.sharpness
        return ceiling * fraction / (1.0 - fraction ** s) ** (1.0 / s)

    @property
    def E_f_saturation(self) -> float:
        """Front-end investment at which F_f reaches q * Fmax_f."""
        return self._E_front(self._raw_for_fraction(self.q, self.Fmax_f))

    @property
    def E_b_saturation(self) -> float:
        """Back-end investment (including e_switch) at which F_b reaches q * Fmax_b."""
        return self.e_switch + self._E_back(self._raw_for_fraction(self.q, self.Fmax_b))

    @property
    def E_b_climb(self) -> float:
        """Investment above the switch needed to saturate the back-end."""
        return self.E_b_saturation - self.e_switch

    def Eb_for_fraction(self, fraction: float) -> float:
        """Back-end investment (including e_switch) buying ``fraction`` of Fmax_b.

        Exact for the hard ramp; for smooth switches it is the hard-ramp
        value and serves as a landing point for large-effect mutations."""
        fraction = float(np.clip(fraction, 1e-9, 1.0 - 1e-9))
        return self.e_switch + self._E_back(self._raw_for_fraction(fraction, self.Fmax_b))

    # ┌────────────────────────────────────────────────────────────┐
    # │ Matched filter  « Wehner (1987); thesis eq 7 »             │
    # └────────────────────────────────────────────────────────────┘

    def front_gain(self, resource_axis: FloatArray, mu: float, sigma: float) -> FloatArray:
        """Fraction of F_f realised on each resource type for an organism
        tuned at ``mu``.  The back-end is untuned and pays on every item."""
        return matched_filter_gain(resource_axis, mu, sigma, self.tuning_width, self.tuning_mode, self.tuning_floor)
