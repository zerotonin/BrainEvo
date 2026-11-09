# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm.environment                                      ║
# ║  « a Gaussian resource axis that depletes, regrows, fluctuates » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Z discrete resource types with Gaussian carrying capacity       ║
# ║  (Ackermann & Doebeli 2004).  Stock is an item count; a hit      ║
# ║  removes one item if the type is above its refuge.  Logistic     ║
# ║  regrowth once a year; optional mean-preserving log-normal       ║
# ║  volatility with tunable correlation length along the axis.      ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from brainevo.constants import EPS, FloatArray


@dataclass(frozen=True)
class EnvironmentParams:
    """Resource environment (thesis Table 3 and the volatility table).

    Attributes:
        n_resources:         Number of discrete resource types Z.
        env_sigma:           SD of the Gaussian carrying-capacity curve.
        capacity_peak:       Carrying capacity of the central type (items).
        regrowth_rate:       Logistic regrowth rate per year.
        energy_per_item:     Energy yield of one item before fidelity.
        floor:               Minimum stock kept for logistic regrowth.
        refuge_frac:         Unharvestable fraction of each type's capacity.
        volatility:          SD ``v`` of the yearly log-scale capacity shock.
        volatility_corr_len: Correlation length ``L`` of shocks along the
                             axis; 0 independent, ``inf`` one shared shock.
    """

    n_resources: int = 100
    env_sigma: float = 15.0
    capacity_peak: float = 400.0
    regrowth_rate: float = 1.0
    energy_per_item: float = 1.0
    floor: float = 1e-3
    refuge_frac: float = 0.1
    volatility: float = 0.0
    volatility_corr_len: float = 0.0


class Environment:
    """Gaussian resource landscape over ``n_resources`` discrete types."""

    def __init__(self, params: EnvironmentParams | None = None, **overrides: object) -> None:
        self.params = params if params is not None else EnvironmentParams(**overrides)  # type: ignore[arg-type]
        p = self.params
        self.n_resources = int(p.n_resources)
        self.env_mu = self.n_resources // 2
        self.env_sigma = float(p.env_sigma)
        self.capacity_peak = float(p.capacity_peak)
        self.regrowth_rate = float(p.regrowth_rate)
        self.energy_per_item = float(p.energy_per_item)
        self.floor = float(p.floor)
        self.refuge_frac = float(p.refuge_frac)

        self.resource_axis = np.arange(self.n_resources)
        self.bin_edges = np.arange(self.n_resources + 1) - 0.5

        capacity = self.capacity_peak * np.exp(
            -0.5 * ((self.resource_axis - self.env_mu) / self.env_sigma) ** 2
        )
        self.capacity_base: FloatArray = np.maximum(capacity, EPS)
        self.capacity: FloatArray = self.capacity_base.copy()
        self.stock: FloatArray = self.capacity.copy()
        self.refuge: FloatArray = self.refuge_frac * self.capacity

        self.volatility = float(p.volatility)
        self.volatility_corr_len = float(p.volatility_corr_len)
        self.shock_mode: str = "none"
        self._shock_kernel = self._build_shock_kernel()

    def __repr__(self) -> str:
        return (f"Environment(Z={self.n_resources}, sigma_env={self.env_sigma:g}, "
                f"K_peak={self.capacity_peak:g}, refuge={self.refuge_frac:g}, "
                f"volatility={self.volatility:g}, L={self.volatility_corr_len:g})")

    # ┌────────────────────────────────────────────────────────────┐
    # │ Niche  « an organism's encounter distribution »            │
    # └────────────────────────────────────────────────────────────┘

    def resource_weights(self, mu: float, sigma: float) -> FloatArray:
        """Gaussian preference over types, discretised at bin edges."""
        sigma = max(float(sigma), EPS)
        return np.diff(ndtr((self.bin_edges - mu) / sigma))

    def preference_distribution(self, mu: float, sigma: float) -> FloatArray | None:
        """Normalised encounter probabilities; ``None`` if the niche is off-axis."""
        w = self.resource_weights(mu, sigma)
        total = w.sum()
        if total <= EPS:
            return None
        return w / total

    def draw(self, p: FloatArray, rng: np.random.Generator) -> int:
        """Sample one resource type from encounter distribution ``p``."""
        cdf = np.cumsum(p)
        cdf[-1] = 1.0
        return int(np.searchsorted(cdf, rng.random(), side="right"))

    def take(self, resource: int) -> bool:
        """Remove one item if a whole one is present above the refuge."""
        if self.stock[resource] >= max(1.0, self.refuge[resource] + 1.0):
            self.stock[resource] -= 1.0
            return True
        return False

    # ┌────────────────────────────────────────────────────────────┐
    # │ Volatility  « thesis eqs 1-2 »                             │
    # └────────────────────────────────────────────────────────────┘

    def _build_shock_kernel(self) -> FloatArray | None:
        if self.volatility <= 0.0:
            self.shock_mode = "none"
            return None
        if np.isinf(self.volatility_corr_len):
            self.shock_mode = "global"
            return None
        if self.volatility_corr_len <= 0.0:
            self.shock_mode = "independent"
            return None
        self.shock_mode = "smoothed"
        half = int(np.ceil(3.0 * self.volatility_corr_len))
        offsets = np.arange(-half, half + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / self.volatility_corr_len) ** 2)
        return kernel / np.sqrt((kernel ** 2).sum())  # preserves shock variance

    def fluctuate(self, rng: np.random.Generator) -> None:
        """Re-draw carrying capacities for the coming year (mean-preserving)."""
        if self.volatility <= 0.0:
            return
        v = self.volatility
        if self.shock_mode == "global":
            eps = np.full(self.n_resources, rng.normal(0.0, v))
        elif self.shock_mode == "smoothed":
            assert self._shock_kernel is not None
            half = (len(self._shock_kernel) - 1) // 2
            white = rng.normal(0.0, v, size=self.n_resources + 2 * half)
            eps = np.convolve(white, self._shock_kernel, mode="valid")
        else:
            eps = rng.normal(0.0, v, size=self.n_resources)

        self.capacity = np.maximum(self.capacity_base * np.exp(eps - 0.5 * v ** 2), EPS)
        self.refuge = self.refuge_frac * self.capacity
        np.clip(self.stock, self.floor, self.capacity, out=self.stock)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Dynamics  « logistic regrowth once a year »                │
    # └────────────────────────────────────────────────────────────┘

    def regrow(self, dt: float = 1.0) -> None:
        np.clip(self.stock, self.floor, None, out=self.stock)
        self.stock += self.regrowth_rate * self.stock * (1.0 - self.stock / self.capacity) * dt
        np.clip(self.stock, self.floor, self.capacity, out=self.stock)

    def reset(self) -> None:
        self.capacity = self.capacity_base.copy()
        self.refuge = self.refuge_frac * self.capacity
        self.stock = self.capacity.copy()

    # ┌────────────────────────────────────────────────────────────┐
    # │ Diagnostics                                                │
    # └────────────────────────────────────────────────────────────┘

    @property
    def total_capacity(self) -> float:
        return float(self.capacity_base.sum())

    @property
    def sustainable_yield(self) -> float:
        """Items per year regrown when every type sits at its refuge."""
        f = self.refuge_frac
        return self.regrowth_rate * f * (1.0 - f) * self.total_capacity

    @property
    def harvestable_stock(self) -> float:
        """Whole items currently above the refuge."""
        reach = np.maximum(self.stock - self.refuge, 0.0)
        return float(np.floor(reach).sum())
