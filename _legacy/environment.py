import numpy as np
import matplotlib.pyplot as plt

try:                                     # Gaussian CDF, vectorised in C
    from scipy.special import ndtr as _phi
except ImportError:                      # fallback: correct but slow
    from math import erf as _erf_scalar
    _erf = np.vectorize(_erf_scalar, otypes=[float])

    def _phi(x):
        return 0.5 * (1.0 + _erf(x))

_EPS = 1e-12


class Environment:
    """
    Gaussian resource landscape over R_n discrete types.

    R is a stock of *items* (counts). One successful
    encounter removes exactly one item and yields `energy_per_item` energy.

    Parameters
    ----------
    n_resources: number of discrete resource types on the resource_axis
    env_mu, env_sigma : centre and breadth of the carrying-capacity curve
    (C_max): carrying capacity (items) of the single most abundant type
    growth_rate : logistic regrowth rate
    energy_per_item : energy yield of one item
    floor: minimum abundance, must be > 0 for logistic regrowth (otherwise resource extinction) 
    refuge_frac: unharvestable fraction of each bin, in [0, 1] - for resource growth stability

    """

    def __init__(self, n_resources, env_sigma, capacity_peak=100.0, regrowth_rate=1.0, energy_per_item=1.0, floor=1e-3, refuge_frac=0.3, between_gen_novelty=0, within_lifetime_drift=0,
                 volatility=0.0, volatility_corr_len=0.0):

        self.n_resources = n_resources 
        self.env_mu = n_resources // 2
        self.env_sigma = env_sigma
        self.capacity_peak = float(capacity_peak)
        self.regrowth_rate = regrowth_rate
        self.energy_per_item = energy_per_item
        self.floor = floor
        self.refuge_frac = refuge_frac

        # fraction of total capacity organisms 'demand' 
        self.f = 0.3

        self.resource_axis = np.arange(n_resources)
        self.bin_edges = np.arange(n_resources + 1) - 0.5

        self.capacity = self.capacity_peak * np.exp(-0.5 * ((self.resource_axis - self.env_mu) / self.env_sigma) ** 2)
        self.capacity = np.maximum(self.capacity, _EPS)          
        # long-run expected capacity; self.capacity is the realised value in a
        # volatile world and equals capacity_base when volatility = 0
        self.capacity_base = self.capacity.copy()
        # dynamic resource stock, starts at capacity
        self.stock = self.capacity.copy()                        

        # unharvestable floor of each bin
        self.refuge = self.refuge_frac * self.capacity
        # smallest stock at which a whole item can still be taken
        self._take_threshold = np.maximum(1.0, self.refuge + 1.0)

        # ---- resource fluctuation ----
        # volatility            sd of the per-year log-normal shock to each
        #                       type's carrying capacity (0 = static world)
        # volatility_corr_len   how far a shock spreads along the resource
        #                       axis. 0 gives independent types, so a broad
        #                       forager averages over many shocks and a
        #                       specialist rides one. Large values move every
        #                       type together, which removes any bet-hedging
        #                       benefit of breadth -- the null control.
        self.volatility = float(volatility)
        self.volatility_corr_len = float(volatility_corr_len)
        self._shock_kernel = self._build_shock_kernel()

        # fixed across organisms, determines shape of utilization curve 
        self.demand = self.f * self.capacity.sum()  

        # self.between_gen_novelty = float(between_gen_novelty)
        # self.within_lifetime_drift = float(within_lifetime_drift)


    # ------------------------------ niche ---------------------------------

    def resource_weights(self, mu, sigma):
        """ 
        raw utilization curve indicating organism resource preferences 
        off-axis only costs though narrower preference range - increased starvation risk with competition
        """
        sigma = max(float(sigma), _EPS)
        # discretise over the resources 
        return np.diff(_phi((self.bin_edges - mu) / sigma))

    
    def preference_distribution(self, mu, sigma):
        w = self.resource_weights(mu, sigma)
        total = w.sum()
        if total <= _EPS:
            return None
        return w / total
    
    def utilisation(self, mu, sigma, demand):
        return float(demand) * self.resource_weights(mu, sigma)

    # def available(self):
    #     """Stock a forager can actually reach, i.e. above the refuge."""
    #     return np.maximum(self.stock - self.refuge, 0.0)
    # def intersection(self, u):
    #     """ 
    #     encounter distrubution is the intersection between 
    #     an organisms niche-specific utilisation curve and world stock
    #     renormalsied to sum to 1.0
    #     """
    #     inter = np.minimum(u, self.stock) # possible replace self.stock with self.avaliable as such, refuge represent underlying hidden resources - only helping with regrowth dynamics
    #     total = inter.sum()
    #     if total <= _EPS:
    #         return None
    #     return inter / total
    
        
    def draw(self, p, rng):
        """" encounder distrubution as cumulative distrubution function for sampling """
        cdf = np.cumsum(p)
        cdf[-1] = 1.0       
        return int(np.searchsorted(cdf, rng.random(), side="right"))              


    # def take(self, resource):
    #     """Remove one item if the type is not exhausted.."""
    #     if self.stock[resource] >= 1.0:
    #         self.stock[resource] -= 1.0
    #         return True
    #     return False
    def take(self, resource):
        # remove one item, if a whole one is present above the refuge
        if self.stock[resource] >= max(1.0, self.refuge[resource] + 1.0):
            self.stock[resource] -= 1.0
            return True
        return False

    # ------------------------- resource fluctuation ------------------------

    def _build_shock_kernel(self):
        """
        Set the shock mode and, for smoothed shocks, build the kernel.

        volatility_corr_len (L):
            0          independent shock per resource type
            0 < L < inf  Gaussian-smoothed shocks, correlation length L
            inf        one shared ("global") shock for every type
        The kernel is normalised so every type keeps shock variance v^2.
        """
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
        return kernel / np.sqrt((kernel ** 2).sum())   # preserves shock variance

    def fluctuate(self, rng):
        """
        Re-draw carrying capacities for the coming year.

        capacity_z = capacity_base_z * exp(eps_z - 0.5 * volatility^2)

        The mean capacity is preserved; only its year-to-year reliability
        changes. Because a whole item can only be taken while stock exceeds the
        refuge, a bad year at a type is a hard block rather than a small
        reduction, which is what makes breadth worth paying for.stock
        """
        if self.volatility <= 0.0:
            return

        v = self.volatility
        if self.shock_mode == "global":
            # one draw shared by every type
            eps = np.full(self.n_resources, rng.normal(0.0, v))
        elif self.shock_mode == "smoothed":
            # pad the white noise and convolve in "valid" mode: output has
            # exactly n_resources values, no edge damping, and any L works
            # (mode="same" damped the edges and crashed once the kernel was
            # longer than the axis)
            half = (len(self._shock_kernel) - 1) // 2
            white = rng.normal(0.0, v, size=self.n_resources + 2 * half)
            eps = np.convolve(white, self._shock_kernel, mode="valid")
        else:
            eps = rng.normal(0.0, v, size=self.n_resources)

        self.capacity = np.maximum(
            self.capacity_base * np.exp(eps - 0.5 * self.volatility ** 2),
            _EPS,
        )
        self.refuge = self.refuge_frac * self.capacity
        self._take_threshold = np.maximum(1.0, self.refuge + 1.0)
        np.clip(self.stock, self.floor, self.capacity, out=self.stock)

    # ---------------------------- dynamics --------------------------------

    def regrow(self, dt=1.0):
        np.clip(self.stock, self.floor, None, out=self.stock)
        self.stock += self.regrowth_rate * self.stock * (1.0 - self.stock / self.capacity) * dt
        np.clip(self.stock, self.floor, self.capacity, out=self.stock)

    def reset(self):
        self.capacity = self.capacity_base.copy()
        self.refuge = self.refuge_frac * self.capacity
        self._take_threshold = np.maximum(1.0, self.refuge + 1.0)
        self.stock = self.capacity.copy()

    # ---------------------------- diagnostics --------------------------------

    @property
    def total_capacity(self):
        return float(self.capacity_base.sum())
 
    @property
    def sustainable_yield(self):
        f = self.refuge_frac
        return self.regrowth_rate * f * (1.0 - f) * self.total_capacity
 
    def harvestable_stock(self):
        reach = self.available()
        return float(reach[reach >= 1.0].sum())

# def plot_dists():
#     env = Environment(n_resources=20, env_sigma=6.0, peak_items=500)
#     z = env.z
    
#     DEMAND, SIGMA = env.demand, 2
#     MUS = [10.0, 15.0, 18.0]                 # on the axis: 0 .. n_resources-1
    
#     fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True)
    
#     for j, mu in enumerate(MUS):
#         u = env.utilisation(mu, SIGMA, DEMAND)
#         inter = np.minimum(u, env.stock)
#         p = env.intersection(u)
#         print(f"probs: {p}")

    
#         ax = axes[0, j]
#         ax.bar(z, env.R, width=.90, color="0.86", edgecolor="0.6", lw=.4,
#             zorder=1, label="R  (stock)")
#         ax.bar(z, u, width=.58, color="tab:blue", alpha=.75, edgecolor="none",
#             zorder=2, label=f"u  (mu={mu:g}, sum={u.sum():.1f})")
#         ax.bar(z, inter, width=.26, color="crimson", edgecolor="none",
#             zorder=3, label=f"I = min(u,R)")
#         ax.set_title(f"mu = {mu:g}",
#                     fontsize=9)
#         ax.legend(fontsize=6.5, loc="upper right")
#         if j == 0:
#             ax.set_ylabel("items")
    
#         ax = axes[1, j]
#         if p is None:                                   # organism reaches no stock
#             ax.text(.5, .5, "no overlap\np is None", ha="center", va="center",
#                     transform=ax.transAxes, fontsize=11, color="0.4")
#         else:
#             ax.bar(z, p, width=.80, color="crimson", edgecolor="none",
#                 label=f"p,  sum={p.sum():.3f},  mode={p.argmax()}")
#             ax.legend(fontsize=6.5)
#         ax.set_xlabel("resource type z")
#         ax.set_ylim(0, .30)
#         if j == 0:
#             ax.set_ylabel("p[z]")
    
#     for ax in axes.ravel():
#         ax.set_xticks(np.arange(0, 40, 5))
#         ax.set_xticks(z, minor=True)
#         ax.grid(axis="x", which="minor", color="0.94", lw=.5, zorder=0)
#         ax.set_xlim(-1, env.n_resources)
#         ax.set_ylim(bottom=0)
    
#     fig.suptitle(f"I[z] = min(u[z], R[z])  with demand={DEMAND:g}, sigma={SIGMA:g}  "
#                 f"—  sweeping mu toward the axis edge", y=.98, fontsize=11)
#     fig.tight_layout()
#     plt.show()