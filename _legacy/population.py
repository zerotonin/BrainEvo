import numpy as np
import matplotlib.pyplot as plt

from .architecture import Architecture
from .environment import Environment
from .organism import Organism
 
NAN = float("nan")

try:                                       # average ranks for ties
    from scipy.stats import rankdata as _rankdata
except ImportError:                        # fallback: ordinal ranks
    def _rankdata(x):
        return np.argsort(np.argsort(x)).astype(float) + 1.0


def spearman(x, y):
    """Rank correlation, NaN-safe. NaN when either trait has no variance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return NAN
    rx = _rankdata(x[mask])
    ry = _rankdata(y[mask])
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return NAN
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))

 
LOG_KEYS = (
    # demography
    "population_size", "births", "deaths",
    "deaths_starved", "deaths_senescent", 
    # the answer
    "committed_frac",
    "a_norm_mean", "a_norm_sd", 
    "a_raw_mean", "a_raw_sd", 
    "a_F_mean", "a_F_sd",   
    # magnitude -- a=0 from "both saturated" is not a=0 from "both zero"
    "Ef_mean", "Ef_sd", "Eb_mean", "Eb_sd", "neural_spend_mean",
    "Ff_mean", "Ff_sd",
    "Fb_mean", "Fb_sd",
    "items_consumed_mean", "items_consumed_sd", "fidelity_mean", "fidelity_sd",
    # tuned front-end: how much of F_f each organism actually cashes in
    "tuning_efficiency_mean", "tuning_efficiency_sd",
    "fidelity_realised_mean", "fidelity_realised_sd",
    # the architecture <-> niche-breadth hypothesis, measured every year
    "rho_sigma_Ef", "rho_sigma_Eb", "rho_sigma_a_norm", "rho_sigma_tuning",
    "rho_absmu_Ef", "rho_absmu_Eb",
    # calibration diagnostics
    "energy", "energy_pre_repro", "reproductive_frac", "age",
    "above_switch_frac",
    # niche (constant while the niche is switched off -- a drift check)
    "mu", "mu_sd", "sigma", "sigma_sd",
    "jumps", "losses",

    # environment
    "stock_depleted", "standing_stock", "harvestable_stock",
)
 

# in each generation:
#     for each organism(o):
#     ->calculate the intersection forage function

#     for i in k:
#     ->shuffle organism array of o
#         ->-> roll on intersection function (draw stochastically but weighed by distrubution)
#         ->->if item abundance > 0:
#             ->->->o.energy += energy_per_item (currently 1.0)
#             ->->->item abundance -= 1

class Population:
    """
    Evolution of niche position (mu) and width (sigma).
 
    One year = `k` foraging rounds. In each round the roster is shuffled and
    every organism gets one roll against the intersection of its
    utilisation curve with the standing stock. A hit removes one item from R.
    """

 
    def __init__(self, environment, architecture, initial_population,
                 initial_energy=10.0, offspring_energy=5.0, repro_scale=60.0,
                 repro_cost=8.0, 
                 basal_metabolism=5.0, max_age=20, E_mutation_sd=0.05,
                 mu_mutation_sd=0.0, sigma_mutation_sd=0.0, niche_mutation_rate = 0.0, arch_mutation_rate=0.0,
                 seed_above_switch = False, initial_Eb=None, initial_backend_frac=None,
                 duplication_rate = 0.001, loss_rate=0.001, jump_overshoot=0.10, jump_sigma=0.5,               
                 sigma_min=1.5, sigma_max=60.0,
                 foraging_rounds=20, sigma_init = None, naive_ns=False, naive_niche=False, 
                 uniform_foraging = True,
                 runaway_guard=100000, rng=None):
 
        self.environment = environment
        self.architecture = architecture
        self.e_switch = architecture.e_switch
        self.E_b_climb = architecture.E_b_saturation - architecture.e_switch

        self.rng = np.random.default_rng(rng)
 
        # life history
        self.max_age = max_age
        self.initial_energy = initial_energy
        self.offspring_energy = offspring_energy
        self.basal_metabolism = basal_metabolism

        self.repro_scale = repro_scale #basal_metabolism * 5    # about 5x the annual basal metabolic cost
        self.repro_cost = repro_cost
        self.demand = environment.demand
 
        # foraging schedule
        self.foraging_rounds = foraging_rounds # rounds per year
        self.uniform_foraging = uniform_foraging    # for testing fidelity only

        # inheritance
        self.dominance=1.0 # set to 1 - fully clonal trait inheritance

        self.E_mutation_sd = E_mutation_sd
        self.Ef_mutation_sd = self.E_mutation_sd
        self.Eb_mutation_sd = self.E_mutation_sd  

        self.mu_mutation_sd = mu_mutation_sd
        self.sigma_mutation_sd = sigma_mutation_sd
        self.arch_mutation_rate = arch_mutation_rate
        self.niche_mutation_rate = niche_mutation_rate
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_init = sigma_init

        # gene duplication
        self.jump_frac=0.01
        self.duplication_rate = duplication_rate
        self.loss_rate = loss_rate
        self.initial_backend_frac=initial_backend_frac
        self.jump_overshoot = jump_overshoot
        self.jump_sigma     = jump_sigma
        self.jumps_this_year = 0
        self.losses_this_year = 0

        self._jumps_since_print = 0
        self._losses_since_print = 0
        # guards
        self.runaway_guard = runaway_guard
 
        # bookkeeping
        self.sim_time = 0
        self.extinct = False
        self.runaway = False
        self.log = {key: [] for key in LOG_KEYS}
 
        # mid-year stashes: record() runs after the year is over and cannot
        # see these otherwise
        self._reset_stashes()

        self.initial_Eb = initial_Eb
        self.initial_population = initial_population 

        self.population = []
        self.initialise_population(self.initial_population, seed_above_switch=seed_above_switch, naive_ns=naive_ns, naive_niche=naive_niche, initial_backend_frac=None)
        # print("Population initial_Eb argument:", self.initial_Eb)
        # print(
        #     "Founder Eb values:",
        #     [org.Eb for org in self.population[:10]]
        # )
        # print(
        #     "Founder mean Eb:",
        #     np.mean([org.Eb for org in self.population])
        # )

        self.record()
 


    def _reset_stashes(self):
        self._org_items_consumed = np.empty(0)
        self._env_items_taken = NAN
        self._energy_pre_repro = NAN
        self._reproductive_frac = NAN
    # ------------------------------ setup ---------------------------------
 

    def initialise_population(self, n, seed_above_switch=False, naive_ns=False, naive_niche=False, initial_backend_frac=None):
        """
        initalise n individuals with ages 0-maximum
        - if naive_ns all organisms have zero neural investment (Ef=Eb=0).
        - if naive_niche all organisms have the same niche (mu, sigma) from env_mu and env_sigma
        """

        arch = self.architecture

        for _ in range(int(n)):

            # seed niche
            if self.sigma_init:
                sigma = self.sigma_init
                mu = self.environment.env_mu
            elif naive_niche: 
                mu = self.environment.env_mu
                sigma = self.environment.env_sigma
            else:
                mu = self.rng.uniform(0.0,  self.environment.n_resources - 1.0)
                sigma = self.rng.uniform(2.0, 10.0)

            # Initial condition convergence experiment
            if self.initial_Eb is not None:
                Ef = 0.0
                Eb = float(self.initial_Eb)

            # Test split starting population 
            elif initial_backend_frac is not None:
                Ef = 0.0
                if self.rng.random() < initial_backend_frac:
                    Eb = arch.e_switch + 0.2 * self.E_b_climb
                else:
                    Eb = 0.0

            # seed architecture
            elif seed_above_switch:
                Ef = 0.0
                Eb = (arch.e_switch + (0.2 * self.E_b_climb) * np.exp(self.rng.normal(0.0, 0.05)))

            elif naive_ns:
                Ef = 0.0
                Eb = 0.0           

            else:
                Ef = self.rng.uniform(0.0, 1.2 * self.architecture.E_f_saturation)
                Eb = self.rng.uniform(0.0, 1.2 * self.architecture.E_b_saturation)

            age = int(self.rng.integers(0, self.max_age))
            self.population.append(self.birth_organism(mu, sigma, Ef, Eb, age=age))


 
    def birth_organism(self, mu, sigma, Ef, Eb, age=0, energy=None):
        """
        construct an organism (for init and repo)
        """
        # check mu withing resource axis 
        mu = float(np.clip(mu, 0.0, self.environment.n_resources - 1.0))
        # check sigma in set range 
        sigma = float(np.clip(sigma, self.sigma_min, self.sigma_max))
        org = Organism(self.environment, self.architecture, 
                        Ef=Ef, Eb=Eb, mu=mu, sigma=sigma, age=age,
                        basal_metabolism=self.basal_metabolism,
                        initial_energy=self.initial_energy if energy is None else energy)
        # print(org.Fidelity)
        return org
 
    # ---------------------------- foraging --------------------------------
 
    def search_distributions(self):
        return [
            org.search_probs
            for org in self.population
        ]

    # def encounter_distributions(self):
    #     """
    #     List of intersections between each organism's utilisation curve and the current resource stock
    #     """
    #     # return [self.environment.intersection(org.resource_weights * self.demand) for org in self.population]
    #     p_dists = []
    #     for org in self.population:
    #         weights = org.resource_weights
    #         total = weights.sum()
    #         if total <= 0.0:
    #             p_dists.append(None)
    #         else:
    #             p_dists.append(weights / total)
    #     return p_dists
    
 
    def forage_year(self):
        """
        foraging_rounds shuffled rounds; one roll per organism per round to get i resources, then settles organisms energy 
        """

        if not self.population:
            self._reset_stashes()
            return

        if self.uniform_foraging:
            search_dists = None
        else:
            search_dists = self.search_distributions()

        items_taken = 0

        for _ in range(self.foraging_rounds):
            if self.uniform_foraging:
                order = self.rng.permutation(len(self.population))
                picks = self.rng.integers(0,  self.environment.n_resources, size=len(self.population))
                for slot, idx in zip(order, picks):
                    idx = int(idx)
                    if self.environment.take(idx):
                        org = self.population[slot]
                        org.items_consumed += 1
                        # the item's type decides how much of the front-end pays
                        org.fidelity_harvest += org.item_fidelity[idx]
                        items_taken += 1

            else:  # niche based forraging
                for slot in self.rng.permutation(len(self.population)):
                    p = search_dists[slot]
                    if p is None:                     # niche entirely off the axis
                        continue
                    idx = self.environment.draw(p, self.rng)
                    if self.environment.take(idx):
                        org = self.population[slot]
                        org.items_consumed += 1
                        org.fidelity_harvest += org.item_fidelity[idx]
                        items_taken += 1

        # stash items taken for record before reset
        self._org_items_consumed = np.fromiter((o.items_consumed for o in self.population),
                                       dtype=float, count=len(self.population))
        self._env_items_taken = float(items_taken)  

        ### --- energetics for year --- ###
        for org in self.population:
            # gain: fidelity is summed per item, so a tuned front-end only
            # pays on items drawn near the organism's own mu. With an untuned
            # architecture item_fidelity is flat and this reduces exactly to
            # items_consumed * energy_per_item * Fidelity.
            org.energy += (org.fidelity_harvest * self.environment.energy_per_item)
            # loss
            org.energy -= (org.basal_metabolism + org.Ef + org.Eb)
            # realised per-item fidelity, for diagnostics
            org.fidelity_realised = (org.fidelity_harvest / org.items_consumed
                                     if org.items_consumed else NAN)
            # reset items
            org.items_consumed = 0
            org.fidelity_harvest = 0.0
 
        # pre-reproduction energy 
        self._energy_pre_repro = float(np.mean([o.energy for o in self.population]))
 
 
    # --------------------------- reproduction -----------------------------
    
    def interneuron_duplication(self):
        # a = self.architecture
        frac = self.jump_frac * float(np.exp(self.rng.normal(0.0, self.jump_sigma)))
        return self.architecture.Eb_for_fraction(min(frac, 0.95))


    def interneuron_loss(self):
        """
        Large-effect reverse event.

        Loss/deletion of the backend architecture returns the
        offspring to the ancestral Eb = 0 state.
        """
        return 0.0


    def mutate(self, mu, sigma, Ef, Eb):
        """
        chance mutation (set my mutation_rate) of mu, sigma, Ef and Eb
        """
        # ---- nervous system evolution ----

        if self.rng.random() < self.arch_mutation_rate:
            Ef = max(Ef + self.rng.normal(0.0, self.Ef_mutation_sd), 0.0)                
        if self.rng.random() < self.arch_mutation_rate:
            Eb = max(Eb + self.rng.normal(0.0, self.Eb_mutation_sd), 0.0)

        # macroevolution (bidirectional)
        if Eb < self.e_switch:
            # gain of backend architecture
            if self.rng.random() < self.duplication_rate:
                Eb = self.interneuron_duplication()
                self.jumps_this_year += 1
        else:
            # loss of backend architecture
            if self.rng.random() < self.loss_rate:
                Eb = self.interneuron_loss()
                self.losses_this_year += 1

        # ---- niche evolution ----

        if self.rng.random() < self.niche_mutation_rate:
            mu += self.rng.normal(0.0, self.mu_mutation_sd)

        if self.rng.random() < self.niche_mutation_rate:
            sigma *= np.exp(self.rng.normal(0.0, self.sigma_mutation_sd))

        return mu, sigma, Ef, Eb

    
    def _eligible(self):
        """
        reproduction eligibility graded by surplus (Bernoulli trail)
        """
        p = np.clip([o.energy / self.repro_scale for o in self.population], 0.0, 1.0)
        return [o for o, keep in zip(self.population, self.rng.random(len(p)) < p) if keep]


    def _inherit(self, trait1, trait2, dominance=None):
        w = self.dominance if dominance is None else dominance
        if w == 0.5:
            return 0.5 * (trait1 + trait2)                  # blending inheritance
        if self.rng.random() < 0.5:
            trait1, trait2 = trait2, trait1                 # if dominance=1; clonal inheritance 
        return w * trait1 + (1.0 - w) * trait2
    
    def reproduce(self):
        """
        checks organisms who are eligable to reproduce 
        randomly pairs them and creates offspring with averaged traits and mutation
        """
        reproductive_candidates = self._eligible()
        self._reproductive_frac = (len(reproductive_candidates) / len(self.population) if self.population else NAN)
        self.rng.shuffle(reproductive_candidates)
        offspring = []
        for i in range(0, len(reproductive_candidates) - 1, 2):
            p1, p2 = reproductive_candidates[i], reproductive_candidates[i + 1]
            p1.energy -= self.repro_cost
            p2.energy -= self.repro_cost
            niche_parent = p1 if self.rng.random() < 0.5 else p2

            mu_c = niche_parent.mu
            sigma_c = niche_parent.sigma
            # clonal inheritance   
            Ef_c    = self._inherit(p1.Ef,    p2.Ef)
            Eb_c    = self._inherit(p1.Eb,    p2.Eb)
            mu, sigma, Ef, Eb = self.mutate(mu_c, sigma_c, Ef_c, Eb_c)

            offspring.append(self.birth_organism(mu, sigma, Ef, Eb, age=0, energy=self.offspring_energy))
        self.population.extend(offspring)
        return len(offspring)

 
    # ----------------------------- mortality ------------------------------
 

    def cull(self):
        """
        age everyone; remove the old and the starved
        """
        survivors, starved, senescent = [], 0, 0
        for o in self.population:
            o.age += 1
            if o.energy <= 0.0:
                starved += 1
            elif o.age > self.max_age:
                senescent += 1
            else:
                survivors.append(o)
        self.population = survivors
 
        return starved, senescent
 
    # -------------------------------- run ---------------------------------
 

    def step_year(self):
        """
        the main loop: forage, reproduce, cull, regrow, reset organism items for next year
        record population statistics at the end of a year

        Returns: check on population extinction  
        """
        if not self.population:
            self.extinct = True
            return False
 
        self.environment.fluctuate(self.rng)
        self.forage_year()
        self.jumps_this_year = 0
        self.losses_this_year = 0

        births = self.reproduce()
        self._jumps_since_print += self.jumps_this_year
        self._losses_since_print += self.losses_this_year
        starved, senescent = self.cull()
        self.environment.regrow()

        self.sim_time += 1
        # update log
        self.record(births, starved, senescent)

        return self.check_population_status()


    def run(self, years=200, verbose_every=None, snapshot_times=None,
            progress_every=None, on_progress=None, label=""):
        """
        simulate years of evolution and record
            history: a dict of population-level time series data
            snapshots: agent-level records at specified timepoints 

        progress_every / on_progress give a live readout during long runs:
        on_progress(population, year, total_years) is called every
        progress_every years and formats whatever it likes. `label` prefixes
        the built-in verbose line so parallel runs can be told apart.
        """
        if snapshot_times is None:
            snapshot_times = set()
        else:
            snapshot_times = set(snapshot_times)

        snapshots = []

        for _ in range(int(years)):
            if not self.step_year():
                break

            if self.sim_time in snapshot_times:
                snap = self.individual_snapshot()
                n = len(snap["Ef"])
                for i in range(n):
                    snapshots.append({
                        "time": self.sim_time,
                        "agent": i,
                        "Ef": snap["Ef"][i],
                        "Eb": snap["Eb"][i],
                        "Ff": snap["Ff"][i],
                        "Fb": snap["Fb"][i],
                        "mu": snap["mu"][i],
                        "sigma": snap["sigma"][i],
                        "a_norm": snap["a_norm"][i],
                        "a_raw": snap["a_raw"][i],
                        "a_F": snap["a_F"][i],
                        "tuning_efficiency": snap["tuning_efficiency"][i],
                        "fidelity_realised": snap["fidelity_realised"][i],
                        "above_switch": snap["above_switch"][i],
                        "age": snap["age"][i],
                        "energy": snap["energy"][i],
                    })

            if (progress_every and on_progress
                    and self.sim_time % progress_every == 0):
                on_progress(self, self.sim_time, int(years))

            if verbose_every and self.sim_time % verbose_every == 0:
                L = self.log
                jumps_per_yr = self._jumps_since_print / verbose_every
                self._jumps_since_print = 0

                losses_per_year = self._losses_since_print / verbose_every
                self._losses_since_print = 0

                print(f"{label}t={self.sim_time:4d}  N={L['population_size'][-1]:4d}  "
                        f"Ef={L['Ef_mean'][-1]:6.2f}  Ef_sd={L['Ef_sd'][-1]:6.2f} Eb={L['Eb_mean'][-1]:6.2f}  Eb_sd={L['Eb_sd'][-1]:6.2f} "
                        f"Ff={L['Ff_mean'][-1]:6.2f}  Ff_sd={L['Ff_sd'][-1]:6.2f} Fb={L['Fb_mean'][-1]:6.2f}  Fb_sd={L['Fb_sd'][-1]:6.2f} "
                        f"jumps/yr={jumps_per_yr:5.2f}  losses/yr={losses_per_year:5.2f} "
                        f"sigma_mean={L['sigma'][-1]:6.2f} "
                        f"a_norm_mean={L['a_norm_mean'][-1]:+.3f}  cmt={L['committed_frac'][-1]:.2f}  "
                        f"F={L['fidelity_mean'][-1]:5.2f}  R_year={L['items_consumed_mean'][-1]:6.2f}  "
                        f"stock={L['harvestable_stock'][-1]:8.1f}", flush=True)
                
        return self.history(), snapshots    
 
    # ------------------------------ record/metrics--------------------------------

    def record(self, births=0, starved=0, senescent=0):
        """Population state at the end of a year.
 
        Traits are measured over SURVIVORS -- that is the evolved state.
        Mortality is split by cause so over-investment (starved) can be told
        apart from ordinary turnover (senescent).
        """
        log, pop = self.log, self.population
        stock = self.environment.stock
 
        log["population_size"].append(len(pop))
        log["births"].append(int(births))
        log["deaths"].append(int(starved + senescent))
        log["deaths_starved"].append(int(starved))
        log["deaths_senescent"].append(int(senescent))
 
        # take() needs stock >= 1.0, so the raw sum overstates what is edible
        log["standing_stock"].append(float(stock.sum()))
        log["harvestable_stock"].append(float(stock[stock >= 1.0].sum()))
        log["stock_depleted"].append(self._env_items_taken)
        log["energy_pre_repro"].append(self._energy_pre_repro)
        log["reproductive_frac"].append(self._reproductive_frac)
        
        if not pop:
            for key in (
                "a_norm_mean", "a_norm_sd",
                "a_raw_mean", "a_raw_sd",
                "a_F_mean", "a_F_sd",
                "committed_frac",
                "above_switch_frac",
                "Ef_mean", "Ef_sd",
                "Eb_mean", "Eb_sd",
                "Ff_mean", "Ff_sd",
                "Fb_mean", "Fb_sd",
                "neural_spend_mean",
                "jumps", "losses",
                "items_consumed_mean", "items_consumed_sd",
                "fidelity_mean", "fidelity_sd",
                "tuning_efficiency_mean", "tuning_efficiency_sd",
                "fidelity_realised_mean", "fidelity_realised_sd",
                "rho_sigma_Ef", "rho_sigma_Eb",
                "rho_sigma_a_norm", "rho_sigma_tuning",
                "rho_absmu_Ef", "rho_absmu_Eb",
                "energy", "age",
                "mu", "mu_sd", "sigma", "sigma_sd"
            ):
                log[key].append(NAN)
            return 
        
        n = len(pop)
        Ef = np.fromiter((o.Ef for o in pop), float, n)
        Eb = np.fromiter((o.Eb for o in pop), float, n)
        log["above_switch_frac"].append(float(np.mean(Eb > self.e_switch)))
        Ff = np.fromiter((o.Ff for o in pop), float, n)
        Fb = np.fromiter((o.Fb for o in pop), float, n)
        a_norm = np.fromiter((o.a_norm for o in pop), float, n)
        a_raw = np.fromiter((o.a_raw for o in pop), float, n)
        a_F = np.fromiter((o.a_F for o in pop), float, n)

        F = np.fromiter((o.Fidelity for o in pop), float, n)

        mu = np.fromiter((o.mu for o in pop), float, n)
        sigma = np.fromiter((o.sigma for o in pop), float, n)
 
        # a is NaN for agents that bought nothing effective -- that is a result,
        # not missing data, so report how many there are alongside the mean
        committed = np.isfinite(a_norm)
        log["committed_frac"].append(float(committed.mean()))

        log["a_norm_mean"].append(float(np.nanmean(a_norm)) if committed.any() else NAN)
        log["a_norm_sd"].append(float(np.nanstd(a_norm)) if committed.any() else NAN)
 
        log["a_raw_mean"].append(float(np.nanmean(a_raw)) if committed.any() else NAN)
        log["a_raw_sd"].append(float(np.nanstd(a_raw)) if committed.any() else NAN)
 
        log["a_F_mean"].append(float(np.nanmean(a_F)) if committed.any() else NAN)
        log["a_F_sd"].append(float(np.nanstd(a_F)) if committed.any() else NAN)
 
        log["Ef_mean"].append(float(Ef.mean()))
        log["Ef_sd"].append(float(Ef.std()))
        log["Eb_mean"].append(float(Eb.mean()))
        log["Eb_sd"].append(float(Eb.std()))
        log["neural_spend_mean"].append(float((Ef + Eb).mean()))

        log["jumps"].append(int(self.jumps_this_year))
        log["losses"].append(int(self.losses_this_year))
        
        log["Ff_mean"].append(float(Ff.mean()))
        log["Ff_sd"].append(float(Ff.std()))
        log["Fb_mean"].append(float(Fb.mean()))
        log["Fb_sd"].append(float(Fb.std()))

        log["fidelity_mean"].append(float(F.mean()))
        log["fidelity_sd"].append(float(F.std()))

        tuning = np.fromiter((o.tuning_efficiency for o in pop), float, n)
        realised = np.fromiter((o.fidelity_realised for o in pop), float, n)
        log["tuning_efficiency_mean"].append(float(np.nanmean(tuning)) if np.isfinite(tuning).any() else NAN)
        log["tuning_efficiency_sd"].append(float(np.nanstd(tuning)) if np.isfinite(tuning).any() else NAN)
        log["fidelity_realised_mean"].append(float(np.nanmean(realised)) if np.isfinite(realised).any() else NAN)
        log["fidelity_realised_sd"].append(float(np.nanstd(realised)) if np.isfinite(realised).any() else NAN)

        # ---- the hypothesis, measured every year over the standing population ----
        # front-loaded specialists  -> rho(sigma, Ef) < 0
        # back-loaded generalists   -> rho(sigma, Eb) > 0
        abs_mu_offset = np.abs(mu - self.environment.env_mu)
        log["rho_sigma_Ef"].append(spearman(sigma, Ef))
        log["rho_sigma_Eb"].append(spearman(sigma, Eb))
        log["rho_sigma_a_norm"].append(spearman(sigma, a_norm))
        log["rho_sigma_tuning"].append(spearman(sigma, tuning))
        log["rho_absmu_Ef"].append(spearman(abs_mu_offset, Ef))
        log["rho_absmu_Eb"].append(spearman(abs_mu_offset, Eb))
 
        items = self._org_items_consumed
        log["items_consumed_mean"].append(float(items.mean()) if items.size else NAN)
        log["items_consumed_sd"].append(float(items.std()) if items.size else NAN)
 
        log["energy"].append(float(np.mean([o.energy for o in pop])))
        log["age"].append(float(np.mean([o.age for o in pop])))
        log["mu"].append(float(mu.mean()))
        log["mu_sd"].append(float(mu.std()))
        log["sigma"].append(float(sigma.mean()))
        log["sigma_sd"].append(float(sigma.std()))

 
    def check_population_status(self):
        """
        check if the population is extinct or runaway
        """
        if not self.population:
            self.extinct = True
            return False
        if len(self.population) > self.runaway_guard:
            self.runaway = True
            return False
        return True

    
    def history(self):
        lengths = {k: len(v) for k, v in self.log.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"ragged log: {lengths}")
        h = {k: np.array(v) for k, v in self.log.items()}
        h["time"] = np.arange(len(self.log["population_size"]))
        h["extinct"], h["runaway"] = self.extinct, self.runaway
        return h


    
    def individual_snapshot(self):
        """
        Return one row of individual-level state per surviving organism.
        """
        return {
            "Ef": np.array([o.Ef for o in self.population], dtype=float),
            "Eb": np.array([o.Eb for o in self.population], dtype=float),

            "Ff": np.array([o.Ff for o in self.population], dtype=float),
            "Fb": np.array([o.Fb for o in self.population], dtype=float),
            "mu": np.array([o.mu for o in self.population], dtype=float),
            "sigma": np.array([o.sigma for o in self.population],dtype=float),

            "a_norm": np.array([o.a_norm for o in self.population], dtype=float),
            "a_raw": np.array([o.a_raw for o in self.population], dtype=float),
            "a_F": np.array([o.a_F for o in self.population], dtype=float),

            "tuning_efficiency": np.array(
                [o.tuning_efficiency for o in self.population],
                dtype=float
            ),
            "fidelity_realised": np.array(
                [o.fidelity_realised for o in self.population],
                dtype=float
            ),

            "above_switch": np.array(
                [o.Eb > self.e_switch for o in self.population],
                dtype=bool
            ),

            "age": np.array([o.age for o in self.population], dtype=int),
            "energy": np.array([o.energy for o in self.population], dtype=float),
        }