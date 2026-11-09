import numpy as np

from .architecture import Architecture
from .environment import Environment

NAN = float("nan")

class Organism:
    """One agent: a niche position (mu) and width (sigma), plus state."""

    def __init__(self, environment, architecture, Ef, Eb, mu, sigma, fidelity_floor = 1.0, basal_metabolism=5.0,
                 age=0, initial_energy=10.0):
        self.environment = environment
        self.architecture = architecture

        # evolvable variables 
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.Ef = float(Ef)
        self.Eb = float(Eb)

        # trait measures 


        self.Fb = self.architecture.b_capability(self.Eb)
        self.Ff = self.architecture.f_capability(self.Ef)

        # drift and novelty discounts of fidelity 
        # self.front_discount = 1.0 - environment.between_gen_novelty
        # self.back_discount = 1.0 - environment.within_lifetime_drift
        # self.Ff_effective = self.front_discount * self.Ff
        # self.Fb_effective = self.back_discount * self.Fb

        # potential fidelity: what the nervous system is worth on an item that
        # the front-end is perfectly tuned to. Kept for comparability with the
        # untuned model.
        self.fidelity_floor = float(fidelity_floor)
        self.Fidelity = self.fidelity_floor + self.Fb + self.Ff


        self.a_norm = self.architecture_index_NORM()
        self.a_raw = self.architecture_index_RAW()
        self.a_F = self.architecture_index_F()
                
        self.resource_weights = environment.resource_weights(
            self.mu,
            self.sigma
        )

        self.search_probs = environment.preference_distribution(
            self.mu,
            self.sigma
        )

        # ---- tuned front-end ----
        # front_gain[z] in [0, 1] is the share of F_f that an item of type z
        # actually delivers; F_b is paid on every item. So per-item fidelity is
        # type-dependent and the front-end only pays off for a forager whose
        # draws land near its own mu.
        self.front_gain = self.architecture.front_gain(environment.resource_axis, self.mu, self.sigma)
        self.item_fidelity = (
            self.fidelity_floor
            + self.Fb
            + self.Ff * self.front_gain
        )

        # expected share of the front-end realised over this organism's own
        # search distribution -- the specialist/generalist coupling in one number
        if self.search_probs is None:
            self.tuning_efficiency = NAN
            self.expected_fidelity = NAN
        else:
            self.tuning_efficiency = float(self.search_probs @ self.front_gain)
            self.expected_fidelity = (
                self.fidelity_floor
                + self.Fb
                + self.Ff * self.tuning_efficiency
            )

        # fixed variables 
        self.basal_metabolism = float(basal_metabolism)

        # running state variables 
        self.items_consumed = 0 # per-year items consumed
        # per-year fidelity harvest: sum of item_fidelity over items actually
        # taken. Energy gain is energy_per_item * fidelity_harvest.
        self.fidelity_harvest = 0.0
        # realised mean per-item fidelity over the last completed year
        self.fidelity_realised = NAN
        self.age = int(age)
        self.energy = float(initial_energy) # running energy


    def architecture_index_NORM(self, eps=1e-12):
        """
        Per-agent architecture, a, measured as relative front-end and back-end investment [-1, 1].
        index is normalised by saturating energy values and back end investment consitered to start after switch-on
        Ef' = Ef/Esat_f,	 Eb' = max(Eb - eswitch, 0)/Esat_b-e_switch 
        """
        arch = self.architecture
        Ef, Eb = self.Ef, max(self.Eb - arch.e_switch, 0.0)
 
        Ef = Ef / arch.E_f_saturation
        Eb = Eb / (arch.E_b_saturation - arch.e_switch)
 
        total = Ef + Eb
        return (Eb - Ef) / total if total > eps else NAN

    def architecture_index_RAW(self, eps=1e-12):
        """
        Per-agent architecture, a, measured as raw spending difference [-1, 1].
        """
        arch = self.architecture
        Ef, Eb = self.Ef, self.Eb
        total = Ef + Eb
        return (Eb - Ef) / total if total > eps else NAN

    def architecture_index_F(self, eps=1e-12):
        """
        Per-agent architecture, a, measured as raw capabiltiy difference [-1, 1].
        """
        arch = self.architecture
        Ff = self.Ff / arch.Fmax_f
        Fb = self.Fb / arch.Fmax_b
        total = Ff + Fb
        return (Fb - Ff) / total if total > eps else NAN

    # def __repr__(self):
    #     return (f"Organism(mu={self.mu:.2f}, sigma={self.sigma:.2f}, "
    #             f"age={self.age}, Ef={self.Ef:.2f}, Eb={self.Eb:.2f}, "
    #             f"a={self.a}. "
    #            f"Fidelity={self.Fidelity:.2f}, energy={self.energy_y:.2f})")


if __name__ == "__main__":
    env = Environment(n_resources=101, env_sigma=15.0,
                      capacity_peak=100.0, regrowth_rate=1.0, energy_per_item=1.0)

    arch = Architecture()
    print(f"e_switch={arch.e_switch:.3f}  E_f_sat={arch.E_f_saturation:.3f}  "
          f"E_b_sat={arch.E_b_saturation:.3f}\n")
 
    Eb_half = arch.e_switch + 0.5 * (arch.E_b_saturation - arch.e_switch)
    cases = [("nothing",           0.0,                  0.0),
             ("front only",        arch.E_f_saturation,  0.0),
             ("back only",         0.0,                  arch.E_b_saturation),
             ("sub-threshold back", 0.0,                 0.5 * arch.e_switch),
             ("front + dead back", arch.E_f_saturation,  0.5 * arch.e_switch),
             ("both saturated",    arch.E_f_saturation,  arch.E_b_saturation),
             ("half and half",     0.5 * arch.E_f_saturation, Eb_half)]
    print("NORM, RAW, F")
    for label, Ef, Eb in cases:
        o = Organism(env, arch, Ef=Ef, Eb=Eb, mu=50.0, sigma=10.0)
        print(f"{label}: {o.architecture_index_NORM}, {o.architecture_index_RAW}, {o.architecture_index_F}")


 