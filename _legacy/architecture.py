import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os
STYLE = os.path.join(os.path.dirname(__file__), "..", "plotting", "figure.mplstyle")


class Architecture:
    """
    Nervous system capability bought by metabolic spending, split front-end / back-end.
    E_f and E_b are independent investment energies:

                    F(E) = softcap[B * log2(1 + SNR * E)]
                    where for the back-end E is measured after e_switch
                    B = Fmax / k

    PARAMETERS
    - Fmax_f:       front-end ceiling (bits). Does not change e_saturation, only lifts ceiling
    - M:            back-end ceiling ratio, M = Fmax_b / Fmax_f
    - k:            fidelity per bandwidth unit, shared by both ends;
                    B_front = Fmax_f / k
    - back_rate:    rate multiplier on the back-end, in (0, 1).
                    B_back = back_rate * Fmax_b / k

                    back_rate = 1      the ceiling is raised at fixed saturation
                                       energy, exactly as raising Fmax_f does.
                                       Nothing else modifies the back-end rate.
                    back_rate < 1      same ceiling, slower approach to it, so
                                       saturation costs more energy.
                    back_rate = 1 / M  B_back = Fmax_f / k, i.e. the back-end
                                       has the same absolute slope as the
                                       front-end and the raised ceiling is paid
                                       for entirely in energy.
    - P:            fraction of front-end saturation energy at which the
                    back-end switches on
    - sharpness:    soft-cap sharpness
    - q: (0.95)     definition of where "saturated" sits
    - SNR:          signal-to-noise parameter used to define energy-information
                    standard conversion

    Energy is in arbitrary units.
    """

    def __init__(self, Fmax_f, k, SNR, M, P, back_rate=1.0, sharpness=5, q=0.95,
                 tuning_width=None, tuning_mode="absolute", tuning_floor=0.0):
        self.Fmax_f = Fmax_f
        self.M = M
        self.Fmax_b = M * Fmax_f

        self.k = float(k)
        self.back_rate = float(back_rate)
        if self.back_rate <= 0.0:
            raise ValueError("back_rate must be > 0")

        self.B_front = self.Fmax_f / self.k
        self.B_back = self.back_rate * self.Fmax_b / self.k

        self.SNR = SNR
        self.q = q
        self.sharpness = sharpness

        self.P = P
        self.e_switch = (self.P * self.E_f_saturation)
        self.switch = (np.log2(1.0 + self.SNR * self.e_switch) / self.k)

        if tuning_mode not in ("absolute", "relative"):
            raise ValueError("tuning_mode must be 'absolute' or 'relative'")
        self.tuning_mode = tuning_mode
        self.tuning_width = None if tuning_width is None else float(tuning_width)
        self.tuning_floor = float(np.clip(tuning_floor, 0.0, 1.0))
        self.tuned = self.tuning_width is not None
        if self.tuned and self.tuning_width <= 0.0:
            raise ValueError("tuning_width must be > 0 (or None for an untuned front-end)")

    # ----- front-end tuning -----

    def front_gain(self, resource_axis, mu, sigma):
        """
        Fraction of front-end capability F_f that is realised on an item of
        each resource type, given an organism centred on mu with breadth sigma.

        Fb is paid out on every item regardless of type. Only the front-end is tuned.
        """
        axis = np.asarray(resource_axis, dtype=float)
        if not self.tuned:
            return np.ones_like(axis)

        w = self.tuning_width * float(sigma) if self.tuning_mode == "relative" else self.tuning_width
        w = max(float(w), 1e-9)
        g = np.exp(-0.5 * ((axis - float(mu)) / w) ** 2)
        return self.tuning_floor + (1.0 - self.tuning_floor) * g


    # ----- energy -> information calculations -----

    def _E_front(self, I):
        return (2.0**(I / self.B_front) - 1.0) / self.SNR

    def _E_back(self, I):
        return (2.0**(I / self.B_back) - 1.0) / self.SNR

    def _raw_for_fraction(self, fraction, ceiling):
        # the raw value (before soft-cap ceiling) of F needed to acheive F=faction*ceiling
        return ceiling * fraction/(1.0 - fraction**self.sharpness)** (1.0 / self.sharpness)

    @property
    def E_f_saturation(self):
        return self._E_front(self._raw_for_fraction(self.q, self.Fmax_f))

    @property
    def E_b_saturation(self):
        # back-end investment required to reach saturation including the switch-on energy
        return self.e_switch + self._E_back(self._raw_for_fraction(self.q, self.Fmax_b))
    
    def Eb_for_fraction(self, fraction):
        """Back-end investment (including e_switch) that buys some fraction of Fmax_b."""
        fraction = float(np.clip(fraction, 1e-9, 1.0 - 1e-9))
        return self.e_switch + self._E_back(self._raw_for_fraction(fraction, self.Fmax_b))


    # ----- curvature -----

    def _soft_cap(self, raw, ceiling):
        # soft-cap function imposes a ceiling on the raw capability (F)
        return raw / (1.0 + (raw / ceiling)**self.sharpness)**(1.0 / self.sharpness)

    def f_capability(self, E_front):
        # F_front = f_capability(E) = B * log2(1 + SNR * E) with soft-cap ceiling
        E = np.maximum(np.asarray(E_front, dtype=float), 0.0)
        raw = self.B_front * np.log1p(E * self.SNR) / np.log(2.0)
        return self._soft_cap(raw, self.Fmax_f)

    def b_capability(self, E_back):
        # F_back = b_capability(E) = B * log2(1 + SNR * E) after the switch-on energy, with soft-cap ceiling
        E = np.maximum(np.asarray(E_back, dtype=float) - self.e_switch, 0.0)
        raw = self.B_back * np.log1p(E * self.SNR) / np.log(2.0)
        return self._soft_cap(raw, self.Fmax_b)

    def capability(self, E_front, E_back):
        # total capability is summed front/back fidelity
        return self.f_capability(E_front) + self.b_capability(E_back)




if __name__ == '__main__':
    Fmax_f = 4.0
    P = 0.5
    k = 3
    SNR = 14.44
    M=1.5

    Fmax_f=0.8
    M=3.0
    p=3.75                     
    k=3
    back_rate=0.9

    arch=Architecture(
        Fmax_f=Fmax_f,
        M=M,
        P=p,
        k=k,
        SNR=SNR,
        back_rate=1)
    print(arch.E_f_saturation)
    print(arch.E_b_saturation)
