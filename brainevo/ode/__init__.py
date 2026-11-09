# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — ode                                                  ║
# ║  « the FEBE model as adaptive dynamics »                         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Under uniform foraging the only ecological feedback is the      ║
# ║  scalar hit probability, so evolution maximises fidelity per     ║
# ║  unit cost, Q(x) = F(x) / C(x).  This sub-package computes the   ║
# ║  landscape, its singular strategies, the canonical equation and  ║
# ║  the bifurcation diagram in the overhead P and switch width w.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Mean-field / adaptive-dynamics reduction of FEBE."""
from __future__ import annotations

from brainevo.ode.canonical import integrate_canonical, ramp_hysteresis
from brainevo.ode.meanfield import MeanField, SingularStrategy
from brainevo.ode.niche import NicheMeanField, ResidentEquilibrium, SingularClassification, Strategy

__all__ = [
    "MeanField", "NicheMeanField", "ResidentEquilibrium", "SingularClassification", "SingularStrategy",
    "Strategy", "integrate_canonical", "ramp_hysteresis",
]
