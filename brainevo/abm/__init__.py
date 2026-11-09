# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm                                                  ║
# ║  « the FEBE agent-based model »                                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Front-End / Back-End Evolution: organisms invest heritable      ║
# ║  energy in peripheral and central processing, forage on a        ║
# ║  Gaussian resource axis, reproduce by surplus and die by age or  ║
# ║  starvation.  Model of Dodd (2026), refactored for publication.  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""FEBE agent-based model."""
from __future__ import annotations

from brainevo.abm.cost_laws import Architecture, CostLawParams
from brainevo.abm.energetics import ATPArchitecture, ATPCostLawParams, CapsuleLaw, GroundedConstant
from brainevo.abm.environment import Environment, EnvironmentParams
from brainevo.abm.organism import Organism
from brainevo.abm.population import (
    InitialisationParams,
    LifeHistoryParams,
    MutationParams,
    Population,
)

__all__ = [
    "ATPArchitecture",
    "ATPCostLawParams",
    "Architecture",
    "CapsuleLaw",
    "CostLawParams",
    "Environment",
    "EnvironmentParams",
    "GroundedConstant",
    "InitialisationParams",
    "LifeHistoryParams",
    "MutationParams",
    "Organism",
    "Population",
]
