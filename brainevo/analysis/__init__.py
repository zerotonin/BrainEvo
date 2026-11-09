# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — analysis                                             ║
# ║  « from sweep directories to publication statistics »            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Reads the run directories written by brainevo-abm sweep, compute║
# ║  establishment probabilities, waiting-time hazards, oxygen-cap   ║
# ║  tracking and niche bimodality, and draws the figures.           ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

from brainevo.analysis.bimodality import bimodality_table, fit_two_component
from brainevo.analysis.establishment import establishment_cells, exponential_hazard, hazard_vs_deficit
from brainevo.analysis.loading import architecture_from_params, load_runs, load_snapshots

__all__ = [
    "architecture_from_params",
    "bimodality_table",
    "establishment_cells",
    "exponential_hazard",
    "fit_two_component",
    "hazard_vs_deficit",
    "load_runs",
    "load_snapshots",
]
