# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — package                                              ║
# ║  « front-loaded or back-loaded: how sensory architecture evolves ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Eco-evolutionary agent-based model (FEBE) and its adaptive-     ║
# ║  dynamics reduction for the evolution of front- vs back-loaded   ║
# ║  sensory processing under energetic and ecological constraints.  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""BrainEvo: evolution of sensory-processing architecture.

Sub-packages:
    abm  -- the FEBE agent-based model (organisms, resources, life cycle)
    ode  -- mean-field / adaptive-dynamics reduction of the same model
"""
from __future__ import annotations

try:
    from brainevo._version import __version__
except ImportError:  # editable install without a tag
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
