# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — config                                               ║
# ║  « machine-specific paths live outside the repo »                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Resolution order for the results root:                          ║
# ║    1. environment variable BRAINEVO_RESULTS_ROOT                 ║
# ║    2. "results_root" in local_paths.json (gitignored)            ║
# ║    3. an explicit --out on the command line (handled by the CLI) ║
# ║  Missing configuration fails loudly and names the template.      ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VAR = "BRAINEVO_RESULTS_ROOT"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATHS = REPO_ROOT / "local_paths.json"
LOCAL_PATHS_TEMPLATE = REPO_ROOT / "local_paths.template.json"


class MissingLocalPathsError(RuntimeError):
    """Raised when no results root can be resolved."""


def resolve_results_root(explicit: Path | None = None) -> Path:
    """Return the directory that receives simulation output."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if ENV_VAR in os.environ:
        return Path(os.environ[ENV_VAR]).expanduser().resolve()
    if LOCAL_PATHS.exists():
        data = json.loads(LOCAL_PATHS.read_text(encoding="utf-8"))
        profile = data.get("profiles", {}).get(data.get("active_profile", "local"), {})
        root = profile.get("results_root")
        if root:
            return Path(root).expanduser().resolve()
    raise MissingLocalPathsError(
        f"No results root configured. Pass --out, set ${ENV_VAR}, or copy "
        f"{LOCAL_PATHS_TEMPLATE.name} to {LOCAL_PATHS.name} and fill in 'results_root'."
    )
