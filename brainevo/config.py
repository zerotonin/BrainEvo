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


PROFILE_ENV_VAR = "BRAINEVO_PROFILE"


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


def load_profile(name: str | None = None) -> dict[str, str]:
    """Return one named profile from local_paths.json (default: active_profile)."""
    if not LOCAL_PATHS.exists():
        raise MissingLocalPathsError(f"{LOCAL_PATHS.name} not found; copy {LOCAL_PATHS_TEMPLATE.name} and fill it in.")
    data = json.loads(LOCAL_PATHS.read_text(encoding="utf-8"))
    name = name or os.environ.get(PROFILE_ENV_VAR) or data.get("active_profile", "local")
    try:
        return dict(data["profiles"][name])
    except KeyError as exc:
        raise MissingLocalPathsError(f"profile {name!r} not in {LOCAL_PATHS.name}") from exc


def export_shell(profile: str | None = None) -> str:
    """Shell ``export`` lines for SLURM wrappers (see slurm/load_paths.sh)."""
    prof = load_profile(profile)
    mapping = {
        "results_root": ENV_VAR,
        "code_root": "BRAINEVO_CODE_ROOT",
        "conda_setup": "BRAINEVO_CONDA_SETUP",
        "conda_env": "BRAINEVO_CONDA_ENV",
        "slurm_account": "SBATCH_ACCOUNT",
        "slurm_partition": "SBATCH_PARTITION",
    }
    lines = []
    for key, var in mapping.items():
        value = prof.get(key)
        if value and not str(value).startswith("<"):
            lines.append(f"export {var}={str(Path(value).expanduser()) if 'root' in key or 'setup' in key else value}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if "--export" in sys.argv:
        idx = sys.argv.index("--export")
        prof = sys.argv[idx + 1] if len(sys.argv) > idx + 1 and not sys.argv[idx + 1].startswith("-") else None
        print(export_shell(prof))
    else:
        print(resolve_results_root())
