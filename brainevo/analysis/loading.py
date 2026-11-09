# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — analysis.loading                                     ║
# ║  « one row per run, whatever the sweep »                         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Walks <root>/<sweep>/<preset>/<tag>/seedNNN/ and flattens params║
# ║  into override-style columns (cost__P, atp__M, capsule__pO2_kPa, ║
# ║  init__arch_mode ...) next to the tail statistics of history.csv.║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from brainevo.abm.cost_laws import Architecture
from brainevo.abm.energetics import ATPArchitecture
from brainevo.abm.presets import Preset

TAIL_YEARS: int = 500
ESTABLISHED_FRAC: float = 0.5


def load_runs(root: Path, sweep: str | None = None, preset: str | None = None) -> pd.DataFrame:
    """Collect every run under ``root`` into one table.

    Args:
        root:   Results root (the directory holding sweep sub-directories).
        sweep:  Restrict to one sweep directory name.
        preset: Restrict to one preset name.
    """
    rows = [_run_row(p) for p in sorted(_iter_params(Path(root), sweep, preset))]
    if not rows:
        raise FileNotFoundError(f"no runs under {root} (sweep={sweep}, preset={preset})")
    return pd.DataFrame(rows)


def load_snapshots(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "snapshots.csv")


def load_params(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "params.json").read_text(encoding="utf-8"))


def preset_from_params(params: dict[str, Any]) -> Preset:
    return Preset.from_dict(params["preset"])


def architecture_from_params(params: dict[str, Any]) -> Architecture | ATPArchitecture:
    return preset_from_params(params).build_architecture()


def flatten_preset(preset: dict[str, Any]) -> dict[str, Any]:
    """``{"cost": {"P": 2}, "atp": {"capsule": {"pO2_kPa": 12}}}`` -> ``{"cost__P": 2, "capsule__pO2_kPa": 12}``."""
    flat: dict[str, Any] = {}
    for group in ("cost", "atp", "environment", "life", "mutation", "init"):
        values = preset.get(group)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, dict):  # nested capsule law
                flat.update({f"{key}__{k}": v for k, v in value.items()})
            else:
                flat[f"{group}__{key}"] = value
    return flat


def _iter_params(root: Path, sweep: str | None, preset: str | None):
    pattern = f"{sweep or '*'}/{preset or '*'}/*/seed*/params.json"
    yield from root.glob(pattern)


def _run_row(params_path: Path) -> dict[str, Any]:
    run_dir = params_path.parent
    params = json.loads(params_path.read_text(encoding="utf-8"))
    hist = pd.read_csv(run_dir / "history.csv")
    tail = hist.iloc[-TAIL_YEARS:]
    above = hist["above_switch_frac"] > ESTABLISHED_FRAC
    established = bool(above.any())
    row: dict[str, Any] = {
        "sweep": run_dir.parents[2].name, "preset": params["preset"]["name"], "tag": run_dir.parent.name,
        "seed": int(params["seed"]), "run_dir": str(run_dir), "years": int(params["preset"]["years"]),
        "years_run": int(len(hist)), "extinct": bool(params.get("extinct", False)),
        "established": established, "year_established": int(hist.index[above][0]) + 1 if established else -1,
        "retained": bool(tail["above_switch_frac"].mean() > ESTABLISHED_FRAC),
        "N": float(tail["population_size"].mean()),
        "Ef": float(tail["Ef_mean"].mean()), "Eb": float(tail["Eb_mean"].mean()),
        "Ec": float(tail["Ec_mean"].mean()) if "Ec_mean" in tail else 0.0,
        "Ff": float(tail["Ff_mean"].mean()), "Fb": float(tail["Fb_mean"].mean()),
        "sigma": float(tail["sigma"].mean()), "sigma_sd": float(tail["sigma_sd"].mean()),
        "above_switch_frac": float(tail["above_switch_frac"].mean()),
        "has_snapshots": (run_dir / "snapshots.csv").exists(),
    }
    row.update({f"derived__{k}": v for k, v in params.get("derived", {}).items()})
    row.update(flatten_preset(params["preset"]))
    return row
