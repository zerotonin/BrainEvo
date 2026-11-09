# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — abm.cli                                              ║
# ║  « run a preset, or sweep a grid across seeds »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  brainevo-abm run   --preset exp2-tuning --seed 1 --years 4000   ║
# ║  brainevo-abm sweep --preset exp1-cost-sweep                     ║
# ║                     --grid cost__M=1,1.5,2 cost__Fmax_f=0.5,1.2  ║
# ║                     --replicates 20 --workers 8                  ║
# ║  Every run writes history.csv, snapshots.csv, params.json and a  ║
# ║  trajectory figure into <out>/<preset>/<tag>/.                   ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from brainevo import __version__
from brainevo.abm.cost_laws import Architecture
from brainevo.abm.energetics import ATPArchitecture
from brainevo.abm.environment import Environment
from brainevo.abm.population import Population
from brainevo.abm.presets import PRESETS, Preset
from brainevo.abm.viz import plot_trajectories
from brainevo.config import MissingLocalPathsError, resolve_results_root
from brainevo.figures import apply_style, save_figure

TAIL_YEARS = 300
"""Window at the end of a run over which end-point means are taken."""


# ┌────────────────────────────────────────────────────────────┐
# │ Orchestration                                              │
# └────────────────────────────────────────────────────────────┘

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        out_root = resolve_results_root(args.out)
    except MissingLocalPathsError as err:
        parser.error(str(err))
    preset = PRESETS[args.preset].with_overrides(**parse_overrides(args.set))
    if args.years:
        preset = preset.__class__(**{**preset.__dict__, "years": args.years})

    if args.command == "run":
        result = run_one(preset, args.seed, out_root / preset.name / f"seed{args.seed:03d}",
                         snapshot_every=args.snapshot_every, progress=True)
        print(json.dumps(result, indent=2))
        return 0
    return run_sweep(preset, args, out_root)


def run_sweep(preset: Preset, args: argparse.Namespace, out_root: Path) -> int:
    grid = parse_grid(args.grid)
    keys = list(grid)
    jobs = []
    for combo in product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, combo, strict=True))
        tag = "_".join(f"{k.split('__')[-1]}{v:g}" for k, v in overrides.items()) or "base"
        for rep in range(args.replicates):
            seed = args.seed + rep
            jobs.append((preset.with_overrides(**overrides), seed, out_root / preset.name / tag / f"seed{seed:03d}"))

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, p, s, d, args.snapshot_every, False) for p, s, d in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"sweep {preset.name}"):
            rows.append(fut.result())
    summary = pd.DataFrame(rows)
    summary_path = out_root / preset.name / "sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path} ({len(summary)} runs)")
    return 0


# ┌────────────────────────────────────────────────────────────┐
# │ One run  « module-level so ProcessPoolExecutor can pickle »│
# └────────────────────────────────────────────────────────────┘

def run_one(preset: Preset, seed: int, out_dir: Path, snapshot_every: int, progress: bool) -> dict[str, object]:
    arch = preset.build_architecture()
    env = Environment(preset.environment)
    pop = Population(env, arch, preset.life, preset.mutation, preset.init, rng=seed)
    snapshot_times = set(range(snapshot_every, preset.years + 1, snapshot_every)) if snapshot_every else set()

    bar = tqdm(total=preset.years, desc=f"{preset.name} seed {seed}", disable=not progress)

    def on_progress(_: Population, year: int, __: int) -> None:
        bar.update(100)

    history, snapshots = pop.run(preset.years, snapshot_times, progress_every=100, on_progress=on_progress)
    bar.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame({k: v for k, v in history.items() if np.ndim(v) == 1})
    hist.to_csv(out_dir / "history.csv", index=False)
    if snapshots:
        pd.DataFrame(snapshots).to_csv(out_dir / "snapshots.csv", index=False)
    params = {"preset": preset.to_dict(), "seed": seed, "brainevo_version": __version__,
              "derived": derived_quantities(arch), "extinct": bool(pop.extinct), "runaway": bool(pop.runaway)}
    (out_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    apply_style()
    fig = plot_trajectories(hist, arch)
    save_figure(fig, "trajectories", out_dir)
    return summarise(hist, arch, preset, seed)


def summarise(hist: pd.DataFrame, arch: Architecture | ATPArchitecture, preset: Preset, seed: int) -> dict[str, object]:
    tail = hist.iloc[-TAIL_YEARS:]
    established = bool((hist["above_switch_frac"] > 0.5).any())
    year_established = int(hist.index[hist["above_switch_frac"] > 0.5][0]) if established else -1
    return {
        "preset": preset.name, "seed": seed,
        **{f"cost_{k}": v for k, v in asdict(preset.atp if preset.atp is not None else preset.cost).items()},
        "N": float(tail["population_size"].mean()),
        "Ef": float(tail["Ef_mean"].mean()), "Eb": float(tail["Eb_mean"].mean()), "Ec": float(tail["Ec_mean"].mean()),
        "Ef_norm": float(tail["Ef_mean"].mean() / arch.E_f_saturation),
        "Eb_norm": float(max(tail["Eb_mean"].mean() - arch.e_switch, 0.0) / arch.E_b_climb),
        "hard_cap": float(getattr(arch, "hard_cap", float("nan"))),
        "a_norm": float(tail["a_norm_mean"].mean()),
        "sigma": float(tail["sigma"].mean()),
        "above_switch_frac": float(tail["above_switch_frac"].mean()),
        "backend_established": established, "year_established": year_established,
        "extinct": bool(hist["population_size"].iloc[-1] == 0),
    }


def derived_quantities(arch: Architecture | ATPArchitecture) -> dict[str, float]:
    out = {"E_f_saturation": arch.E_f_saturation, "e_switch": arch.e_switch,
           "E_b_saturation": arch.E_b_saturation, "Fmax_b": arch.Fmax_b, "P": arch.P}
    for name in ("B_front", "B_back"):
        if hasattr(arch, name):
            out[name] = getattr(arch, name)
    return out


# ┌────────────────────────────────────────────────────────────┐
# │ Argument parsing                                           │
# └────────────────────────────────────────────────────────────┘

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brainevo-abm", description="FEBE agent-based model runner.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--preset", choices=sorted(PRESETS), required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--years", type=int, default=None, help="override the preset's duration")
        p.add_argument("--out", type=Path, default=None, help="results root (else local_paths.json)")
        p.add_argument("--snapshot-every", type=int, default=1000)
        p.add_argument("--set", nargs="*", default=[], metavar="group__field=value",
                       help="single-value overrides, e.g. cost__P=2.5")
    sweep = sub.choices["sweep"]
    sweep.add_argument("--grid", nargs="*", default=[], metavar="group__field=v1,v2,...")
    sweep.add_argument("--replicates", type=int, default=1)
    sweep.add_argument("--workers", type=int, default=None)
    return parser


def parse_overrides(items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = _coerce(value)
    return out


def parse_grid(items: list[str]) -> dict[str, list[object]]:
    grid: dict[str, list[object]] = {}
    for item in items:
        key, _, values = item.partition("=")
        grid[key] = [_coerce(v) for v in values.split(",")]
    return grid


def _coerce(text: str) -> object:
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if lowered in ("inf", "infinity"):
        return float("inf")
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


if __name__ == "__main__":
    sys.exit(main())
