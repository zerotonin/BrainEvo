# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — analysis.cli                                         ║
# ║  « brainevo-analyse: from results root to figures »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║    brainevo-analyse thesis    --results ROOT --out DIR           ║
# ║    brainevo-analyse grounded  --results ROOT --out DIR           ║
# ║    brainevo-analyse branching --results ROOT --out DIR           ║
# ║    brainevo-analyse all       --results ROOT --out DIR           ║
# ║  Each command writes its tables (CSV) and figures (SVG+PNG+CSV). ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from brainevo.analysis.bimodality import bimodality_table
from brainevo.analysis.establishment import establishment_cells, hazard_vs_deficit
from brainevo.analysis.loading import architecture_from_params, load_params, load_runs, load_snapshots
from brainevo.analysis.stats import across_levels, bimodal_fraction, half_establishment, naive_vs_seeded, two_sample
from brainevo.analysis.viz import (
    plot_bimodal_fraction,
    plot_grid_maps,
    plot_hazard_vs_deficit,
    plot_oxygen,
    plot_sensitivity,
    plot_sigma_through_time,
    plot_taxon_endstates,
    plot_taxon_trajectories,
)
from brainevo.figures import apply_style, save_figure
from brainevo.ode.meanfield import MeanField

GROUNDED_SENSITIVITY: list[str] = [
    "atp__min_circuit", "atp__back_share", "atp__front_share", "atp__front_exponent",
    "atp__wiring_exponent", "atp__half_saturation", "atp__switch_steepness",
]
CAPSULE_SENSITIVITY: list[str] = ["capsule__transport", "capsule__build_years", "capsule__tracheal_exponent"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commands = {"thesis": analyse_thesis, "grounded": analyse_grounded, "branching": analyse_branching,
                "taxa": analyse_taxa, "ablation": analyse_ablation}
    todo = list(commands) if args.command == "all" else [args.command]
    for name in todo:
        try:
            commands[name](Path(args.results), out / name, sweep=name if args.sweep is None else args.sweep)
        except FileNotFoundError as exc:
            print(f"[{name}] skipped: {exc}", file=sys.stderr)
    return 0


# ┌────────────────────────────────────────────────────────────┐
# │ Thesis law  « accessibility, not affordability »           │
# └────────────────────────────────────────────────────────────┘

def analyse_thesis(results: Path, out: Path, sweep: str = "thesis") -> None:
    runs = load_runs(results, sweep=sweep)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    grid = runs[runs["preset"] == "exp1-cost-sweep"]
    if len(grid):
        for P, sub in grid.groupby("cost__P"):
            cells = establishment_cells(sub, ["cost__Fmax_f", "cost__M"])
            cells, fit = hazard_vs_deficit(cells)
            cells.to_csv(out / f"cells_P{P:g}.csv", index=False)
            save_figure(plot_grid_maps(cells, "cost__Fmax_f", "cost__M", title=f"thesis law, P = {P:g}"),
                        f"grid_maps_P{P:g}", out, cells)
            tests = naive_vs_seeded(sub, ["cost__Fmax_f", "cost__M"])
            tests.to_csv(out / f"naive_vs_seeded_P{P:g}.csv", index=False)
            save_figure(plot_hazard_vs_deficit(cells, fit.slope, fit.intercept), f"hazard_vs_deficit_P{P:g}", out,
                        cells)
            (out / f"hazard_fit_P{P:g}.json").write_text(json.dumps(fit.__dict__, indent=2), encoding="utf-8")
    overhead = runs[runs["preset"] == "exp1-overhead-sweep"]
    if len(overhead) and overhead["cost__P"].nunique() > 1:
        naive = overhead[overhead["init__arch_mode"] == "naive"]
        cells = establishment_cells(overhead, ["cost__P"])
        cells.to_csv(out / "overhead_cells.csv", index=False)
        hill = half_establishment(cells["cost__P"].to_numpy(), cells["p_establish"].to_numpy())
        (out / "overhead_hill_fit.json").write_text(json.dumps(hill, indent=2), encoding="utf-8")
        across_levels(naive, "cost__P", "year_established").to_csv(out / "overhead_pairwise.csv", index=False)
        save_figure(plot_sensitivity(naive, ["cost__P"], "year_established"), "overhead_establishment_year", out, naive)
        for shape_col in ("cost__switch_width", "cost__switch_steepness"):
            if shape_col in overhead and overhead[shape_col].nunique() > 1:
                short = shape_col.split("__")[-1]
                across_levels(overhead, shape_col, "Eb").to_csv(out / f"{short}_pairwise.csv", index=False)
                save_figure(plot_sensitivity(overhead, [shape_col], "Eb"), f"{short}_Eb", out, overhead)


# ┌────────────────────────────────────────────────────────────┐
# │ Grounded law  « sensitivity, capsule, oxygen »             │
# └────────────────────────────────────────────────────────────┘

def analyse_grounded(results: Path, out: Path, sweep: str = "grounded") -> None:
    runs = load_runs(results, sweep=sweep)
    out.mkdir(parents=True, exist_ok=True)
    runs = runs.assign(n_eff=[_effective_neurons(r) for _, r in runs.iterrows()])
    runs.to_csv(out / "runs.csv", index=False)

    uniform = runs[runs["preset"] == "exp4-atp-uniform"]
    if len(uniform) and {"naive", "above_switch"} <= set(uniform["init__arch_mode"]):
        cells = establishment_cells(uniform[uniform["atp__min_circuit"] == uniform["atp__min_circuit"].mode()[0]],
                                    ["atp__Fmax_f", "atp__M"])
        if len(cells) > 1:
            cells.to_csv(out / "grid_cells.csv", index=False)
            save_figure(plot_grid_maps(cells, "atp__Fmax_f", "atp__M", "grounded law"), "grid_maps", out, cells)
            naive_vs_seeded(uniform, ["atp__Fmax_f", "atp__M"]).to_csv(out / "naive_vs_seeded.csv", index=False)
    naive = uniform[uniform["init__arch_mode"] == "naive"] if len(uniform) else uniform
    swept = [p for p in GROUNDED_SENSITIVITY if p in naive and naive[p].nunique() > 1]
    if swept:
        for response in ("year_established", "Eb", "n_eff"):
            save_figure(plot_sensitivity(naive, swept, response), f"sensitivity_{response}", out,
                        naive[swept + [response]])
        pd.concat([across_levels(naive, p, "Eb") for p in swept], ignore_index=True).to_csv(
            out / "sensitivity_pairwise_Eb.csv", index=False)
        pd.concat([across_levels(naive, p, "year_established") for p in swept], ignore_index=True).to_csv(
            out / "sensitivity_pairwise_year.csv", index=False)

    capsule = runs[runs["preset"].str.startswith("exp5")]
    if len(capsule) and "capsule__pO2_kPa" in capsule:
        table = _oxygen_table(capsule)
        table.to_csv(out / "oxygen.csv", index=False)
        save_figure(plot_oxygen(table), "oxygen", out, table)
        rows = []
        for pO2, grp in table.groupby("pO2"):
            t, c = grp[grp["lineage"] == "tracheal"]["n_eff"], grp[grp["lineage"] == "convective"]["n_eff"]
            rows.append({"pO2": pO2, "n_tracheal": len(t), "n_convective": len(c),
                         "p_tracheal_vs_convective": two_sample(t, c)})
        pd.DataFrame(rows).to_csv(out / "oxygen_tests.csv", index=False)
        for lineage, grp in capsule.groupby("capsule__lineage"):
            across_levels(grp, "capsule__pO2_kPa", "n_eff").to_csv(out / f"oxygen_pairwise_{lineage}.csv", index=False)
        frog = [p for p in CAPSULE_SENSITIVITY if p in capsule and capsule[p].nunique() > 1]
        if frog:
            save_figure(plot_sensitivity(capsule, frog, "n_eff"), "frog_neurons", out, capsule[frog + ["n_eff"]])
            save_figure(plot_sensitivity(capsule, frog, "Ec"), "frog_capsule", out, capsule[frog + ["Ec"]])
            pd.concat([across_levels(capsule, p, "n_eff") for p in frog], ignore_index=True).to_csv(
                out / "frog_pairwise.csv", index=False)


def _effective_neurons(row: pd.Series) -> float:
    params = load_params(Path(row["run_dir"]))
    arch = architecture_from_params(params)
    if not hasattr(arch, "effective_neurons"):
        return np.nan
    return float(arch.effective_neurons(row["Eb"], row["Ec"]))


def _oxygen_table(capsule: pd.DataFrame) -> pd.DataFrame:
    table = capsule.rename(columns={"capsule__lineage": "lineage", "capsule__pO2_kPa": "pO2"})[
        ["lineage", "pO2", "seed", "tag", "n_eff", "Eb", "Ec", "Fb", "N", "run_dir"]].copy()
    mf_rows = {}
    for (lineage, pO2), grp in table.groupby(["lineage", "pO2"]):
        arch = architecture_from_params(load_params(Path(grp["run_dir"].iloc[0])))
        mixed = MeanField(arch).best_mixed()
        mf_rows[(lineage, pO2)] = (float(arch.hard_cap), float(arch.neurons_for_cost(mixed.Eb)) if mixed else np.nan)
    keys = list(zip(table["lineage"], table["pO2"], strict=True))
    table["hard_cap"] = [mf_rows[k][0] for k in keys]
    table["n_meanfield"] = [mf_rows[k][1] for k in keys]
    return table.drop(columns="run_dir")


# ┌────────────────────────────────────────────────────────────┐
# │ Branching  « is sigma bimodal inside runs? »               │
# └────────────────────────────────────────────────────────────┘

def analyse_branching(results: Path, out: Path, sweep: str = "branching") -> None:
    runs = load_runs(results, sweep=sweep)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    fits = bimodality_table(runs)
    if fits.empty:
        raise FileNotFoundError("no snapshots in the branching sweep")
    fits.to_csv(out / "bimodality_fits.csv", index=False)
    late = fits[fits["time"] >= fits["time"].max() // 2]
    fraction = bimodal_fraction(late, ["preset", "tag"])
    fraction.to_csv(out / "bimodal_fraction.csv", index=False)
    fraction["label"] = fraction["preset"].str.replace("exp", "") + " " + fraction["tag"]
    save_figure(plot_bimodal_fraction(fraction, "label"), "bimodal_fraction", out, fraction)
    best = fits.sort_values("delta_bic", ascending=False).iloc[0]

    def same_run(frame: pd.DataFrame) -> pd.Series:
        return (frame["tag"] == best["tag"]) & (frame["seed"] == best["seed"]) & (frame["preset"] == best["preset"])

    run = runs[same_run(runs)].iloc[0]
    snaps = load_snapshots(Path(run["run_dir"]))
    these = fits[same_run(fits)]
    stem = f"sigma_through_time_{best['preset']}_{best['tag']}_seed{best['seed']:03d}"
    save_figure(plot_sigma_through_time(snaps, these), stem, out, these)


# ┌────────────────────────────────────────────────────────────┐
# │ Taxa  « four animals, one law »                            │
# └────────────────────────────────────────────────────────────┘

def analyse_taxa(results: Path, out: Path, sweep: str = "taxa") -> None:
    runs = load_runs(results, sweep=sweep)
    out.mkdir(parents=True, exist_ok=True)
    runs["taxon"] = runs["preset"].str.replace("taxon-", "", regex=False)
    runs["a_index"] = runs["Eb"] / (runs["Ef"] + runs["Eb"])
    runs["brain_share"] = (runs["Eb"] + runs["Ec"]) / runs["life__basal_metabolism"]
    runs["n_eff"] = [_effective_neurons(r) for _, r in runs.iterrows()]
    runs.to_csv(out / "runs.csv", index=False)

    base = runs[runs["tag"] == "base"] if (runs["tag"] == "base").any() else runs
    predictions = {}
    histories: dict[str, list[pd.DataFrame]] = {}
    for taxon, grp in base.groupby("taxon"):
        histories[str(taxon)] = [pd.read_csv(Path(d) / "history.csv") for d in grp["run_dir"]]
        arch = architecture_from_params(load_params(Path(grp["run_dir"].iloc[0])))
        mixed = MeanField(arch).best_mixed()
        if mixed is not None:
            predictions[str(taxon)] = {"Ef": mixed.Ef, "Eb": mixed.Eb, "Ec": float(arch.optimal_capsule(mixed.Eb))}
    save_figure(plot_taxon_trajectories(histories, predictions), "taxon_trajectories", out,
                pd.DataFrame(predictions).T.reset_index(names="taxon"))
    for response in ("a_index", "brain_share", "n_eff"):
        save_figure(plot_taxon_endstates(base, response), f"taxon_{response}", out, base[["taxon", "seed", response]])
        across_levels(base, "taxon", response).to_csv(out / f"taxon_pairwise_{response}.csv", index=False)
    base = base.assign(retina_share=base["Ef"] / base["life__basal_metabolism"])
    summary = base.groupby("taxon").agg(n_runs=("seed", "size"), a_index=("a_index", "median"),
                                        brain_share=("brain_share", "median"), retina_share=("retina_share", "median"),
                                        n_eff=("n_eff", "median"), established=("established", "mean"))
    summary.to_csv(out / "taxon_summary.csv")
    niche = runs[runs["tag"] != "base"]
    for taxon, grp in niche.groupby("taxon"):
        candidates = ("atp__Fmax_f", "atp__M", "atp__half_saturation", "capsule__transport")
        swept = [c for c in candidates if grp[c].nunique() > 1]
        if swept:
            save_figure(plot_sensitivity(grp, swept, "n_eff"), f"niche_sensitivity_{taxon}", out,
                        grp[swept + ["n_eff"]])


# ┌────────────────────────────────────────────────────────────┐
# │ Ablation  « necessity in the agent model »                 │
# └────────────────────────────────────────────────────────────┘

def _intact_baselines(results: Path, taxa_sweep: str = "taxa") -> pd.DataFrame:
    """The unablated taxon runs (naive founders, base tag) from the taxa sweep,
    so the ablation table has an ``intact`` column to compare against."""
    try:
        taxa = load_runs(results, sweep=taxa_sweep)
    except FileNotFoundError:
        return pd.DataFrame()
    intact = taxa[(taxa["tag"] == "base") & ~taxa["preset"].str.contains("-from-")]
    return intact.assign(sweep=taxa_sweep)


def analyse_ablation(results: Path, out: Path, sweep: str = "ablation") -> None:
    runs = load_runs(results, sweep=sweep)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.concat([runs, _intact_baselines(results)], ignore_index=True)
    pattern = r"^taxon-(?P<taxon>[a-z]+)(?:--(?P<ablation>[a-z-]+?))?(?:-from-(?P<seed_from>[a-z]+))?$"
    parsed = runs["preset"].str.extract(pattern)
    runs = pd.concat([runs, parsed], axis=1)
    runs["ablation"] = runs["ablation"].fillna("intact")
    runs["a_index"] = runs["Eb"] / (runs["Ef"] + runs["Eb"])
    runs["brain_share"] = (runs["Eb"] + runs["Ec"]) / runs["life__basal_metabolism"]
    runs["n_eff"] = [_effective_neurons(r) for _, r in runs.iterrows()]
    runs.to_csv(out / "runs.csv", index=False)

    naive = runs[runs["seed_from"].isna() & (runs["tag"] == "base")]
    if len(naive):
        table = naive.groupby(["taxon", "ablation"]).agg(
            n=("seed", "size"), a_index=("a_index", "median"), n_eff=("n_eff", "median"),
            brain_share=("brain_share", "median"), established=("established", "mean"),
            year_established=("year_established", "median"))
        table.to_csv(out / "ablation_table.csv")
        tests = []
        for taxon, grp in naive.groupby("taxon"):
            for response in ("a_index", "n_eff", "brain_share"):
                pw = across_levels(grp, "ablation", response)
                if len(pw):
                    tests.append(pw.assign(taxon=taxon))
        if tests:
            pd.concat(tests, ignore_index=True).to_csv(out / "ablation_pairwise.csv", index=False)
        for response in ("a_index", "n_eff", "brain_share"):
            labelled = naive.assign(ablation=naive["taxon"] + "\n" + naive["ablation"])
            fig = plot_sensitivity(labelled, ["ablation"], response)
            save_figure(fig, f"ablation_{response}", out, naive[["taxon", "ablation", "seed", response]])

    frog = runs[(runs["taxon"] == "frog") & (runs["seed_from"] == "mouse")]
    if len(frog):
        frog.groupby("ablation").agg(n=("seed", "size"), n_eff=("n_eff", "median"), Eb=("Eb", "median")).to_csv(
            out / "frog_loss_under_ablation.csv")
        across_levels(frog, "ablation", "n_eff").to_csv(out / "frog_loss_pairwise.csv", index=False)
        save_figure(plot_sensitivity(frog, ["ablation"], "n_eff"), "frog_loss_under_ablation", out,
                    frog[["ablation", "seed", "n_eff"]])

    demand = runs[(runs["taxon"] == "mouse") & (runs["seed_from"] == "mouse") & (runs["tag"] != "base")]
    if len(demand) and demand["atp__M"].nunique() > 1:
        save_figure(plot_sensitivity(demand, ["atp__M"], "n_eff"), "demand_loss", out,
                    demand[["atp__M", "seed", "n_eff"]])
        across_levels(demand, "atp__M", "n_eff").to_csv(out / "demand_loss_pairwise.csv", index=False)

    regain = runs[(runs["taxon"] == "mouse") & (runs["seed_from"] == "frog")]
    if len(regain):
        hists = {"mouse law from frog state": [pd.read_csv(Path(d) / "history.csv") for d in regain["run_dir"]]}
        arch = architecture_from_params(load_params(Path(regain["run_dir"].iloc[0])))
        mixed = MeanField(arch).best_mixed()
        pred = None
        if mixed:
            capsule = float(arch.optimal_capsule(mixed.Eb))
            pred = {"mouse law from frog state": {"Ef": mixed.Ef, "Eb": mixed.Eb, "Ec": capsule}}
        save_figure(plot_taxon_trajectories(hists, pred), "regain_trajectories", out,
                    regain[["seed", "Ef", "Eb", "Ec"]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brainevo-analyse", description="Analyse brainevo-abm sweep results.")
    parser.add_argument("command", choices=["thesis", "grounded", "branching", "taxa", "ablation", "all"])
    parser.add_argument("--results", type=Path, required=True, help="results root holding the sweep directories")
    parser.add_argument("--out", type=Path, required=True, help="directory for tables and figures")
    parser.add_argument("--sweep", default=None, help="sweep directory name if it differs from the command")
    return parser


if __name__ == "__main__":
    sys.exit(main())
