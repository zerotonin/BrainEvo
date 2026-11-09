# ╔══════════════════════════════════════════════════════════════════╗
# ║  BrainEvo — ode.cli                                              ║
# ║  « brainevo-ode: figures of the adaptive-dynamics reduction »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  landscape    Q(E_f, E_b) with singular strategies               ║
# ║  valley       the fitness valley for several overheads P         ║
# ║  bifurcation  E_b* branches against P, for hard and smooth       ║
# ║               switches                                           ║
# ║  hysteresis   forward/reverse ramp of P in the canonical eq.     ║
# ║  fold-map     number of attractors over the (P, w) plane         ║
# ║  grid         mean-field prediction over the thesis Fig-7 grid   ║
# ║  niche-flow   canonical equation with niche feedback, and the    ║
# ║               Geritz classification of the end point             ║
# ║  niche-pip    pairwise invasibility plot in niche breadth        ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from brainevo.abm.cost_laws import Architecture, CostLawParams
from brainevo.config import MissingLocalPathsError, resolve_results_root
from brainevo.figures import apply_style, save_figure
from brainevo.ode.canonical import ramp_hysteresis
from brainevo.ode.meanfield import DEFAULT_BASAL, DEFAULT_REPRO_DRAIN, MeanField
from brainevo.ode.niche import NicheMeanField, Strategy
from brainevo.ode.viz import (
    plot_bifurcation,
    plot_fold_map,
    plot_hysteresis,
    plot_landscape,
    plot_niche_trajectory,
    plot_pip,
    plot_valley,
)

P_LABEL = r"back-end overhead $P$ (multiples of $E_{f,\mathrm{sat}}$)"
FIG7_FMAX = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.7, 4.0)
FIG7_M = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        out = resolve_results_root(args.out) / "ode"
    except MissingLocalPathsError as err:
        parser.error(str(err))
    apply_style()
    base = CostLawParams(Fmax_f=args.Fmax, M=args.M, P=args.P, back_rate=args.back_rate,
                         switch_shape=args.shape, switch_width=args.w, switch_steepness=args.n)

    def mk(P: float | None = None, w: float | None = None, n: float | None = None) -> MeanField:
        params = replace(base, P=args.P if P is None else P,
                         switch_width=args.w if w is None else w,
                         switch_steepness=args.n if n is None else n)
        return MeanField(Architecture(params), basal=args.basal, repro_drain=args.repro_drain)

    {
        "landscape": lambda: save_figure(plot_landscape(mk()), "landscape", out),
        "valley": lambda: save_figure(
            plot_valley({f"P = {P:g}": mk(P=P) for P in (0.5, 1.0, 2.0, 3.75)}), "valley", out),
        "bifurcation": lambda: bifurcation(mk, out, args),
        "hysteresis": lambda: hysteresis(mk, out, args),
        "fold-map": lambda: fold_map(mk, out, args),
        "grid": lambda: grid(args, out),
        "niche-flow": lambda: niche_flow(base, out, args),
        "niche-pip": lambda: niche_pip(base, out, args),
    }[args.command]()
    print(f"wrote figures to {out}")
    return 0


def bifurcation(mk: Callable[..., MeanField], out: Path, args: argparse.Namespace) -> None:
    Ps = np.linspace(0.0, args.P_max, 61)
    for w in (0.0, args.w if args.w > 0 else 0.3):
        branches: dict[str, list[tuple[float, float]]] = {"front-only": [], "mixed": [], "global": []}
        for P in Ps:
            mf = mk(P=P, w=w)
            for s in mf.singular_strategies():
                branches[s.branch].append((P, s.Eb))
            branches["global"].append((P, mf.global_optimum().Eb))
        fig = plot_bifurcation(Ps, branches, P_LABEL)
        rows = [{"branch": b, "P": p, "Eb": e} for b, pts in branches.items() for p, e in pts]
        save_figure(fig, f"bifurcation_{args.shape}_w{w:g}", out, pd.DataFrame(rows))


def hysteresis(mk: Callable[..., MeanField], out: Path, args: argparse.Namespace) -> None:
    Ps = np.linspace(0.0, args.P_max, 41)
    loop = ramp_hysteresis(lambda P: mk(P=P), Ps)
    tag = f"w{args.w:g}" if args.shape == "ramp" else f"n{args.n:g}"
    save_figure(plot_hysteresis(loop, P_LABEL), f"hysteresis_{args.shape}_{tag}", out, pd.DataFrame(loop))


def fold_map(mk: Callable[..., MeanField], out: Path, args: argparse.Namespace) -> None:
    """Attractor count and valley deficit over (P, switch parameter).

    Ramp: the parameter is the width w.  Hill: the parameter is the flank
    steepness n, swept on a log grid.
    """
    Ps = np.linspace(0.0, args.P_max, 31)
    if args.shape == "ramp":
        ys, ylabel, kw = np.linspace(0.0, 1.5, 31), r"switch-on width $w$ (energy units)", "w"
    else:
        ys, ylabel, kw = np.logspace(-0.5, 2.0, 31), r"Hill flank steepness $n$", "n"
    n_attr = np.zeros((len(ys), len(Ps)))
    deficit = np.zeros_like(n_attr)
    for i, y in enumerate(tqdm(ys, desc="fold map")):
        for j, P in enumerate(Ps):
            mf = mk(P=P, **{kw: y})
            n_attr[i, j] = len(mf.singular_strategies())
            deficit[i, j] = mf.valley_deficit()
    rows = [{"P": P, kw: y, "n_attractors": n_attr[i, j], "valley_deficit": deficit[i, j]}
            for i, y in enumerate(ys) for j, P in enumerate(Ps)]
    fig = plot_fold_map(Ps, ys, n_attr, deficit, ylabel, log_y=args.shape == "hill")
    save_figure(fig, f"fold_map_{args.shape}", out, pd.DataFrame(rows))


def grid(args: argparse.Namespace, out: Path) -> None:
    """Mean-field prediction over the thesis Fig-7 grid (P = 2, back_rate = 0.8)."""
    rows = []
    for Fmax in FIG7_FMAX:
        for M in FIG7_M:
            mf = MeanField(Architecture(CostLawParams(Fmax_f=Fmax, M=M, P=args.P, back_rate=args.back_rate)),
                           basal=args.basal, repro_drain=args.repro_drain)
            front, mixed = mf.best_front_only(), mf.best_mixed()
            rows.append({"Fmax": Fmax, "M": M, "MFmax": Fmax * M, "P": args.P,
                         "Q_front": front.Q, "Ef_front": front.Ef,
                         "Q_mixed": mixed.Q if mixed else np.nan,
                         "Ef_mixed": mixed.Ef if mixed else np.nan, "Eb_mixed": mixed.Eb if mixed else np.nan,
                         "backend_globally_favoured": bool(mixed and mixed.Q > front.Q),
                         "valley_depth": mf.valley_depth(), "break_even_Eb": mf.break_even_Eb()})
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"meanfield_grid_P{args.P:g}.csv", index=False)
    print(frame.pivot(index="Fmax", columns="M", values="backend_globally_favoured").to_string())


def _niche_model(base: CostLawParams, args: argparse.Namespace) -> NicheMeanField:
    from dataclasses import replace as _replace

    cost = _replace(base, tuning_width=args.tuning_width)
    return NicheMeanField(Architecture(cost), repro_drain=args.repro_drain)


def niche_flow(base: CostLawParams, out: Path, args: argparse.Namespace) -> None:
    """Integrate the canonical equation from front-only and from mixed founders."""
    model = _niche_model(base, args)
    mu = float(model.env.env_mu)
    starts = {"front-only": Strategy(0.3, 0.0, mu, args.sigma0),
              "mixed": Strategy(0.3, model.arch.e_switch + 0.3 * model.arch.E_b_climb, mu, args.sigma0)}
    rows = []
    for label, s0 in starts.items():
        t, traj, N = model.integrate_canonical(s0, t_max=args.t_max)
        Ef = np.array([s.Ef for s in traj])
        Eb = np.array([s.Eb for s in traj])
        sig = np.array([s.sigma for s in traj])
        save_figure(plot_niche_trajectory(t, Ef, Eb, sig, N), f"niche_flow_{label}", out,
                    pd.DataFrame({"t": t, "Ef": Ef, "Eb": Eb, "sigma": sig, "N": N}))
        end = traj[-1]
        free = (0, 2) if end.Eb <= 1e-6 else (0, 1, 2)
        cls = model.classify(end, free=free)
        rows.append({"start": label, "Ef": end.Ef, "Eb": end.Eb, "sigma": end.sigma, "N": N[-1],
                     "kind": cls.kind, "jacobian_eig": cls.jacobian_eigenvalues.tolist(),
                     "hessian_eig": cls.hessian_eigenvalues.tolist()})
        print(f"{label:>10}: Ef={end.Ef:.3f} Eb={end.Eb:.3f} sigma={end.sigma:.2f} N={N[-1]:.0f}  -> {cls.kind}")
    pd.DataFrame(rows).to_csv(out / "niche_singular.csv", index=False)


def niche_pip(base: CostLawParams, out: Path, args: argparse.Namespace) -> None:
    """PIP in niche breadth for a front-only and a mixed architecture."""
    model = _niche_model(base, args)
    mu = float(model.env.env_mu)
    sigmas = np.linspace(2.0, 40.0, args.pip_n)
    for label, Eb in (("front-only", 0.0), ("mixed", model.arch.e_switch + 0.5 * model.arch.E_b_climb)):
        S = model.pip_sigma(Strategy(0.5, Eb, mu, 15.0), sigmas)
        fig = plot_pip(sigmas, S, title=f"{label}: $E_f$=0.5, $E_b$={Eb:.2f}")
        rows = [{"sigma_resident": sigmas[j], "sigma_mutant": sigmas[i], "s": S[i, j]}
                for i in range(len(sigmas)) for j in range(len(sigmas))]
        save_figure(fig, f"niche_pip_{label}", out, pd.DataFrame(rows))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brainevo-ode", description="Adaptive-dynamics reduction of FEBE.")
    p.add_argument("command", choices=["landscape", "valley", "bifurcation", "hysteresis", "fold-map", "grid",
                                       "niche-flow", "niche-pip"])
    p.add_argument("--Fmax", type=float, default=1.2)
    p.add_argument("--M", type=float, default=2.5)
    p.add_argument("--P", type=float, default=2.0)
    p.add_argument("--back-rate", dest="back_rate", type=float, default=0.8)
    p.add_argument("--shape", choices=["ramp", "hill"], default="ramp", help="switch-on shape")
    p.add_argument("--w", type=float, default=0.0, help="ramp width (0 = thesis kink)")
    p.add_argument("--n", type=float, default=float("inf"), help="Hill flank steepness (inf = step)")
    p.add_argument("--P-max", dest="P_max", type=float, default=6.0)
    p.add_argument("--basal", type=float, default=DEFAULT_BASAL)
    p.add_argument("--repro-drain", dest="repro_drain", type=float, default=DEFAULT_REPRO_DRAIN)
    p.add_argument("--tuning-width", dest="tuning_width", type=float, default=3.75, help="matched-filter width")
    p.add_argument("--sigma0", type=float, default=15.0, help="founder niche breadth for niche-flow")
    p.add_argument("--t-max", dest="t_max", type=float, default=50.0)
    p.add_argument("--pip-n", dest="pip_n", type=int, default=41)
    p.add_argument("--out", type=Path, default=None)
    return p


if __name__ == "__main__":
    sys.exit(main())
