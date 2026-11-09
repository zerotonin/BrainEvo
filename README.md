# BrainEvo

**One continuum or two strategies?** An eco-evolutionary agent-based model
(FEBE: Front-End / Back-End Evolution) of how nervous systems distribute
sensory processing between a peripheral front-end and a central back-end
under energetic and ecological constraints, together with its mean-field
adaptive-dynamics reduction.

The model was developed by Theo Dodd for a BSc(Hons) thesis in Neuroscience
(University of Otago, 2026), supervised by Bart Geurten. This repository is
the publication-ready refactor: same model, same random-number stream, with
a package layout, tests, a command-line runner, and the analytical reduction
that turns the simulation results into a bifurcation diagram.

## Install

```bash
conda env create -f environment.yml
conda activate brainevo
# or
pip install -e ".[dev]"
```

Then tell the package where results should go. Either copy the template and
fill in a path,

```bash
cp local_paths.template.json local_paths.json
```

or set `BRAINEVO_RESULTS_ROOT`, or pass `--out DIR` on every call.

## Quick start

```bash
# one 4000-year run of the matched-filter experiment (thesis Exp. 2)
brainevo-abm run --preset exp2-tuning --seed 1

# the cost-law sweep of thesis Fig. 7, 3 replicates per cell, 8 workers
brainevo-abm sweep --preset exp1-cost-sweep \
    --grid cost__Fmax_f=0.1,0.2,0.3,0.5,0.8,1.2,1.8,2.7,4 cost__M=1,1.5,2,2.5,3,3.5,4,4.5,5 \
    --set cost__P=2 --replicates 3 --workers 8

# the same experiment under the grounded ATP cost law
brainevo-abm run --preset exp4-atp-uniform --seed 1

# capsule as a third heritable trait, insect-type tracheal oxygen cap at Carboniferous pO2
brainevo-abm run --preset exp5-capsule-tracheal --set capsule__pO2_kPa=30 --seed 1

# the four taxon test cases (grounded law, sources in brainevo/abm/taxa.py)
brainevo-abm run --preset taxon-frog-from-mouse --seed 1

# analysis: tables (reRandomStats) and figures from a results root
brainevo-analyse all --results /path/to/results --out /path/to/figures

# cluster sweeps (SLURM): fill the hpc profile of local_paths.json, then
source slurm/load_paths.sh && mkdir -p logs
for s in grounded thesis branching taxa ablation; do sbatch slurm/sweep_$s.sh; done

# necessity table, Haller's rule, demand loss, reversibility (mean field)
brainevo-ode predictions --out /path/to/figures

# the adaptive-dynamics reduction
brainevo-ode grid            # mean-field prediction over the Fig-7 grid
brainevo-ode bifurcation     # E_b* branches against the overhead P
brainevo-ode hysteresis --w 0.3
brainevo-ode fold-map --shape ramp     # softplus ramp of width w
brainevo-ode fold-map --shape hill     # Hill saturation gate, sweep steepness n
brainevo-ode niche-flow                # canonical equation with niche feedback
brainevo-ode niche-pip                 # pairwise invasibility in niche breadth
```

Every run writes `history.csv`, `snapshots.csv`, `params.json` and a
trajectory figure (SVG with editable text, PNG). Every figure comes with a
CSV of the plotted data.

## Package layout

```
brainevo/
├── constants.py       colours, figure rules, type aliases
├── config.py          results root via local_paths.json
├── figures.py         save_figure(): SVG + PNG + CSV
├── abm/
│   ├── cost_laws.py   Architecture: the thesis' Shannon energy -> fidelity laws
│   ├── energetics.py  ATPArchitecture + CapsuleLaw: grounded laws, capsule trait, oxygen cap
│   ├── taxa.py        arthropod, mouse, frog, primate: sourced parameter sets and expectations
│   ├── environment.py Gaussian resource axis, depletion, regrowth, volatility
│   ├── organism.py    one agent
│   ├── population.py  life cycle: forage, reproduce, mutate, cull
│   ├── recording.py   yearly log and snapshots
│   ├── presets.py     thesis experiments as named parameter sets
│   ├── viz.py, cli.py
└── ode/
    ├── meanfield.py   Q = F / C landscape, singular strategies, invasion fitness
    ├── canonical.py   canonical equation, forward/reverse ramps
    ├── niche.py       vector feedback: resident equilibrium, invasion fitness, branching test
    └── viz.py, cli.py
```

`_legacy/` holds the thesis code verbatim; `tests/test_legacy_equivalence.py`
checks that the refactor reproduces it exactly for a shared seed.

## The reduction in one paragraph

Under uniform foraging the only ecological feedback is the scalar
probability that a foraging roll finds an item, so a mutant invades a
resident exactly when it has a higher fidelity per unit cost,
`Q = (1 + F_f + F_b) / (b + c + E_f + E_b)`. Singular strategies satisfy
`F_f'(E_f) = Q` and `F_b'(E_b) = Q`: invest at each end until the marginal
return equals the organism's average return. With a hard switch-on, the
front-only state is always a local attractor, so the two architectures are
bistable for every parameter value and the transition is an accessibility
(fitness-valley crossing) problem, not an affordability threshold. A smooth
switch-on of width `w` removes the front-only attractor at a fold, which is
where a genuine saddle-node bifurcation and hysteresis appear.

## Citation

See `CITATION.cff`.
