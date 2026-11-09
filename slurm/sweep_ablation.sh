#!/bin/bash
#SBATCH --job-name=be_ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# Account + partition come from local_paths.json via load_paths.sh
# (SBATCH_ACCOUNT / SBATCH_PARTITION).  Submit from the repo root:
#     source slurm/load_paths.sh && mkdir -p logs && sbatch slurm/sweep_ablation.sh
# Quick test:  sbatch --time=00:20:00 --export=ALL,REPS=1 slurm/sweep_ablation.sh
# ============================================================
#  Step 8 — necessity (ablations) and emergent-feature tests in the ABM
# ============================================================
#  Every taxon with one component removed; the frog's secondary loss
#  under each ablation; demand loss (mouse seeded at its optimum, M
#  lowered); reversibility (mouse law seeded at the frog's reduced state).

SWEEP_NAME=ablation

# ---- machine-specific paths (slurm/load_paths.sh; nothing hard-coded) ----
if [ -z "${BRAINEVO_RESULTS_ROOT:-}" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
    if [ -f "${_here}/load_paths.sh" ]; then source "${_here}/load_paths.sh";
    elif [ -n "${BRAINEVO_CODE_ROOT:-}" ]; then source "${BRAINEVO_CODE_ROOT}/slurm/load_paths.sh";
    else echo "no BRAINEVO_RESULTS_ROOT and no load_paths.sh found" >&2; exit 1; fi
fi
sleep 5
source "${BRAINEVO_CONDA_SETUP}"
if [ -n "${BRAINEVO_CONDA_ENV:-}" ]; then conda activate "${BRAINEVO_CONDA_ENV}"; fi

echo "============================================"
echo "  Job:       ${SLURM_JOB_ID}   ${SLURM_JOB_NAME}"
echo "  Partition: ${SLURM_JOB_PARTITION}   Node: $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK}"
echo "  Python:    $(which python)   brainevo $(python -c 'import brainevo;print(brainevo.__version__)')"
echo "  Results:   ${BRAINEVO_RESULTS_ROOT}"
echo "  Started:   $(date)"
echo "============================================"

OUT="${BRAINEVO_RESULTS_ROOT}/${SWEEP_NAME}"
mkdir -p "${OUT}"
W="${SLURM_CPUS_PER_TASK}"
REPS="${REPS:-10}"

stage () {  # stage <label> <brainevo-abm sweep args...>
    local label="$1"; shift
    local T0; T0=$(date +%s)
    echo "--- ${label}  ($(date))"
    brainevo-abm sweep --out "${OUT}" --workers "${W}" --replicates "${REPS}" "$@"
    echo "--- ${label} took $(( $(date +%s) - T0 )) s"
}

REPS=10 stage "taxon-arthropod--no-capsule" --preset taxon-arthropod--no-capsule
REPS=10 stage "taxon-arthropod--no-min-circuit" --preset taxon-arthropod--no-min-circuit
REPS=10 stage "taxon-arthropod--no-oxygen-cap" --preset taxon-arthropod--no-oxygen-cap
REPS=10 stage "taxon-arthropod--no-saturation" --preset taxon-arthropod--no-saturation
REPS=10 stage "taxon-arthropod--thesis-prices" --preset taxon-arthropod--thesis-prices
REPS=10 stage "taxon-frog--no-capsule" --preset taxon-frog--no-capsule
REPS=10 stage "taxon-frog--no-capsule-from-mouse" --preset taxon-frog--no-capsule-from-mouse
REPS=10 stage "taxon-frog--no-min-circuit" --preset taxon-frog--no-min-circuit
REPS=10 stage "taxon-frog--no-min-circuit-from-mouse" --preset taxon-frog--no-min-circuit-from-mouse
REPS=10 stage "taxon-frog--no-oxygen-cap" --preset taxon-frog--no-oxygen-cap
REPS=10 stage "taxon-frog--no-oxygen-cap-from-mouse" --preset taxon-frog--no-oxygen-cap-from-mouse
REPS=10 stage "taxon-frog--no-saturation" --preset taxon-frog--no-saturation
REPS=10 stage "taxon-frog--no-saturation-from-mouse" --preset taxon-frog--no-saturation-from-mouse
REPS=10 stage "taxon-frog--thesis-prices" --preset taxon-frog--thesis-prices
REPS=10 stage "taxon-frog--thesis-prices-from-mouse" --preset taxon-frog--thesis-prices-from-mouse
REPS=10 stage "taxon-mouse--no-capsule" --preset taxon-mouse--no-capsule
REPS=10 stage "taxon-mouse--no-min-circuit" --preset taxon-mouse--no-min-circuit
REPS=10 stage "taxon-mouse--no-oxygen-cap" --preset taxon-mouse--no-oxygen-cap
REPS=10 stage "taxon-mouse--no-saturation" --preset taxon-mouse--no-saturation
REPS=10 stage "taxon-mouse--thesis-prices" --preset taxon-mouse--thesis-prices
REPS=20 stage "regain (mouse law from frog state)" --preset taxon-mouse-from-frog --snapshot-every 500
REPS=10 stage "taxon-primate--no-capsule" --preset taxon-primate--no-capsule
REPS=10 stage "taxon-primate--no-min-circuit" --preset taxon-primate--no-min-circuit
REPS=10 stage "taxon-primate--no-oxygen-cap" --preset taxon-primate--no-oxygen-cap
REPS=10 stage "taxon-primate--no-saturation" --preset taxon-primate--no-saturation
REPS=10 stage "taxon-primate--thesis-prices" --preset taxon-primate--thesis-prices
REPS=10 stage "demand loss (mouse seeded, M lowered)" --preset taxon-mouse-from-mouse --grid atp__M=0.05,0.1,0.2,0.3,0.5,1,2,3

echo "DONE: $(date)"
