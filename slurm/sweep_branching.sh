#!/bin/bash
#SBATCH --job-name=be_branching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# Account + partition come from local_paths.json via load_paths.sh
# (SBATCH_ACCOUNT / SBATCH_PARTITION).  Submit from the repo root:
#     source slurm/load_paths.sh && mkdir -p logs && sbatch slurm/sweep_branching.sh
# Quick test:  sbatch --time=00:20:00 --export=ALL,REPS=1 slurm/sweep_branching.sh
# ============================================================
#  Step 6 — niche branching: does sigma go bimodal within runs?
# ============================================================
#  Niche-based foraging with matched-filter tuning under both cost laws,
#  with and without volatility, over finite correlation lengths.  Dense
#  individual snapshots (every 250 yr) for the bimodality test.
SWEEP_NAME=branching

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

SNAP="--snapshot-every 250"
REPS=20 stage "exp2 tuning"           --preset exp2-tuning      $SNAP --grid cost__Fmax_f=0.8,1.2
REPS=20 stage "exp2 relative control" --preset exp2-relative-control $SNAP --grid cost__Fmax_f=0.8
REPS=20 stage "exp4 atp tuning"       --preset exp4-atp-tuning  $SNAP --grid atp__Fmax_f=0.8,1.2
REPS=20 stage "exp3 volatility x L"   --preset exp3-volatility  $SNAP --grid environment__volatility=0.1,0.25,0.5 environment__volatility_corr_len=1,5,20,inf
REPS=20 stage "env sigma"             --preset exp2-tuning      $SNAP --grid environment__env_sigma=10,15,25

echo "DONE: $(date)"
