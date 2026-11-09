#!/bin/bash
#SBATCH --job-name=be_thesis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# Account + partition come from local_paths.json via load_paths.sh
# (SBATCH_ACCOUNT / SBATCH_PARTITION).  Submit from the repo root:
#     source slurm/load_paths.sh && mkdir -p logs && sbatch slurm/sweep_thesis.sh
# Quick test:  sbatch --time=00:20:00 --export=ALL,REPS=1 slurm/sweep_thesis.sh
# ============================================================
#  Step 5 — thesis Shannon law: replicated Fig-7 grid, naive vs seeded, establishment times
# ============================================================
#  The naive-vs-seeded contrast at every grid point is the test that the
#  Fig-7 boundary is accessibility (waiting time), not affordability.
#  year_established in the summaries feeds the waiting-time fit.
SWEEP_NAME=thesis

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

stage "exp1 grid P=0.5"   --preset exp1-cost-sweep --grid cost__Fmax_f=0.4,0.8,1.2,1.6,2.0 cost__M=1,1.5,2,2.5,3,4 cost__P=0.5  init__arch_mode=naive,above_switch
stage "exp1 grid P=1"     --preset exp1-cost-sweep --grid cost__Fmax_f=0.4,0.8,1.2,1.6,2.0 cost__M=1,1.5,2,2.5,3,4 cost__P=1    init__arch_mode=naive,above_switch
stage "exp1 grid P=2"     --preset exp1-cost-sweep --grid cost__Fmax_f=0.4,0.8,1.2,1.6,2.0 cost__M=1,1.5,2,2.5,3,4 cost__P=2    init__arch_mode=naive,above_switch
stage "exp1 grid P=3.75"  --preset exp1-cost-sweep --grid cost__Fmax_f=0.4,0.8,1.2,1.6,2.0 cost__M=1,1.5,2,2.5,3,4 cost__P=3.75 init__arch_mode=naive,above_switch

# overhead sweep at the thesis reference point, more replicates for the waiting-time distribution
REPS=20 stage "overhead"  --preset exp1-overhead-sweep --grid cost__P=0.25,0.5,0.75,1,1.5,2,2.5,3,3.75,4.5 init__arch_mode=naive,above_switch

# switch shape in the ABM (ramp width, Hill steepness) at the reference point
stage "ramp width"        --preset exp1-overhead-sweep --grid cost__switch_width=0,0.1,0.2,0.4,0.8 cost__P=2
stage "hill steepness"    --preset exp1-overhead-sweep --grid cost__switch_shape=hill cost__switch_steepness=1,2,5,15,100 cost__P=2

echo "DONE: $(date)"
