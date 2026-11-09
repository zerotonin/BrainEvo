#!/bin/bash
#SBATCH --job-name=be_grounded
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# Account + partition come from local_paths.json via load_paths.sh
# (SBATCH_ACCOUNT / SBATCH_PARTITION).  Submit from the repo root:
#     source slurm/load_paths.sh && mkdir -p logs && sbatch slurm/sweep_grounded.sh
# Quick test:  sbatch --time=00:20:00 --export=ALL,REPS=1 slurm/sweep_grounded.sh
# ============================================================
#  Step 4 — grounded ATP cost law: demand grid, naive vs seeded, constant-by-constant sensitivity, capsule and oxygen
# ============================================================
#  Every GroundedConstant range in brainevo/abm/energetics.py is swept
#  one at a time around (Fmax_f 1.2, M 2.5).  Each stage writes its own
#  sweep_summary_<keys>.csv; finished runs are skipped on resubmission.
SWEEP_NAME=grounded

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

# demand grid, naive and seeded founders
stage "exp4 demand grid"        --preset exp4-atp-uniform --grid atp__Fmax_f=0.4,0.8,1.2,1.6,2.0 atp__M=1,1.5,2,2.5,3,4 init__arch_mode=naive,above_switch

# one-at-a-time sensitivity around the reference point (naive founders)
stage "min_circuit"             --preset exp4-atp-uniform --grid atp__min_circuit=0.003,0.01,0.02,0.05,0.1,0.2,0.3,0.5
stage "back_share"              --preset exp4-atp-uniform --grid atp__back_share=0.02,0.05,0.08,0.12,0.2
stage "front_share"             --preset exp4-atp-uniform --grid atp__front_share=0.06,0.08,0.10
stage "front_exponent"          --preset exp4-atp-uniform --grid atp__front_exponent=1.5,1.7
stage "wiring_exponent"         --preset exp4-atp-uniform --grid atp__wiring_exponent=0,0.3333
stage "half_saturation"         --preset exp4-atp-uniform --grid atp__half_saturation=0.1,0.25,0.5,1.0
stage "hill gate"               --preset exp4-atp-uniform --grid atp__switch_shape=hill atp__switch_steepness=2,5,15,100
stage "min_circuit x seeded"    --preset exp4-atp-uniform --grid atp__min_circuit=0.1,0.2,0.3,0.5 init__arch_mode=naive,above_switch

# capsule as third trait: oxygen history for both lineages, frog knobs
stage "convective pO2"          --preset exp5-capsule-convective --grid capsule__pO2_kPa=10,12,15,18,21.2,25,30,35
stage "tracheal pO2"            --preset exp5-capsule-tracheal   --grid capsule__pO2_kPa=10,12,15,18,21.2,25,30,35
stage "tracheal pO2 x demand"   --preset exp5-capsule-tracheal   --grid capsule__pO2_kPa=12,21.2,30 atp__Fmax_f=0.4,1.2,2.0 atp__M=1.5,2.5,4
stage "tracheal exponent"       --preset exp5-capsule-tracheal   --grid capsule__tracheal_exponent=0,1,1.5,2,3 capsule__pO2_kPa=12,30
stage "transport (frog)"        --preset exp5-capsule-convective --grid capsule__transport=0.1,0.25,0.5,1,2,4
stage "build_years"             --preset exp5-capsule-convective --grid capsule__build_years=0,2,5,10,20
stage "demand (frog)"           --preset exp5-capsule-convective --grid atp__Fmax_f=0.2,0.5,0.8,1.2 atp__M=1.5,2.5

echo "DONE: $(date)"
