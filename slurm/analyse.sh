#!/bin/bash
#SBATCH --job-name=be_analyse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# Runs brainevo-analyse over every sweep in the results root on a compute node.
#     source slurm/load_paths.sh && sbatch slurm/analyse.sh
# ============================================================
if [ -z "${BRAINEVO_RESULTS_ROOT:-}" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
    if [ -f "${_here}/load_paths.sh" ]; then source "${_here}/load_paths.sh";
    elif [ -n "${BRAINEVO_CODE_ROOT:-}" ]; then source "${BRAINEVO_CODE_ROOT}/slurm/load_paths.sh";
    else echo "no BRAINEVO_RESULTS_ROOT and no load_paths.sh found" >&2; exit 1; fi
fi
sleep 5
source "${BRAINEVO_CONDA_SETUP}"
if [ -n "${BRAINEVO_CONDA_ENV:-}" ]; then conda activate "${BRAINEVO_CONDA_ENV}"; fi
echo "Job ${SLURM_JOB_ID} on $(hostname), $(date); results ${BRAINEVO_RESULTS_ROOT}"
OUT="${BRAINEVO_RESULTS_ROOT}/../figures"
mkdir -p "${OUT}"
T0=$(date +%s)
brainevo-analyse all --results "${BRAINEVO_RESULTS_ROOT}" --out "${OUT}"
echo "analysis took $(( $(date +%s) - T0 )) s; DONE $(date)"
