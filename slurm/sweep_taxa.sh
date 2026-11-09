#!/bin/bash
#SBATCH --job-name=be_taxa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# Account + partition come from local_paths.json via load_paths.sh
# (SBATCH_ACCOUNT / SBATCH_PARTITION).  Submit from the repo root:
#     source slurm/load_paths.sh && mkdir -p logs && sbatch slurm/sweep_taxa.sh
# Quick test:  sbatch --time=00:20:00 --export=ALL,REPS=1 slurm/sweep_taxa.sh
# ============================================================
#  Step 7 — taxon test cases: arthropod, mouse, frog (secondary loss), primate
# ============================================================
#  Each taxon's grounded law with naive founders (20 replicates), the
#  frog seeded at the mouse optimum, and a niche sweep per taxon so the
#  two assumed numbers (Fmax_f, M) are shown not to drive the ordering.
SWEEP_NAME=taxa

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

REPS=20 stage "arthropod"        --preset taxon-arthropod
REPS=20 stage "mouse"            --preset taxon-mouse
REPS=20 stage "frog"             --preset taxon-frog
REPS=20 stage "primate"          --preset taxon-primate
REPS=20 stage "frog from mouse"  --preset taxon-frog-from-mouse --snapshot-every 500

# niche assumptions swept around each taxon (naive founders, 5 reps)
REPS=5 stage "arthropod niche"   --preset taxon-arthropod --grid atp__Fmax_f=1.0,1.4,2.0 atp__M=1.0,1.5,2.5
REPS=5 stage "mouse niche"       --preset taxon-mouse     --grid atp__Fmax_f=0.4,0.6,0.9 atp__M=2,3,4
REPS=5 stage "frog niche"        --preset taxon-frog      --grid atp__Fmax_f=0.6,0.8,1.2 atp__M=1.0,1.2,1.8
REPS=5 stage "primate niche"     --preset taxon-primate   --grid atp__Fmax_f=0.8,1.2,1.6 atp__M=3,4,6
# the frog's loss driver and its runner-up
REPS=10 stage "frog saturation"  --preset taxon-frog-from-mouse --grid atp__half_saturation=0.02,0.05,0.1,0.25
REPS=10 stage "frog carriage"    --preset taxon-frog-from-mouse --grid capsule__transport=0.25,1,2,4

echo "DONE: $(date)"
