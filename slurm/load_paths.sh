#!/usr/bin/env bash
# ============================================================
#  BrainEvo — load machine-specific paths into the shell
# ============================================================
#  Source this (do not execute it) before submitting jobs:
#
#     source slurm/load_paths.sh              # 'hpc' profile
#     BRAINEVO_PROFILE=local source slurm/load_paths.sh
#
#  Reads local_paths.json (gitignored) through brainevo/config.py and
#  exports BRAINEVO_RESULTS_ROOT, BRAINEVO_CODE_ROOT,
#  BRAINEVO_CONDA_SETUP, BRAINEVO_CONDA_ENV, SBATCH_ACCOUNT and
#  SBATCH_PARTITION.  sbatch reads the last two directly, so no account
#  or partition is hard-coded in any job script.
# ============================================================
: "${BRAINEVO_PROFILE:=hpc}"
export BRAINEVO_PROFILE

_lp_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -f "${_lp_dir}/../brainevo/config.py" ]; then
    _repo="$(cd "${_lp_dir}/.." && pwd)"
elif [ -n "${BRAINEVO_CODE_ROOT:-}" ] && [ -f "${BRAINEVO_CODE_ROOT}/brainevo/config.py" ]; then
    _repo="${BRAINEVO_CODE_ROOT}"
else
    echo "load_paths.sh: cannot locate the BrainEvo repo (brainevo/config.py)." >&2
    return 1 2>/dev/null || exit 1
fi

_exports="$(python3 "${_repo}/brainevo/config.py" --export "${BRAINEVO_PROFILE}")" || {
    echo "load_paths.sh: local_paths.json missing or incomplete (copy local_paths.template.json)." >&2
    return 1 2>/dev/null || exit 1
}
eval "${_exports}"
export BRAINEVO_CODE_ROOT="${BRAINEVO_CODE_ROOT:-${_repo}}"
