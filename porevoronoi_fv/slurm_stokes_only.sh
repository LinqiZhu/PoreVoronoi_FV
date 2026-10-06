#!/bin/bash
#SBATCH --job-name=pvfv-stokes-only
#SBATCH --comment=pvfv-stokes-only
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=../outputs/%x-%j.out
# Submit from porevoronoi_fv/ with sbatch --partition=<your partition> slurm_stokes_only.sh after mkdir -p ../outputs
# One allocation, all six cases. Per case: G1 (production vs vectorised assembly) -> E0 (pure solve on the
# published partition) -> eig (spectral quantities for the declared E0-a velocity tolerance; capped in time and
# address space) -> a (gates E0-a, one JSON per case). Then one summary. NO assisted solve is run.
# Requeue-safe: every stage skips on a valid checkpoint; an interrupted stage restarts from its own start.
# Environment: PVFV_DIR = this folder (default: submit directory); PVFV_PYTHON = Python interpreter (default python3);
# PVFV_EIG_TIMEOUT (seconds) and PVFV_EIG_VMEM_KB cap the eig stage.
set -u
T="${PVFV_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY="${PVFV_PYTHON:-python3}"
EIG_TIMEOUT="${PVFV_EIG_TIMEOUT:-5400}"
EIG_VMEM_KB="${PVFV_EIG_VMEM_KB:-41943040}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$T" || exit 2
mkdir -p ../outputs
R="${SLURM_JOB_ID:-local}_${SLURM_RESTART_COUNT:-0}"
{ date -u +%FT%TZ; hostname; lscpu; grep -m1 "model name" /proc/cpuinfo; "$PY" -B -c "import sys,numpy,scipy;print(sys.version,numpy.__version__,scipy.__version__)"; sha256sum ./*.py ./*.sh config.json ../data/controlled_cases/*/* ../data/controlled_cases/expected/*; } > "../outputs/env_${R}.txt" 2>&1

stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
stamp "start restart=${SLURM_RESTART_COUNT:-0} tree=$T"
"$PY" -B evaluate_stokes_only.py --stage protocol || { stamp "protocol refused"; exit 3; }

( while sleep 300; do stamp "beat $(ls ../outputs/c?/e0.json ../outputs/c?/a.json 2>/dev/null | tr '\n' ' ')"; done ) &
BEAT=$!

run_case() {
  local c=$1 rc
  stamp "$c g1 begin"
  "$PY" -B run_stokes_only.py --case "$c" --stage g1; rc=$?; stamp "$c g1 end rc=$rc"
  stamp "$c e0 begin"
  "$PY" -B run_stokes_only.py --case "$c" --stage e0; rc=$?; stamp "$c e0 end rc=$rc"
  if [ $rc -ne 0 ]; then return; fi
  stamp "$c eig begin"
  ( ulimit -v "$EIG_VMEM_KB"; timeout "$EIG_TIMEOUT" "$PY" -B evaluate_stokes_only.py --case "$c" --stage eig ); rc=$?
  stamp "$c eig end rc=$rc"
  stamp "$c a begin"
  "$PY" -B evaluate_stokes_only.py --case "$c" --stage a; rc=$?; stamp "$c a end rc=$rc"
}

PIDS=()
for c in c1 c2 c3 c4 c5 c6; do
  run_case "$c" > "../outputs/log_${c}_${R}.txt" 2>&1 &
  PIDS+=("$!")
done
for p in "${PIDS[@]}"; do wait "$p"; done
kill "$BEAT" 2>/dev/null
"$PY" -B evaluate_stokes_only.py --stage summary
stamp "done"
