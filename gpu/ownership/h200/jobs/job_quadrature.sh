#!/bin/bash
#SBATCH --job-name=quadrature_h200
#SBATCH --account=<account>
#SBATCH --partition=<gpu-partition>
#SBATCH --qos=normal
#SBATCH --time=01:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Submit from gpu/ownership/h200 (sbatch jobs/job_quadrature.sh); the job runs in that folder.
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
set -euo pipefail
umask 077
mkdir -p out/receipts
trap 'rc=$?; printf "{\"job_id\":\"%s\",\"host\":\"%s\",\"exit_code\":%s}\n" "$SLURM_JOB_ID" "$(hostname)" "$rc" > "out/receipts/${SLURM_JOB_ID}.json"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-12.8}
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
export CUPY_CACHE_DIR="$PWD/.cupy_cache"
printf 'START %s %s %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
:
${PYTHON:-python} -u run_quadrature.py
printf 'FINISH %s %s\n' "$SLURM_JOB_ID" "$(date -Is)"
