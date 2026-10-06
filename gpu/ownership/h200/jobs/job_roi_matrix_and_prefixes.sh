#!/bin/bash
#SBATCH --job-name=roi_h200
#SBATCH --account=<account>
#SBATCH --partition=<gpu-partition>
#SBATCH --time=00:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# Submit from gpu/ownership/h200 (sbatch jobs/job_roi_matrix_and_prefixes.sh); the job runs in that folder.
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
set -euo pipefail
umask 077
mkdir -p out/receipts
trap 'rc=$?; printf "{\"job_id\":\"%s\",\"host\":\"%s\",\"exit_code\":%s}\n" "$SLURM_JOB_ID" "$(hostname)" "$rc" > "out/receipts/${SLURM_JOB_ID}.json"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_PATH=/usr/local/cuda-12.8
export LD_PRELOAD=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
export CUPY_COMPILE_WITH_PTX=1
export CUPY_CACHE_DIR="$PWD/.cupy_cache"
printf 'START %s %s %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
${PYTHON:-python} -c "import cupy; cupy.show_config()"
${PYTHON:-python} -u run_roi_matrix.py
${PYTHON:-python} -u time_paper_prefixes.py
printf 'FINISH %s %s\n' "$SLURM_JOB_ID" "$(date -Is)"
