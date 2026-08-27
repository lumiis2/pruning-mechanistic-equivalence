#!/usr/bin/env bash
#SBATCH --job-name=inspect-gpu
#SBATCH --partition=new_p_general
#SBATCH --account=general_users
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1080Ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/inspect-gpu-%j.out
#SBATCH --error=logs/inspect-gpu-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${PROJECT_ROOT}"

if [[ ! -f .venv310/bin/activate ]]; then
    echo "Missing ${PROJECT_ROOT}/.venv310" >&2
    exit 2
fi

module purge
module load python/3.10.13
module load pytorch/2.5.1-py3.10-cuda12.8
source .venv310/bin/activate

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export PYTHONPATH=.

echo "=== SLURM ==="
date --iso-8601=seconds
echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURM_JOB_NODELIST}"
echo "visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo

echo "=== MODULES ==="
module list 2>&1
echo "python=$(command -v python)"
python --version
echo

echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv
python - <<'PY'
import torch

print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("visible_gpu_count=", torch.cuda.device_count())
assert torch.cuda.is_available(), "allocated GPU is unavailable to PyTorch"
assert torch.cuda.device_count() == 1, "the job should expose exactly one GPU"
left = torch.tensor([[1.0, 2.0]], device="cuda")
right = torch.tensor([[3.0], [4.0]], device="cuda")
result = left @ right
torch.cuda.synchronize()
assert result.item() == 11.0
print("gpu_name=", torch.cuda.get_device_name(0))
print("cuda_matmul=", result.item())
PY
echo

echo "=== DEPENDENCIES ==="
python - <<'PY'
import PIL
import matplotlib
import pytest
import yaml

print("Pillow=", PIL.__version__)
print("matplotlib=", matplotlib.__version__)
print("pytest=", pytest.__version__)
print("PyYAML=", yaml.__version__)
PY
echo

echo "=== TESTS ==="
python -m pytest -q
echo
echo "ENVIRONMENT INSPECTION PASSED"
