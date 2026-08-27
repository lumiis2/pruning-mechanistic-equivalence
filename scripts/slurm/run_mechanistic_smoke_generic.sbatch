#!/usr/bin/env bash
#SBATCH --job-name=mech-smoke-gpu
#SBATCH --partition=new_p_general
#SBATCH --account=general_users
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=logs/mech-smoke-generic-%j.out
#SBATCH --error=logs/mech-smoke-generic-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${PROJECT_ROOT}"

module purge
module load python/3.10.13
module load pytorch/2.5.1-py3.10-cuda12.8
source .venv310/bin/activate

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

echo "job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv
python - <<'PY'
import torch

print("torch=", torch.__version__)
print("torch_file=", torch.__file__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
assert torch.cuda.is_available()
value = torch.tensor([[1.0, 2.0]], device="cuda") @ torch.tensor([[3.0], [4.0]], device="cuda")
torch.cuda.synchronize()
assert value.item() == 11.0
print("gpu=", torch.cuda.get_device_name(0))
print("cuda_matmul=", value.item())
PY

python scripts/run_mechanistic_characterization.py \
    --split discovery \
    --smoke-test \
    --no-resume \
    --output-dir outputs/mechanistic_characterization_validation

echo "MECHANISTIC DISCOVERY SMOKE PASSED"
