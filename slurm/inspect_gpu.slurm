#!/usr/bin/env bash
#SBATCH --job-name=inspect-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/inspect-gpu-%j.out
#SBATCH --error=slurm/logs/inspect-gpu-%j.err

set -uo pipefail

echo "=== SLURM JOB ==="
date --iso-8601=seconds
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "job_name=${SLURM_JOB_NAME:-unknown}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

echo
echo "=== HOST ==="
hostname
uname -a
uptime

echo
echo "=== CPU ==="
lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\)|NUMA|MHz' || true

echo
echo "=== MEMORY ==="
free -h

echo
echo "=== NVIDIA-SMI ==="
nvidia-smi
nvidia_smi_status=$?

echo
echo "=== GPU DETAILS ==="
nvidia-smi \
  --query-gpu=index,name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit \
  --format=csv || true

echo
echo "=== SOFTWARE ==="
echo "python=$(command -v python3 || true)"
python3 --version || true
echo "nvcc=$(command -v nvcc || true)"
nvcc --version || true

echo
echo "=== PYTORCH ==="
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "Activated $(pwd)/.venv"
fi

python - <<'PY' || true
import platform

print("Python:", platform.python_version())
try:
    import torch
except ImportError as error:
    print("PyTorch import failed:", error)
else:
    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Visible GPU count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {properties.name}; "
            f"VRAM={properties.total_memory / 1024**3:.2f} GiB; "
            f"compute_capability={properties.major}.{properties.minor}"
        )
PY

echo
if [[ $nvidia_smi_status -eq 0 ]]; then
  echo "GPU INSPECTION PASSED"
else
  echo "GPU INSPECTION FAILED: nvidia-smi exited with status $nvidia_smi_status"
  exit "$nvidia_smi_status"
fi
