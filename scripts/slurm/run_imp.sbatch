#!/usr/bin/env bash
#SBATCH --job-name=shape-color-imp
#SBATCH --partition=new_p_general
#SBATCH --account=general_users
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/imp-%j.out
#SBATCH --error=logs/imp-%j.err

set -Eeuo pipefail

# Safety limits. df reports KiB; du -sk also reports KiB.
MIN_SHARED_FREE_KIB=$((500 * 1024 * 1024))
MAX_IMP_OUTPUT_KIB=$((5 * 1024 * 1024))
CHECK_INTERVAL_SECONDS=300

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the repository root}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/sanity_check/imp"
STOP_REASON_FILE="${OUTPUT_DIR}/slurm_stop_reason.txt"
WORKLOAD_PID=""
WATCHDOG_PID=""

timestamp() {
    date --iso-8601=seconds
}

available_kib() {
    df -Pk "${OUTPUT_DIR}" | awk 'NR == 2 {print $4}'
}

output_size_kib() {
    du -sk "${OUTPUT_DIR}" | awk '{print $1}'
}

check_storage() {
    local free_kib used_kib
    free_kib="$(available_kib)"
    used_kib="$(output_size_kib)"

    if (( free_kib < MIN_SHARED_FREE_KIB )); then
        printf '%s Shared filesystem has only %s KiB free; required minimum is %s KiB.\n' \
            "$(timestamp)" "${free_kib}" "${MIN_SHARED_FREE_KIB}" | tee "${STOP_REASON_FILE}" >&2
        return 1
    fi
    if (( used_kib > MAX_IMP_OUTPUT_KIB )); then
        printf '%s IMP output is %s KiB; configured maximum is %s KiB.\n' \
            "$(timestamp)" "${used_kib}" "${MAX_IMP_OUTPUT_KIB}" | tee "${STOP_REASON_FILE}" >&2
        return 1
    fi
}

stop_workload() {
    if [[ -n "${WORKLOAD_PID}" ]] && kill -0 "${WORKLOAD_PID}" 2>/dev/null; then
        kill -TERM "${WORKLOAD_PID}" 2>/dev/null || true
    fi
}

handle_signal() {
    printf '%s Batch job received a termination signal; stopping the active round.\n' \
        "$(timestamp)" | tee "${STOP_REASON_FILE}" >&2
    stop_workload
    exit 143
}

cleanup() {
    if [[ -n "${WATCHDOG_PID}" ]]; then
        kill "${WATCHDOG_PID}" 2>/dev/null || true
    fi
    stop_workload
}

watch_storage() {
    while kill -0 "${WORKLOAD_PID}" 2>/dev/null; do
        sleep "${CHECK_INTERVAL_SECONDS}"
        if ! check_storage; then
            stop_workload
            return 1
        fi
    done
}

trap handle_signal TERM INT
trap cleanup EXIT

cd "${PROJECT_ROOT}"
if [[ ! -f scripts/run_imp.py || ! -f configs/sanity_check.yaml ]]; then
    echo "Submit from the pruning-mechanistic-equivalence repository root." >&2
    exit 2
fi
if [[ ! -f .venv310/bin/activate ]]; then
    echo "Missing .venv310. Follow the environment setup instructions before submitting." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
rm -f "${STOP_REASON_FILE}"
check_storage

module purge
module load python/3.10.13
module load pytorch/2.5.1-py3.10-cuda12.8
source .venv310/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

python -c 'import torch, matplotlib, yaml, PIL; assert torch.cuda.is_available(), "Slurm allocated a GPU but PyTorch cannot use it"'

echo "$(timestamp) job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} output=${OUTPUT_DIR}"
echo "Resources: cpus=${SLURM_CPUS_PER_TASK} memory=12G gpu=${CUDA_VISIBLE_DEVICES:-unknown}"
echo "Safety: minimum_shared_free=500GiB maximum_imp_output=5GiB check_interval=${CHECK_INTERVAL_SECONDS}s"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --kill-on-bad-exit=1 python scripts/run_imp.py \
    --config configs/sanity_check.yaml \
    --output-dir "${OUTPUT_DIR}" \
    --resume &
WORKLOAD_PID=$!
watch_storage &
WATCHDOG_PID=$!

set +e
wait "${WORKLOAD_PID}"
workload_status=$?
set -e

kill "${WATCHDOG_PID}" 2>/dev/null || true
wait "${WATCHDOG_PID}" 2>/dev/null || true
WATCHDOG_PID=""

if (( workload_status != 0 )); then
    echo "$(timestamp) IMP exited with status ${workload_status}. Re-submission is safe: completed rounds resume." >&2
    exit "${workload_status}"
fi

check_storage
echo "$(timestamp) IMP completed successfully."

