#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 JOB_ID" >&2
    exit 2
fi

JOB_ID="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/sanity_check/imp"

echo "Queue state"
squeue -j "${JOB_ID}" -o '%.12i %.10P %.20j %.2t %.10M %.10l %.4C %.10m %b %R'

echo
echo "Accounting state"
sacct -j "${JOB_ID}" -X -o JobID,JobName,State,Elapsed,Timelimit,AllocCPUS,ReqMem,ReqTRES,ExitCode

if squeue -h -j "${JOB_ID}" | grep -q .; then
    echo
    echo "Live resource usage (may be empty during startup)"
    sstat -j "${JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,AveRSS -P 2>/dev/null || true
fi

echo
echo "Storage"
df -h "${OUTPUT_DIR}"
du -sh "${OUTPUT_DIR}"

echo
echo "Completed IMP rounds"
for seed in 42 45 46; do
    complete="$(find "${OUTPUT_DIR}/seed_${seed}" -mindepth 2 -maxdepth 2 -name status.json -exec grep -l '"status": "complete"' {} + 2>/dev/null | wc -l)"
    printf 'seed %s: %s/13\n' "${seed}" "${complete}"
done

if [[ -f "${OUTPUT_DIR}/slurm_stop_reason.txt" ]]; then
    echo
    echo "Safety stop reason"
    cat "${OUTPUT_DIR}/slurm_stop_reason.txt"
fi

echo
echo "Recent log output"
tail -n 30 "${PROJECT_ROOT}/logs/imp-${JOB_ID}.out" 2>/dev/null || true
tail -n 30 "${PROJECT_ROOT}/logs/imp-${JOB_ID}.err" 2>/dev/null || true

