#!/usr/bin/env bash
set -euo pipefail

# Process-agnostic active-learning loop. Keep campaign CSV columns as
# caseId,x,y,id; translate x/y into simulator inputs inside BATCH_RUNNER.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-campaign}"
BATCH_RUNNER="${BATCH_RUNNER:?set BATCH_RUNNER to a batch adapter executable}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-11}"

# Replace these bounds/scales for the physical process. The model sees only x/y.
X_MIN="${X_MIN:-1}"
X_MAX="${X_MAX:-100}"
Y_MIN="${Y_MIN:-0.001}"
Y_MAX="${Y_MAX:-0.1}"
X_SCALE="${X_SCALE:-log10}"
Y_SCALE="${Y_SCALE:-log10}"

mkdir -p "${CAMPAIGN_DIR}/contours" "${CAMPAIGN_DIR}/proposals" "${CAMPAIGN_DIR}/completed"
if [[ ! -f "${CAMPAIGN_DIR}/completed/Sweep-0_completed.csv" ]]; then
  echo "Missing ${CAMPAIGN_DIR}/completed/Sweep-0_completed.csv" >&2
  exit 2
fi

MODEL_ARGS=(
  --mode monotone-y
  --contour-fit local-linear
  --x-scale "${X_SCALE}"
  --y-scale "${Y_SCALE}"
  --x-min "${X_MIN}"
  --x-max "${X_MAX}"
  --y-min "${Y_MIN}"
  --y-max "${Y_MAX}"
  --grid-size 21
  --posterior-samples 0
  --transition-width 0.04
  --label-noise 0.005
  --length-scale-x 0.18
)

last_completed="$(python3 -c 'import glob,re,sys; values=[int(m.group(1)) for p in glob.glob(sys.argv[1] + "/completed/Sweep-*_completed.csv") if (m := re.search(r"Sweep-(\d+)_completed\.csv$", p))]; print(max(values, default=0))' "${CAMPAIGN_DIR}")"

for ((iteration = last_completed + 1; iteration <= MAX_ITERATIONS; iteration++)); do
  completed_files=("${CAMPAIGN_DIR}"/completed/Sweep-*_completed.csv)
  contour_file="${CAMPAIGN_DIR}/contours/Sweep-$((iteration - 1))_contour.csv"
  python3 "${ROOT_DIR}/assess_contour.py" \
    "${completed_files[@]}" \
    --outfile "${contour_file}" \
    --state "${CAMPAIGN_DIR}/state.json" \
    --iteration "$((iteration - 1))" \
    "${MODEL_ARGS[@]}"

  status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${CAMPAIGN_DIR}/state.json")"
  if [[ "${status}" == "converged" ]]; then
    echo "Contour converged after $((iteration - 1)) completed batches."
    exit 0
  fi
  if [[ "${status}" == "stalled" ]]; then
    echo "Contour is stable but unresolved. Inspect state.json, adjust parameters, and rearm." >&2
    exit 3
  fi

  proposal_file="${CAMPAIGN_DIR}/proposals/Sweep-${iteration}_proposed.csv"
  completed_file="${CAMPAIGN_DIR}/completed/Sweep-${iteration}_completed.csv"
  python3 "${ROOT_DIR}/propose_next_sweep.py" \
    "${completed_files[@]}" \
    --outfile "${proposal_file}" \
    --n-simulations "${BATCH_SIZE}" \
    --n-new "${BATCH_SIZE}" \
    --n-repeats 0 \
    --seed "$((SEED + iteration))" \
    "${MODEL_ARGS[@]}"

  # Contract: adapter INPUT_PROPOSALS.csv OUTPUT_COMPLETED.csv
  # The adapter may fan out through a scheduler, but must preserve caseId,x,y
  # and replace every id=-1 with id=0 or id=1 before returning.
  "${BATCH_RUNNER}" "${proposal_file}" "${completed_file}"
done

completed_files=("${CAMPAIGN_DIR}"/completed/Sweep-*_completed.csv)
final_iteration="$(python3 -c 'import glob,re,sys; values=[int(m.group(1)) for p in glob.glob(sys.argv[1] + "/completed/Sweep-*_completed.csv") if (m := re.search(r"Sweep-(\d+)_completed\.csv$", p))]; print(max(values, default=0))' "${CAMPAIGN_DIR}")"
python3 "${ROOT_DIR}/assess_contour.py" \
  "${completed_files[@]}" \
  --outfile "${CAMPAIGN_DIR}/contours/Sweep-${final_iteration}_contour.csv" \
  --state "${CAMPAIGN_DIR}/state.json" \
  --iteration "${final_iteration}" \
  "${MODEL_ARGS[@]}"
echo "Reached MAX_ITERATIONS=${MAX_ITERATIONS}; inspect ${CAMPAIGN_DIR}/state.json and rearm if needed."
