#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${TRAIN_DEVICE:-}" ]]; then
  TRAIN_DEVICE="$("$PYTHON_BIN" - <<'PY'
import torch
if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    print("mps")
elif torch.cuda.is_available():
    print("cuda:0")
else:
    print("cpu")
PY
)"
fi
RUN_NAME="${RUN_NAME:-mac_run}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/$RUN_NAME}"
SIZES="${SIZES:-2:20,3:25,2:30,3:50,4:75,5:100,5:150}"

TEACHER_DATASET="${TEACHER_DATASET:-$RUN_DIR/teacher_dataset.pt}"
OPTIONAL_WARMSTART_CHECKPOINT="${OPTIONAL_WARMSTART_CHECKPOINT:-$RUN_DIR/outcome_distilled_warmstart.pt}"
PPO_RESUME_CHECKPOINT="${PPO_RESUME_CHECKPOINT:-}"
PPO_CHECKPOINT_DIR="${PPO_CHECKPOINT_DIR:-$RUN_DIR/ppo}"
PPO_LOG="${PPO_LOG:-$RUN_DIR/ppo_train.jsonl}"
VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-$RUN_DIR/validation.json}"
MODE_COMPARISON_OUTPUT="${MODE_COMPARISON_OUTPUT:-$RUN_DIR/mode_comparison.json}"
VALIDITY_OUTPUT="${VALIDITY_OUTPUT:-$RUN_DIR/validity_tests.txt}"

mkdir -p "$RUN_DIR" "$PPO_CHECKPOINT_DIR"

if [[ "$TRAIN_DEVICE" == "mps" ]]; then
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-0}"
fi

export PLACEMENT_TEACHER_MAX_ITERS="${PLACEMENT_TEACHER_MAX_ITERS:-256}"
export PLACEMENT_TEACHER_STEP_SCALE="${PLACEMENT_TEACHER_STEP_SCALE:-1.2}"
export PLACEMENT_TEACHER_MARGIN="${PLACEMENT_TEACHER_MARGIN:-0.5}"

echo "Using device: $TRAIN_DEVICE"
echo "Run directory: $RUN_DIR"

"$PYTHON_BIN" - <<'PY'
import importlib
for module in ("torch", "matplotlib"):
    importlib.import_module(module)
PY

if [[ ! -f "$TEACHER_DATASET" ]]; then
  echo "Generating teacher dataset at $TEACHER_DATASET"
  TEACHER_CMD=(
    "$PYTHON_BIN" teacher_data.py
    --output "$TEACHER_DATASET"
    --teacher-solver "${TEACHER_SOLVER:-prior_solver:train_prior_solver_placement}"
    --num-cases "${TEACHER_CASES:-64}"
    --sizes "$SIZES"
    --device "$TRAIN_DEVICE"
    --max-demo-overlap "${MAX_DEMO_OVERLAP:-0.02}"
  )
  if [[ -n "${DAGGER_POLICY_CHECKPOINT:-}" ]]; then
    TEACHER_CMD+=(--dagger-policy-checkpoint "$DAGGER_POLICY_CHECKPOINT")
  fi
  if [[ -n "${DAGGER_CASES:-}" ]]; then
    TEACHER_CMD+=(--dagger-cases "$DAGGER_CASES")
  fi
  if [[ -n "${DAGGER_STEPS:-}" ]]; then
    TEACHER_CMD+=(--dagger-steps "$DAGGER_STEPS")
  fi
  "${TEACHER_CMD[@]}"
else
  echo "Reusing existing teacher dataset at $TEACHER_DATASET"
fi

if [[ "${WRITE_OPTIONAL_WARMSTART_CHECKPOINT:-0}" == "1" ]]; then
  echo "Writing standalone warm-start checkpoint to $OPTIONAL_WARMSTART_CHECKPOINT"
  "$PYTHON_BIN" distill.py \
    --teacher-dataset "$TEACHER_DATASET" \
    --output "$OPTIONAL_WARMSTART_CHECKPOINT" \
    --device "$TRAIN_DEVICE" \
    --epochs "${DISTILL_EPOCHS:-5}" \
    --batch-size "${DISTILL_BATCH_SIZE:-1}" \
    --lr "${DISTILL_LR:-1e-4}"
fi

echo "Training PPO policy with integrated outcome distillation and metric-gated hardening"
TRAIN_CMD=(
  "$PYTHON_BIN" train_ppo.py
  --device "$TRAIN_DEVICE"
  --teacher-dataset "$TEACHER_DATASET"
  --distill-epochs "${DISTILL_EPOCHS:-5}"
  --distill-batch-size "${DISTILL_BATCH_SIZE:-1}"
  --distill-lr "${DISTILL_LR:-1e-4}"
  --teacher-lambda0 "${TEACHER_LAMBDA0:-1.0}"
  --teacher-anneal-updates "${TEACHER_ANNEAL_UPDATES:-50}"
  --teacher-aux-batch-size "${TEACHER_AUX_BATCH_SIZE:-4}"
  --teacher-aux-steps-per-update "${TEACHER_AUX_STEPS_PER_UPDATE:-1}"
  --teacher-aux-lr-scale "${TEACHER_AUX_LR_SCALE:-0.10}"
  --teacher-aux-loss-cap "${TEACHER_AUX_LOSS_CAP:-256}"
  --teacher-aux-weight-cap "${TEACHER_AUX_WEIGHT_CAP:-0.25}"
  --updates "${PPO_UPDATES:-200}"
  --episodes-per-update "${EPISODES_PER_UPDATE:-8}"
  --horizon "${PPO_HORIZON:-4}"
  --coordinate-steps "${PPO_COORDINATE_STEPS:-8}"
  --ppo-epochs "${PPO_EPOCHS:-4}"
  --minibatch-size "${PPO_MINIBATCH_SIZE:-8}"
  --validation-interval "${VALIDATION_INTERVAL:-25}"
  --validation-episodes "${VALIDATION_EPISODES:-4}"
  --metric-gated-hardening
  --sizes "$SIZES"
  --checkpoint-dir "$PPO_CHECKPOINT_DIR"
  --log "$PPO_LOG"
)
if [[ -n "$PPO_RESUME_CHECKPOINT" ]]; then
  TRAIN_CMD+=(--resume-checkpoint "$PPO_RESUME_CHECKPOINT")
fi
"${TRAIN_CMD[@]}"

FINAL_CHECKPOINT="$PPO_CHECKPOINT_DIR/best_lexicographic.pt"
if [[ ! -f "$FINAL_CHECKPOINT" ]]; then
  FINAL_CHECKPOINT="$PPO_CHECKPOINT_DIR/ordering_policy.pt"
fi
if [[ ! -f "$FINAL_CHECKPOINT" ]]; then
  FINAL_CHECKPOINT="$PPO_CHECKPOINT_DIR/best_exact_overlap.pt"
fi
if [[ ! -f "$FINAL_CHECKPOINT" ]]; then
  FINAL_CHECKPOINT="$PPO_CHECKPOINT_DIR/ordering_policy_latest.pt"
fi

echo "Validating checkpoint $FINAL_CHECKPOINT"
"$PYTHON_BIN" validate_policy.py \
  --checkpoint "$FINAL_CHECKPOINT" \
  --device "$TRAIN_DEVICE" \
  --cases "${VALIDATE_CASES:-first10}" \
  --samples "${VALIDATE_SAMPLES:-4}" \
  --steps "${VALIDATE_STEPS:-32}" \
  --mode both \
  --output "$VALIDATION_OUTPUT"

echo "Comparing declared Mode A and Mode B"
"$PYTHON_BIN" compare_inference_modes.py \
  --checkpoint "$FINAL_CHECKPOINT" \
  --device "$TRAIN_DEVICE" \
  --cases "${VALIDATE_CASES:-first10}" \
  --rollouts "${MODE_COMPARE_ROLLOUTS:-4}" \
  --steps "${VALIDATE_STEPS:-32}" \
  --temperature "${VALIDATE_TEMPERATURE:-0.35}" \
  --output "$MODE_COMPARISON_OUTPUT"

echo "Running lightweight validity checks"
"$PYTHON_BIN" validity_tests.py > "$VALIDITY_OUTPUT"

echo "Finished."
echo "Validation written to $VALIDATION_OUTPUT"
echo "Mode comparison written to $MODE_COMPARISON_OUTPUT"
echo "Validity checks written to $VALIDITY_OUTPUT"
