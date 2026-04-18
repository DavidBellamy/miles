#!/usr/bin/env bash
# Drive Qwen3.6-35B-A3B MTP RL training through multiple parallelism configs.
# Each run: 10 rollouts, smoke test for logprob diff < 0.02.

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULT_DIR="${RESULT_DIR:-/root/shared_data/qwen36_configs}"
mkdir -p "$RESULT_DIR"

# ------------------------------------------------------------
# Configurations on 8xH200 (TP:EP:CP:PP:ETP)
# Constraint: world = PP*TP*CP*DP = PP*ETP*EP*EDP = 8
# ------------------------------------------------------------
CONFIGS=(
  "EP8:1:8:1:1:1"
  "CP2_EP8:1:8:2:1:1"
  "PP2_CP4:1:2:4:2:1"
  "PP2_EP2_TP2:2:2:1:2:1"
  "PP2_EP2_CP2:1:2:2:2:1"
  "CP2_TP4:4:1:2:1:1"       # TP>num_kv_heads(=2); expected to fail unless KV replication
  "TP8:8:1:1:1:1"           # same caveat
  "TPEP8:8:1:1:1:8"         # TP=ETP=8 EP=1
)

run_one() {
  local name="$1" tp="$2" ep="$3" cp="$4" pp="$5" etp="$6"
  local log="$RESULT_DIR/${name}.log"

  echo "===================================================="
  echo ">>> Running $name (TP=$tp EP=$ep CP=$cp PP=$pp ETP=$etp)"
  echo ">>> Log: $log"
  echo "===================================================="

  # Clean ray state between runs
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 sglang 2>/dev/null || true
  sleep 5
  rm -rf /tmp/ray/session_* || true

  env \
    -u MILES_SCRIPT_OUTPUT_DIR \
    -u MILES_SCRIPT_DATA_DIR \
    -u MILES_SCRIPT_MODEL_DIR \
    python3 scripts/run_qwen3_6_35b_a3b_mtp.py \
      --tp "$tp" --ep "$ep" --cp "$cp" --pp "$pp" --etp "$etp" \
      --num-rollout 10 \
      --rollout-max-response-len 1024 \
      --skip-prepare \
      >"$log" 2>&1 && echo ">>> $name OK" || echo ">>> $name FAILED (see $log)"
}

for entry in "${CONFIGS[@]}"; do
  IFS=":" read -r NAME TP EP CP PP ETP <<< "$entry"
  run_one "$NAME" "$TP" "$EP" "$CP" "$PP" "$ETP"
done

# ------------------------------------------------------------
# Summarise logprob diff per config
# ------------------------------------------------------------
echo
echo "==================== SUMMARY ===================="
printf "%-20s %-10s %-10s %-10s\n" "CONFIG" "steps" "abs_diff" "status"
for entry in "${CONFIGS[@]}"; do
  IFS=":" read -r NAME _ _ _ _ _ <<< "$entry"
  LOG="$RESULT_DIR/${NAME}.log"
  if [[ ! -f "$LOG" ]]; then
    printf "%-20s %-10s %-10s %-10s\n" "$NAME" "-" "-" "missing"
    continue
  fi
  # last reported abs_diff
  DIFF=$(grep -aoE "train_rollout_logprob_abs_diff['\"]?[: ]+[0-9.eE+-]+" "$LOG" | tail -1 | grep -oE "[0-9.eE+-]+$")
  STEPS=$(grep -acE "rollout_id=[0-9]+" "$LOG")
  if [[ -z "$DIFF" ]]; then
    printf "%-20s %-10s %-10s %-10s\n" "$NAME" "$STEPS" "-" "NO_LOGPROB"
  else
    STATUS=$(awk -v d="$DIFF" 'BEGIN { print (d+0 < 0.02) ? "PASS" : "FAIL" }')
    printf "%-20s %-10s %-10s %-10s\n" "$NAME" "$STEPS" "$DIFF" "$STATUS"
  fi
done
