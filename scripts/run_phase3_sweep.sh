#!/usr/bin/env bash
# Phase 3-A: 경계 gain 스윕. head_gain × λ 그리드를 fold0·fold2에서 학습(빠른 신호).
#   configs: head_gain ∈ {0.25, 1.0}, lambda ∈ {2, 10}
#   folds  : 0, 2 (진단에서 효과 갈렸던 fold)  → 4×2 = 8 run, --test 포함
# 결과: runs/phase3_sweep/sw_h{HEAD}_l{LAM}_fold{i}
# 로그: runs/phase3_sweep/logs/<name>.log
set -u
cd /data/HJ_thyroid

PY=/data/my_envs/thyroid_seg/bin/python
MODEL=yolo11n-seg-brm.yaml
WEIGHTS=/data/HJ_thyroid/_phase0_check/yolo11n-seg.pt
PROJ=runs/phase3_sweep
LOGDIR=$PROJ/logs
mkdir -p "$LOGDIR"
POOL=datasets/splits/pooled_5fold

# "head|lambda|tag"
CONFIGS=("0.25|2|h025_l2" "0.25|10|h025_l10" "1.0|2|h100_l2" "1.0|10|h100_l10")
FOLDS=(0 2)

# 플랫 작업 리스트 생성 후 GPU 0/1/2 라운드로빈 분배
JOBS=()
for c in "${CONFIGS[@]}"; do
  for f in "${FOLDS[@]}"; do
    JOBS+=("$c|$f")
  done
done

declare -A QUEUE
idx=0
for j in "${JOBS[@]}"; do
  dev=$(( idx % 3 ))
  QUEUE[$dev]+="$j"$'\n'
  idx=$(( idx + 1 ))
done

run_queue() {
  local dev="$1"; local q="$2"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    IFS='|' read -r head lam tag fold <<< "$line"
    local name="sw_${tag}_fold${fold}"
    echo "[GPU$dev] START $name (head=$head lam=$lam) ($(date +%H:%M:%S))" | tee -a "$LOGDIR/_driver.log"
    BRM_HEAD_GAIN=$head BRM_LAMBDA=$lam \
    $PY scripts/train_seg.py --data "$POOL/fold${fold}/data.yaml" \
        --model "$MODEL" --weights "$WEIGHTS" \
        --name "$name" --project "$PROJ" --device "$dev" --test \
        > "$LOGDIR/$name.log" 2>&1
    echo "[GPU$dev] DONE  $name exit=$? ($(date +%H:%M:%S))" | tee -a "$LOGDIR/_driver.log"
  done <<< "$q"
}

for dev in 0 1 2; do
  run_queue "$dev" "${QUEUE[$dev]}" &
done
wait
echo "=== ALL DONE ($(date +%H:%M:%S)) ===" | tee -a "$LOGDIR/_driver.log"
