#!/usr/bin/env bash
# Phase 3: BRM nano(yolo11n-seg-brm) 학습 — Phase 2와 동일 조건(5-fold + LODO)으로 공정 비교.
#   - pooled fold0~4: --test 로 held-out tn3k_test 평가 포함
#   - LODO tn3k2ddti / ddti2tn3k: val==test 라 --test 생략
#   - 아키텍처=yolo11n-seg-brm.yaml, pretrained=_phase0_check/yolo11n-seg.pt 이식(.load)
# 로그: runs/phase3_brm/logs/<name>.log
set -u
cd /data/HJ_thyroid

PY=/data/my_envs/thyroid_seg/bin/python
MODEL=yolo11n-seg-brm.yaml
WEIGHTS=/data/HJ_thyroid/_phase0_check/yolo11n-seg.pt
PROJ=runs/phase3_brm
LOGDIR=$PROJ/logs
mkdir -p "$LOGDIR"

POOL=datasets/splits/pooled_5fold
LODO=datasets/splits/lodo

# GPU별 큐: "data.yaml|name|test플래그" (7 run을 3/2/2로 분산)
declare -A QUEUE
QUEUE[0]="$POOL/fold0/data.yaml|nbrm_fold0|--test
$POOL/fold3/data.yaml|nbrm_fold3|--test
$LODO/ddti2tn3k/data.yaml|nbrm_ddti2tn3k|"

QUEUE[1]="$POOL/fold1/data.yaml|nbrm_fold1|--test
$POOL/fold4/data.yaml|nbrm_fold4|--test"

QUEUE[2]="$POOL/fold2/data.yaml|nbrm_fold2|--test
$LODO/tn3k2ddti/data.yaml|nbrm_tn3k2ddti|"

run_queue() {
  local dev="$1"; local q="$2"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    IFS='|' read -r data name testflag <<< "$line"
    echo "[GPU$dev] START $name ($(date +%H:%M:%S))" | tee -a "$LOGDIR/_driver.log"
    $PY scripts/train_seg.py --data "$data" --model "$MODEL" --weights "$WEIGHTS" \
        --name "$name" --project "$PROJ" --device "$dev" $testflag \
        > "$LOGDIR/$name.log" 2>&1
    echo "[GPU$dev] DONE  $name exit=$? ($(date +%H:%M:%S))" | tee -a "$LOGDIR/_driver.log"
  done <<< "$q"
}

for dev in 0 1 2; do
  run_queue "$dev" "${QUEUE[$dev]}" &
done
wait
echo "=== ALL DONE ($(date +%H:%M:%S)) ===" | tee -a "$LOGDIR/_driver.log"
