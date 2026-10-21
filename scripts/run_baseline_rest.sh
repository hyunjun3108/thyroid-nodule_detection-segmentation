#!/usr/bin/env bash
# Phase 2 Step B: fold0 검증 통과 후 나머지 베이스라인 학습 일괄 실행.
# 남은 12개 run(nano 6 + small 6)을 GPU 0/1/2에 분산, GPU별 순차(백그라운드) 처리.
#   - pooled fold1~4 (n,s): --test 로 held-out tn3k_test 평가 포함
#   - LODO tn3k2ddti / ddti2tn3k (n,s): val==test(남긴 도메인)라 --test 생략
# 로그: runs/phase2_baseline/logs/<name>.log
set -u
cd /data/HJ_thyroid

PY=/data/my_envs/thyroid_seg/bin/python
NW=/data/HJ_thyroid/_phase0_check/yolo11n-seg.pt   # nano pretrained (로컬)
SW=yolo11s-seg.pt                                   # small pretrained (다운로드됨)
LOGDIR=runs/phase2_baseline/logs
mkdir -p "$LOGDIR"

# run 정의: "data.yaml|model|name|test플래그"
POOL=datasets/splits/pooled_5fold
LODO=datasets/splits/lodo

# GPU별 큐 (2 nano + 2 small씩 균형 배분)
declare -A QUEUE
QUEUE[0]="$POOL/fold1/data.yaml|$NW|n_fold1|--test
$POOL/fold2/data.yaml|$NW|n_fold2|--test
$POOL/fold1/data.yaml|$SW|s_fold1|--test
$POOL/fold2/data.yaml|$SW|s_fold2|--test"

QUEUE[1]="$POOL/fold3/data.yaml|$NW|n_fold3|--test
$POOL/fold4/data.yaml|$NW|n_fold4|--test
$POOL/fold3/data.yaml|$SW|s_fold3|--test
$POOL/fold4/data.yaml|$SW|s_fold4|--test"

QUEUE[2]="$LODO/tn3k2ddti/data.yaml|$NW|n_tn3k2ddti|
$LODO/ddti2tn3k/data.yaml|$NW|n_ddti2tn3k|
$LODO/tn3k2ddti/data.yaml|$SW|s_tn3k2ddti|
$LODO/ddti2tn3k/data.yaml|$SW|s_ddti2tn3k|"

run_queue() {
  local dev="$1"; local q="$2"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    IFS='|' read -r data model name testflag <<< "$line"
    echo "[GPU$dev] START $name ($(date +%H:%M:%S))" | tee -a "$LOGDIR/_driver.log"
    $PY scripts/train_seg.py --data "$data" --model "$model" \
        --name "$name" --device "$dev" $testflag \
        > "$LOGDIR/$name.log" 2>&1
    echo "[GPU$dev] DONE  $name exit=$? ($(date +%H:%M:%S))" | tee -a "$LOGDIR/_driver.log"
  done <<< "$q"
}

for dev in 0 1 2; do
  run_queue "$dev" "${QUEUE[$dev]}" &
done
wait
echo "=== ALL DONE ($(date +%H:%M:%S)) ===" | tee -a "$LOGDIR/_driver.log"
