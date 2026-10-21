#!/usr/bin/env bash
# Phase 3 최종 보정: 오염됐던 fold0/1/2 만 승자 gain(1.0, 2.0)으로 깨끗이 재학습.
# fold3/4 + LODO 는 1차 실행 정상본을 그대로 재사용(건드리지 않음).
# fold0→GPU0, fold1→GPU1, fold2→GPU2 동시 실행.
# 중복 실행 가드: 대상 디렉토리가 이미 있으면 거부(누적 오염 방지). 재실행하려면 FORCE=1.
set -u
cd /data/HJ_thyroid

export BRM_HEAD_GAIN=1.0
export BRM_LAMBDA=2.0

PY=/data/my_envs/thyroid_seg/bin/python
MODEL=yolo11n-seg-brm.yaml
WEIGHTS=/data/HJ_thyroid/_phase0_check/yolo11n-seg.pt
PROJ=runs/phase3_brm_final
LOGDIR=$PROJ/logs
mkdir -p "$LOGDIR"

# --- 중복 실행 가드 ---
for i in 0 1 2; do
  if [ -d "$PROJ/nbrm_fold$i" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $PROJ/nbrm_fold$i 이미 존재. 누적 오염 방지로 중단."
    echo "  재학습하려면 먼저 해당 폴더들을 지우거나 FORCE=1 로 실행하세요."
    exit 1
  fi
done

echo "[redo] fold0/1/2 재학습 시작 (head=$BRM_HEAD_GAIN lam=$BRM_LAMBDA) $(date +%H:%M:%S)" | tee -a "$LOGDIR/_redo.log"
pids=()
for i in 0 1 2; do
  $PY scripts/train_seg.py --data "datasets/splits/pooled_5fold/fold$i/data.yaml" \
      --model "$MODEL" --weights "$WEIGHTS" \
      --name "nbrm_fold$i" --project "$PROJ" --device "$i" --test \
      > "$LOGDIR/nbrm_fold$i.log" 2>&1 &
  pids+=($!)
  echo "[redo] START nbrm_fold$i on GPU$i (pid ${pids[-1]})" | tee -a "$LOGDIR/_redo.log"
done

# 각 종료 대기 + exit code 기록
fail=0
for idx in 0 1 2; do
  if wait "${pids[$idx]}"; then rc=0; else rc=$?; fi
  echo "[redo] DONE nbrm_fold$idx exit=$rc $(date +%H:%M:%S)" | tee -a "$LOGDIR/_redo.log"
  [ "$rc" -ne 0 ] && fail=1
done
echo "=== REDO ALL DONE (fail=$fail) $(date +%H:%M:%S) ===" | tee -a "$LOGDIR/_redo.log"
