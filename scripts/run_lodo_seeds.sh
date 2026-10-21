#!/usr/bin/env bash
# Phase 5 확인 실험: LODO(교차 데이터셋)를 시드 43/44 로 반복해서
# "BRM이 cross-domain 에서 나빠진다"가 실행간 변동인지 실제 효과인지 분리한다.
#
# 시드 42 는 이미 있음(runs/phase2_baseline/n_*2*, runs/phase3_brm_final/nbrm_*2*).
# 여기서는 두 방향 × {baseline, BRM} × 시드 {43,44} = 8회를 GPU 0/1/2 에 나눠 돌린다.
# 출력: runs/phase5_lodo_seeds/{n,nbrm}_{tn3k2ddti,ddti2tn3k}_s{43,44}
#
# LODO 는 val==test(타깃 도메인)이라 --test 는 생략한다.
# 중복 실행 가드: 대상 디렉토리가 이미 있으면 거부. 재실행하려면 FORCE=1.
set -u
cd /data/HJ_thyroid

export BRM_HEAD_GAIN=1.0
export BRM_LAMBDA=2.0

PY=/data/my_envs/thyroid_seg/bin/python
WEIGHTS=/data/HJ_thyroid/_phase0_check/yolo11n-seg.pt
PROJ=runs/phase5_lodo_seeds
LOGDIR=$PROJ/logs
mkdir -p "$LOGDIR"

LODO=datasets/splits/lodo

# "data.yaml|이름|모델" — baseline 은 .pt 직접, BRM 은 yaml+weights 이식
JOBS=()
for d in tn3k2ddti ddti2tn3k; do
  for s in 43 44; do
    JOBS+=("$LODO/$d/data.yaml|n_${d}_s${s}|$WEIGHTS|$s")
    JOBS+=("$LODO/$d/data.yaml|nbrm_${d}_s${s}|yolo11n-seg-brm.yaml|$s")
  done
done

# --- 중복 실행 가드 ---
for j in "${JOBS[@]}"; do
  IFS='|' read -r _ name _ _ <<< "$j"
  if [ -d "$PROJ/$name" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $PROJ/$name 이미 존재. 누적 오염 방지로 중단(FORCE=1 로 무시)."
    exit 1
  fi
done

run_one() {
  IFS='|' read -r data name model seed <<< "$1"
  local dev="$2"
  local wflag=""
  [[ "$model" == *.yaml ]] && wflag="--weights $WEIGHTS"
  echo "[GPU$dev] START $name (seed=$seed) $(date +%H:%M:%S)" | tee -a "$LOGDIR/_driver.log"
  $PY scripts/train_seg.py --data "$data" --model "$model" $wflag \
      --name "$name" --project "$PROJ" --device "$dev" --seed "$seed" \
      > "$LOGDIR/$name.log" 2>&1
  echo "[GPU$dev] DONE  $name exit=$? $(date +%H:%M:%S)" | tee -a "$LOGDIR/_driver.log"
}

# GPU 0/1/2 에 라운드로빈으로 배분, 각 GPU 는 자기 큐를 순차 실행
for dev in 0 1 2; do
  (
    for ((i = dev; i < ${#JOBS[@]}; i += 3)); do
      run_one "${JOBS[$i]}" "$dev"
    done
  ) &
done
wait
echo "=== LODO SEEDS ALL DONE $(date +%H:%M:%S) ===" | tee -a "$LOGDIR/_driver.log"
