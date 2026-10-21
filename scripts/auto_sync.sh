#!/usr/bin/env bash
# 프로젝트 변경분을 GitHub에 자동 커밋·푸시한다. cron이 주기적으로 호출한다.
#
#   - 변경이 없으면 아무것도 하지 않는다(빈 커밋 없음).
#   - 바뀐 최상위 파일·폴더마다 따로 커밋하고, 메시지는 그 항목의 역할 설명으로 한다.
#   - 매번 ultralytics 포크의 현재 상태로 ultralytics_patches/thyroid.patch 를 재생성한다.
#   - 50MB 넘는 파일은 커밋에서 빼고 로그에 경고한다(GitHub 100MB 제한 이전에 차단).
#   - flock 으로 동시 실행을 막는다(이전 실행이 push 중이면 이번 회차는 건너뜀).
#
# 수동 실행: bash scripts/auto_sync.sh
# 로그:      /data/HJ_thyroid/.git_auto_sync.log
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)   # 이 스크립트가 있는 저장소
FORK=$REPO/ultralytics_thyroid
LOG=$REPO/.git_auto_sync.log
MAX_MB=50

exec 9>"$REPO/.git/auto_sync.lock"
flock -n 9 || exit 0

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
cd "$REPO" || exit 1

# 로그가 무한히 커지지 않게 최근 2000줄만 유지
[ -f "$LOG" ] && tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"

# 1) 포크 변경분을 패치로 갱신 (권한 비트 변경은 무시)
if [ -d "$FORK/.git" ]; then
  tmp=$(mktemp)
  (
    cd "$FORK" &&
    git -c core.fileMode=false diff &&
    git status --porcelain | awk '$1=="??"{print $2}' | while read -r f; do
      git diff --no-index -- /dev/null "$f"
    done
  ) > "$tmp" 2>/dev/null
  if [ -s "$tmp" ] && ! cmp -s "$tmp" ultralytics_patches/thyroid.patch; then
    mkdir -p ultralytics_patches && mv "$tmp" ultralytics_patches/thyroid.patch
    log "ultralytics 패치 갱신"
  else
    rm -f "$tmp"
  fi
fi

# 커밋 메시지 = 최상위 파일·폴더의 역할. GitHub 파일 목록에서 각 항목 옆에 이 설명이 보인다.
declare -A ROLE=(
  [README.md]="프로젝트 개요 및 실행 방법"
  [.gitignore]="업로드 제외 파일 목록"
  [datasets]="데이터 분할 목록 (5-fold, LODO, 테스트)"
  [runs]="실험 결과 및 그림"
  [scripts]="전처리·학습·평가 코드"
  [thesis]="학위논문 원고"
  [ultralytics_patches]="YOLOv11 수정 사항 (경계 정교화 모듈)"
  [YOLOv11_갑상선결절_연구계획서.docx]="연구계획서"
  [갑상선결절_경계정교화_경량화_발표.pptx]="발표자료"
)

# 2) 바뀐 경로를 최상위 항목별로 묶는다
git add -A
mapfile -t groups < <(git -c core.quotepath=false diff --cached --name-only | awk -F/ '{print $1}' | sort -u)
git reset -q
[ ${#groups[@]} -eq 0 ] && exit 0   # 바뀐 것 없음

# 3) 항목마다 따로 커밋한다. 50MB 넘는 파일은 빼고 로그에 경고한다.
made=0
for top in "${groups[@]}"; do
  git add -A -- "$top"
  while IFS= read -r -d '' f; do
    if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt $((MAX_MB * 1024 * 1024)) ]; then
      git reset -q -- "$f"
      log "경고: ${MAX_MB}MB 초과로 제외 — $f  (.gitignore 에 추가하세요)"
    fi
  done < <(git -c core.quotepath=false diff --cached --name-only -z -- "$top")
  git diff --cached --quiet && continue
  git commit -q -m "${ROLE[$top]:-$top}" && made=1 && log "커밋: $top"
done
[ "$made" -eq 0 ] && exit 0

# 4) 푸시 — 원격이 아직 없으면 로컬 커밋만 쌓아 둔다
if git remote get-url origin >/dev/null 2>&1; then
  if out=$(git push -q origin HEAD 2>&1); then
    log "푸시 완료"
  else
    log "푸시 실패(다음 회차에 재시도): $(echo "$out" | tail -1)"
  fi
fi
