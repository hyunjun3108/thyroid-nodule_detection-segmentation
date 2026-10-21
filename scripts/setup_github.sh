#!/usr/bin/env bash
# GitHub 원격 연결 + 첫 푸시 + 자동 동기화(cron) 등록. 한 번만 실행하면 된다.
#
# 사전 준비(브라우저에서):
#   1) github.com 에서 빈 저장소 생성 (README·.gitignore 추가하지 말 것)
#   2) 저장소 Settings → Deploy keys → Add deploy key
#      ~/.ssh/hj_thyroid_deploy.pub 내용을 붙여넣고 "Allow write access" 체크
#
# 사용:  bash scripts/setup_github.sh <GitHub사용자명>/<저장소명> [주기(분), 기본 30]
set -eu
REPO=/data/HJ_thyroid
SLUG=${1:?사용법: bash scripts/setup_github.sh <owner>/<repo> [interval_min]}
EVERY=${2:-30}
cd "$REPO"

echo "== 1) 인증 확인 =="
# GitHub 는 셸을 주지 않으므로 성공해도 exit 1 — 메시지로 판정한다
msg=$(ssh -o BatchMode=yes -T git@github-hj-thyroid 2>&1 || true)
if ! echo "$msg" | grep -q "successfully authenticated"; then
  echo "인증 실패: $msg"
  echo "→ 배포 키를 등록했는지, 'Allow write access'를 체크했는지 확인하세요."
  exit 1
fi
echo "   $msg"

echo "== 2) 원격 연결 =="
url="git@github-hj-thyroid:${SLUG}.git"
git remote get-url origin >/dev/null 2>&1 && git remote set-url origin "$url" || git remote add origin "$url"
echo "   origin = $url"

echo "== 3) 첫 푸시 =="
git push -u origin main

echo "== 4) 자동 동기화 등록 (${EVERY}분마다) =="
line="*/${EVERY} * * * * /bin/bash $REPO/scripts/auto_sync.sh"
( crontab -l 2>/dev/null | grep -v "scripts/auto_sync.sh" ; echo "$line" ) | crontab -
crontab -l | grep auto_sync

echo
echo "완료. https://github.com/${SLUG}"
echo "동기화 로그: tail -f $REPO/.git_auto_sync.log"
