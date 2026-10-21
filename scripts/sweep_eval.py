#!/usr/bin/env python
"""
Phase 3-A 경계 gain 스윕 결과표 — head_gain × λ 그리드의 경계 지표를 한 표로.

스윕(runs/phase3_sweep, fold 0·2)과 함께 비교 기준 두 가지를 같은 조건으로 평가한다.
  - baseline nano        : runs/phase2_baseline/n_fold{0,2}
  - 초기 기본값 (0.5, 5) : runs/phase3_brm/nbrm_fold{0,2}   ← 스윕 전에 쓰던 값
held-out tn3k_test 614장, conf 0.25, 이미지 단위 평균 후 fold 평균.

출력: runs/phase3_sweep/sweep_table.csv        (--split test, held-out)
      runs/phase3_sweep/sweep_table_val.csv    (--split val, 각 fold 검증 분할 — 하이퍼파라미터 선택용)
이미지별 값은 runs/stats/per_image_cache.json 에 캐시(stats_test.py 와 공유).

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/sweep_eval.py --device 0 [--split val]

held-out 집합으로 계수를 고르면 최종 평가가 낙관적으로 편향되므로, 선택의 근거는 --split val 결과를 쓴다.
"""
import os
import sys
import csv
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_boundary as EB
from stats_test import Cache

ROOT = "/data/HJ_thyroid"
OUT = f"{ROOT}/runs/phase3_sweep/sweep_table.csv"
FOLDS = [0, 2]

# (라벨, head_gain, λ, 캐시키 접두사, 가중치 경로 템플릿, test_metrics 경로 템플릿)
CONFIGS = [
    ("baseline", None, None, "A|base|f",
     f"{ROOT}/runs/phase2_baseline/n_fold{{f}}/weights/best.pt",
     f"{ROOT}/runs/phase2_baseline/n_fold{{f}}_test/test_metrics.json"),
    ("initial (0.5, 5)", 0.5, 5.0, "S|h050_l5|f",
     f"{ROOT}/runs/phase3_brm/nbrm_fold{{f}}/weights/best.pt",
     f"{ROOT}/runs/phase3_brm/nbrm_fold{{f}}_test/test_metrics.json"),
] + [
    (f"h={h} λ={l}", h, l, f"S|{tag}|f",
     f"{ROOT}/runs/phase3_sweep/sw_{tag}_fold{{f}}/weights/best.pt",
     f"{ROOT}/runs/phase3_sweep/sw_{tag}_fold{{f}}_test/test_metrics.json")
    for h, l, tag in [(0.25, 2.0, "h025_l2"), (0.25, 10.0, "h025_l10"),
                      (1.0, 2.0, "h100_l2"), (1.0, 10.0, "h100_l10")]
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--cache", default=f"{ROOT}/runs/stats/per_image_cache.json")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    args = ap.parse_args()

    test_imgs = [l.strip() for l in open(EB.TEST_LIST) if l.strip()]
    cache = Cache(args.cache)

    def images(f):
        if args.split == "test":
            return test_imgs
        return [l.strip() for l in open(f"{ROOT}/datasets/splits/pooled_5fold/fold{f}/val.txt") if l.strip()]
    rows = []
    for label, h, l, key, wtmpl, mtmpl in CONFIGS:
        per_fold = []
        for f in FOLDS:
            # test: 베이스라인 키가 stats_test.py 와 같은 이름("A|base|f0")이라 그대로 재사용된다
            ck = f"{key}{f}" if args.split == "test" else f"V|{key}{f}"
            per = cache.get_or_run(ck, wtmpl.format(f=f), images(f), args.device)
            v = np.array(list(per.values()))        # (N,3) = biou, hd, dice
            if args.split == "test":
                with open(mtmpl.format(f=f)) as fh:
                    mp = float(json.load(fh)["metrics/mAP50-95(M)"])
            else:   # 검증 mAP 는 학습 중 기록된 체크포인트 epoch 값
                from make_thesis import best_epoch_metrics
                row, _, _ = best_epoch_metrics(os.path.dirname(os.path.dirname(wtmpl.format(f=f))))
                mp = float(row["metrics/mAP50-95(M)"])
            per_fold.append((v[:, 2].mean(), v[:, 1].mean(), v[:, 0].mean(), mp))
        a = np.array(per_fold)
        rows.append(dict(config=label, head_gain=h, lam=l,
                         dice=a[:, 0].mean(), hd95=a[:, 1].mean(),
                         biou=a[:, 2].mean(), map=a[:, 3].mean(),
                         **{f"dice_f{f}": a[i, 0] for i, f in enumerate(FOLDS)},
                         **{f"hd95_f{f}": a[i, 1] for i, f in enumerate(FOLDS)}))

    out = OUT if args.split == "test" else OUT.replace(".csv", "_val.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'config':<20}{'Dice':>9}{'HD95%':>9}{'BIoU':>9}{'mAP':>9}")
    for r in rows:
        print(f"{r['config']:<20}{r['dice']:>9.4f}{r['hd95']:>9.3f}{r['biou']:>9.4f}{r['map']:>9.4f}")
    print(f"\n저장: {out}  (split={args.split}, folds={FOLDS})")


if __name__ == "__main__":
    main()
