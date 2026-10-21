#!/usr/bin/env python
"""
Phase 2 결과 집계: pooled 5-fold(val+held-out test) + LODO 지표를 표로 정리.

- 각 학습 run 의 results.csv 에서 best epoch(=mAP50-95(M) 최대) 행을 val 지표로 사용
- pooled fold: <name>_test/test_metrics.json 에서 held-out tn3k_test 지표 사용
- 모델(n/s)별 5-fold 평균±표준편차 + LODO 양방향 출력

사용: python collect_results.py
"""
import os
import glob
import json
import csv
import statistics as st

PROJ = "/data/HJ_thyroid/runs/phase2_baseline"

# 표시할 지표 (results.csv 컬럼명 -> 짧은 라벨)
VAL_KEYS = [
    ("metrics/mAP50(B)", "Box50"),
    ("metrics/mAP50-95(B)", "Box50-95"),
    ("metrics/mAP50(M)", "Mask50"),
    ("metrics/mAP50-95(M)", "Mask50-95"),
]


def read_best_row(name):
    """results.csv 에서 mAP50-95(M) 최대 행을 dict 로 반환."""
    csv_path = os.path.join(PROJ, name, "results.csv")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()}
                for r in csv.DictReader(f)]
    if not rows:
        return None
    key = "metrics/mAP50-95(M)"
    return max(rows, key=lambda r: float(r.get(key, 0) or 0))


def read_test(name):
    p = os.path.join(PROJ, name + "_test", "test_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def fmt_row(label, d):
    if d is None:
        return f"  {label:<18} (결과 없음)"
    vals = " ".join(f"{lbl}={float(d.get(k,0) or 0):.3f}" for k, lbl in VAL_KEYS)
    return f"  {label:<18} {vals}"


def summarize(model_prefix):
    print(f"\n===== {model_prefix.upper()} (yolo11{model_prefix}-seg) =====")
    # pooled 5-fold val
    print("[pooled 5-fold — VAL]")
    val_rows = []
    for i in range(5):
        r = read_best_row(f"{model_prefix}_fold{i}")
        val_rows.append(r)
        print(fmt_row(f"fold{i}", r))
    print_meanstd("5-fold VAL 평균", val_rows)

    # pooled 5-fold held-out test
    print("[pooled 5-fold — held-out tn3k_test]")
    test_rows = []
    for i in range(5):
        t = read_test(f"{model_prefix}_fold{i}")
        test_rows.append(t)
        print(fmt_row(f"fold{i}", t))
    print_meanstd("5-fold TEST 평균", test_rows)

    # LODO
    print("[LODO — cross-dataset generalization]")
    for name, label in [("tn3k2ddti", "TN3K→DDTI"), ("ddti2tn3k", "DDTI→TN3K")]:
        r = read_best_row(f"{model_prefix}_{name}")
        print(fmt_row(label, r))


def print_meanstd(label, rows):
    good = [r for r in rows if r is not None]
    if not good:
        print(f"  {label:<18} (결과 없음)")
        return
    parts = []
    for k, lbl in VAL_KEYS:
        vals = [float(r.get(k, 0) or 0) for r in good]
        m = st.mean(vals)
        s = st.pstdev(vals) if len(vals) > 1 else 0.0
        parts.append(f"{lbl}={m:.3f}±{s:.3f}")
    print(f"  {label:<18} {' '.join(parts)}  (n={len(good)})")


def main():
    if not os.path.isdir(PROJ):
        print(f"결과 폴더 없음: {PROJ}")
        return
    for mp in ("n", "s"):
        summarize(mp)
    print("\n참고: VAL=best epoch(mAP50-95(M) 최대) 기준, TEST=held-out tn3k_test, "
          "LODO=남긴 도메인 전체 평가.")


if __name__ == "__main__":
    main()
