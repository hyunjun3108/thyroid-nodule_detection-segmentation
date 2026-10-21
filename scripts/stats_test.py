#!/usr/bin/env python
"""
Phase 5: BRM 개선의 통계적 유의성 + 교차 데이터셋(LODO) 경계 일반화 검정.

지금까지의 표는 5-fold '평균'만 보고했다. 논문 심사에서는 그 차이가
우연인지(n=5의 변동 안인지)를 묻는다. 여기서는 두 축으로 검정한다.

  Part A (in-domain, held-out tn3k_test)
    - fold 수준(n=5): 대응표본 t / Wilcoxon  ... 보수적, 검정력 낮음
    - 이미지 수준(n=614): fold 5개 예측을 이미지별로 평균낸 뒤 대응표본 검정
      + Cohen's dz + 평균차의 부트스트랩 95% CI  ... 주 분석
  Part B (cross-domain, LODO)
    - tn3k2ddti / ddti2tn3k 두 방향, 시드 42/43/44 반복.
      이미지별로 3시드 예측을 평균낸 뒤 baseline vs BRM 대응표본 검정
      + 시드별 평균 산포를 함께 출력(효과가 실행간 변동보다 큰지 판단).

지표: Dice↑, BoundaryIoU↑, HD95%↓

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/stats_test.py --part all --device 0
  (재실행 시 캐시로 완료분 건너뜀)
"""
import os
import sys
import json
import argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_boundary as EB

ROOT = "/data/HJ_thyroid"
OUTDIR = f"{ROOT}/runs/stats"
TEST_LIST = f"{ROOT}/datasets/splits/heldout_test/tn3k_test.txt"

# 지표 인덱스(eval_fold_per_image 반환 순서) 와 방향(True=클수록 좋음)
METRICS = [("Dice", 2, True), ("BoundaryIoU", 0, True), ("HD95%", 1, False)]

BASE = f"{ROOT}/runs/phase2_baseline"
BRM = f"{ROOT}/runs/phase3_brm_final"

SEEDS = f"{ROOT}/runs/phase5_lodo_seeds"

LODO = [  # (방향, 디렉토리 slug, 평가 이미지 목록 = 타깃 도메인)
    ("TN3K→DDTI", "tn3k2ddti", f"{ROOT}/datasets/splits/lodo/tn3k2ddti/val.txt"),
    ("DDTI→TN3K", "ddti2tn3k", f"{ROOT}/datasets/splits/lodo/ddti2tn3k/val.txt"),
]
LODO_SEEDS = [42, 43, 44]  # 42 = Phase2/3 원본, 43·44 = Phase5 반복


def lodo_run(model, slug, seed):
    """(캐시키, 가중치경로). 시드 42 는 Phase2/3 위치, 나머지는 phase5_lodo_seeds."""
    tag = slug.upper()
    if seed == 42:  # 기존 캐시 키(무접미)를 그대로 재사용 — 재추론 방지
        d = f"{BASE}/n_{slug}" if model == "base" else f"{BRM}/nbrm_{slug}"
        return f"B|{model}|{tag}", f"{d}/weights/best.pt"
    pre = "n" if model == "base" else "nbrm"
    return f"B|{model}|{tag}|s{seed}", f"{SEEDS}/{pre}_{slug}_s{seed}/weights/best.pt"


def load_list(p):
    return [l.strip() for l in open(p) if l.strip()]


class Cache:
    """(key -> {img: [biou,hd,dice]}) 디스크 캐시. 항목 단위로 즉시 저장."""

    def __init__(self, path, enabled=True):
        self.path, self.enabled = path, enabled
        self.d = {}
        if enabled and os.path.exists(path):
            with open(path) as f:
                self.d = json.load(f)
            print(f"캐시 로드: {path} ({len(self.d)}개 실행)")

    def get_or_run(self, key, weight, imgs, device):
        if self.enabled and key in self.d:
            print(f"  [cache] {key}")
            return self.d[key]
        assert os.path.exists(weight), f"가중치 없음: {weight}"
        print(f"  [run]   {key}  ({len(imgs)} imgs)")
        per = EB.eval_fold_per_image(weight, imgs, device)
        self.d[key] = {k: list(map(float, v)) for k, v in per.items()}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.d, f)
        return self.d[key]


def boot_ci(diff, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    m = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def paired(x_base, x_new, higher_better, do_boot=True):
    """대응표본 요약: 평균차/유의확률/효과크기/개선 비율."""
    d = x_new - x_base                       # BRM - baseline
    signed = d if higher_better else -d      # 항상 '양수 = 개선'
    out = {
        "n": len(d),
        "base": float(x_base.mean()),
        "new": float(x_new.mean()),
        "delta": float(d.mean()),
        "win_rate": float((signed > 0).mean()),
    }
    if len(d) > 1 and np.any(d != 0):
        out["t_p"] = float(stats.ttest_rel(x_new, x_base).pvalue)
        out["w_p"] = float(stats.wilcoxon(x_new, x_base).pvalue)
        out["dz"] = float(signed.mean() / signed.std(ddof=1)) if signed.std(ddof=1) > 0 else float("nan")
    else:
        out["t_p"] = out["w_p"] = float("nan")
        out["dz"] = float("nan")
    if do_boot and len(d) > 1:
        lo, hi = boot_ci(d)
        out["ci_lo"], out["ci_hi"] = float(lo), float(hi)
    return out


def sig(p):
    if p != p:
        return "n/a"
    return f"{p:.2e}" + ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else " ns")


def print_table(title, rows, pkey="w_p"):
    """pkey: 표시할 유의확률. 표본이 아주 작으면(n<5) Wilcoxon 은 최소 달성 p 자체가
    커서 무의미하므로 대응표본 t(t_p)를 쓴다."""
    print("\n" + "=" * 104)
    print(f"  {title}")
    print("=" * 104)
    plabel = {"w_p": "Wilcoxon p", "t_p": "paired-t p"}[pkey]
    hdr = ["metric", "n", "baseline", "BRM", "Δ(BRM-base)", "95% CI", plabel, "dz", "개선비율"]
    w = [14, 6, 10, 10, 13, 20, 15, 7, 9]
    print("  " + "".join(f"{h:<{x}}" for h, x in zip(hdr, w)))
    for r in rows:
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]" if "ci_lo" in r else "-"
        cells = [r["metric"], str(r["n"]), f"{r['base']:.4f}", f"{r['new']:.4f}",
                 f"{r['delta']:+.4f}", ci, sig(r[pkey]), f"{r['dz']:.2f}",
                 f"{r['win_rate']*100:.1f}%"]
        print("  " + "".join(f"{c:<{x}}" for c, x in zip(cells, w)))


def part_a(cache, folds, device, csv_rows):
    imgs = load_list(TEST_LIST)
    print(f"\n### Part A: in-domain 유의성 (held-out tn3k_test, {len(imgs)} imgs, folds={folds})")
    runs = {}
    for f in folds:
        runs[("base", f)] = cache.get_or_run(
            f"A|base|f{f}", f"{BASE}/n_fold{f}/weights/best.pt", imgs, device)
        runs[("brm", f)] = cache.get_or_run(
            f"A|brm|f{f}", f"{BRM}/nbrm_fold{f}/weights/best.pt", imgs, device)

    # --- 이미지 수준(주 분석): fold 5개 예측을 이미지별로 평균 ---
    keys = sorted(set.intersection(*[set(runs[k]) for k in runs]))
    img_rows = []
    for name, idx, hb in METRICS:
        b = np.array([[runs[("base", f)][k][idx] for f in folds] for k in keys]).mean(axis=1)
        n = np.array([[runs[("brm", f)][k][idx] for f in folds] for k in keys]).mean(axis=1)
        r = paired(b, n, hb)
        r["metric"] = name
        img_rows.append(r)
        csv_rows.append(dict(analysis="A_image_level", **r))
    print_table("Part A-1  이미지 수준 대응표본 (fold 평균 예측, 주 분석)", img_rows)

    # --- fold 수준(보조): n=5, 보수적 ---
    fold_rows = []
    for name, idx, hb in METRICS:
        b = np.array([np.mean([v[idx] for v in runs[("base", f)].values()]) for f in folds])
        n = np.array([np.mean([v[idx] for v in runs[("brm", f)].values()]) for f in folds])
        r = paired(b, n, hb, do_boot=False)
        r["metric"] = name
        fold_rows.append(r)
        csv_rows.append(dict(analysis="A_fold_level", **r))
    print_table(f"Part A-2  fold 수준 대응표본 (n={len(folds)}, 보수적)", fold_rows)


def print_seed_spread(title, per_seed):
    """시드별 평균을 나란히 — 효과가 실행간 변동보다 큰지 눈으로 확인."""
    print("\n  " + title)
    w = [14, 10, 10, 10, 22]
    print("    " + "".join(f"{h:<{x}}" for h, x in
                           zip(["metric", "model", "mean", "sd(seed)", "seed별 값"], w)))
    for name in per_seed:
        for model in ("base", "brm"):
            v = np.array(per_seed[name][model])
            cells = [name if model == "base" else "", model, f"{v.mean():.4f}",
                     f"{v.std(ddof=1):.4f}", "  ".join(f"{x:.4f}" for x in v)]
            print("    " + "".join(f"{c:<{x}}" for c, x in zip(cells, w)))


def part_b(cache, device, csv_rows, seeds=LODO_SEEDS):
    print(f"\n### Part B: cross-domain 경계 일반화 (LODO, seeds={seeds})")
    for direction, slug, lst in LODO:
        imgs = load_list(lst)
        tag = slug.upper()
        runs = {}
        for model in ("base", "brm"):
            for s in seeds:
                key, w = lodo_run(model, slug, s)
                runs[(model, s)] = cache.get_or_run(key, w, imgs, device)

        keys = sorted(set.intersection(*[set(d) for d in runs.values()]))
        rows, per_seed = [], {}
        for name, idx, hb in METRICS:
            # 이미지별로 시드 평균 -> 대응표본 (Part A 의 fold 평균과 같은 방식)
            xb = np.array([[runs[("base", s)][k][idx] for s in seeds] for k in keys]).mean(axis=1)
            xn = np.array([[runs[("brm", s)][k][idx] for s in seeds] for k in keys]).mean(axis=1)
            r = paired(xb, xn, hb)
            r["metric"] = name
            rows.append(r)
            csv_rows.append(dict(analysis=f"B_{tag}", **r))
            per_seed[name] = {m: [float(np.mean([v[idx] for v in runs[(m, s)].values()]))
                                  for s in seeds] for m in ("base", "brm")}
        print_table(f"Part B  {direction} (타깃 {len(imgs)} imgs, {len(seeds)}시드 평균)", rows)
        print_seed_spread(f"시드별 평균 (seeds={seeds}) — 효과 vs 실행간 변동", per_seed)

        # 시드 수준 검정(n=시드 수). 이미지 수준 p 는 '이 3개 모델'에 대한 진술이고,
        # '이 기법'을 일반화하려면 실행간 변동을 오차로 삼는 이쪽이 옳은 단위다.
        srows = []
        for name, idx, hb in METRICS:
            xb = np.array(per_seed[name]["base"])
            xn = np.array(per_seed[name]["brm"])
            r = paired(xb, xn, hb, do_boot=False)
            r["metric"] = name
            srows.append(r)
            csv_rows.append(dict(analysis=f"B_{tag}_seed_level", **r))
        print_table(f"Part B  {direction} — 시드 수준 대응표본 (n={len(seeds)}, 올바른 일반화 단위)",
                    srows, pkey="t_p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "all"], default="all")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--device", default="0")
    ap.add_argument("--cache", default=f"{OUTDIR}/per_image_cache.json")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    cache = Cache(args.cache, enabled=not args.no_cache)
    folds = [int(x) for x in args.folds.split(",")]
    csv_rows = []

    if args.part in ("A", "all"):
        part_a(cache, folds, args.device, csv_rows)
    if args.part in ("B", "all"):
        part_b(cache, args.device, csv_rows)

    import csv as _csv
    out = f"{OUTDIR}/paired_stats.csv"
    cols = ["analysis", "metric", "n", "base", "new", "delta",
            "ci_lo", "ci_hi", "t_p", "w_p", "dz", "win_rate"]
    with open(out, "w", newline="") as f:
        wtr = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wtr.writeheader()
        wtr.writerows(csv_rows)
    print(f"\n저장: {out}")
    print("주의: p 는 Wilcoxon signed-rank(양측). dz 는 대응표본 Cohen's d,")
    print("      부호는 '양수=BRM 개선'으로 통일(HD95는 감소가 개선).")


if __name__ == "__main__":
    main()
