#!/usr/bin/env python
"""
Fig9 통계 검정 그림 — in-domain 개선의 유의성과 cross-domain 차이의 해석, 4패널.

(a) in-domain ΔDice 이미지별 분포(n=614) — 0 기준 개선/악화, 평균 ±95% CI
(b) in-domain ΔHD95% 이미지별 분포 — 음수가 개선
(c) 세 설정(in-domain / TN3K→DDTI / DDTI→TN3K) × 세 지표의 효과크기(dz)
(d) LODO 시드별(42/43/44) Dice — cross-domain 차이가 실행간 변동 안에 있음을 보임

입력: runs/stats/per_image_cache.json, runs/stats/paired_stats.csv (scripts/stats_test.py 산출)
출력: runs/figures/fig9_significance.{png,pdf}      (1×4, 발표용)
      runs/figures/fig9_significance_2x2.{png,pdf}  (2×2, 학위논문용)

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/make_stats_figure.py
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/HJ_thyroid"
OUT = f"{ROOT}/runs/figures"
CACHE = f"{ROOT}/runs/stats/per_image_cache.json"
STATS = f"{ROOT}/runs/stats/paired_stats.csv"

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
INK, MUTED, GRID = "#1a1a19", "#5c5c5a", "#d8d8d5"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

IDX = {"BoundaryIoU": 0, "HD95%": 1, "Dice": 2}
FOLDS = [0, 1, 2, 3, 4]


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def load_stats():
    with open(STATS) as f:
        return list(csv.DictReader(f))


def indomain_diff(cache, metric):
    """이미지별 (BRM - baseline) — fold 5개 예측을 이미지별 평균낸 값의 차."""
    i = IDX[metric]
    b = [cache[f"A|base|f{f}"] for f in FOLDS]
    n = [cache[f"A|brm|f{f}"] for f in FOLDS]
    keys = sorted(set.intersection(*[set(d) for d in b + n]))
    xb = np.array([[d[k][i] for d in b] for k in keys]).mean(axis=1)
    xn = np.array([[d[k][i] for d in n] for k in keys]).mean(axis=1)
    return xn - xb


def hist_panel(ax, diff, row, metric, unit, higher_better):
    """개선/악화를 색으로 구분한 대응차 히스토그램."""
    lim = np.percentile(np.abs(diff), 99)
    bins = np.linspace(-lim, lim, 41)
    counts, edges = np.histogram(np.clip(diff, -lim, lim), bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    good = centers > 0 if higher_better else centers < 0
    ax.bar(centers[good], counts[good], width=np.diff(edges)[0] * 0.92,
           color=C["orange"], alpha=.85, label="BRM better", zorder=3)
    ax.bar(centers[~good], counts[~good], width=np.diff(edges)[0] * 0.92,
           color=C["blue"], alpha=.55, label="baseline better", zorder=3)
    ax.axvline(0, color=MUTED, lw=1.0, zorder=4)

    m = float(row["delta"])
    lo, hi = float(row["ci_lo"]), float(row["ci_hi"])
    ax.axvspan(lo, hi, color=INK, alpha=.10, zorder=2)
    ax.axvline(m, color=INK, lw=1.6, ls="--", zorder=5)

    ax.set_xlabel(f"Δ{metric}  (BRM − baseline{unit})")
    ax.set_ylabel("# test images")
    ax.grid(axis="y", zorder=0)
    ax.set_xlim(-lim, lim)
    dirn = "higher is better" if higher_better else "lower is better"
    ax.set_title(f"{metric}  ({dirn})", loc="left")
    txt = (f"mean Δ = {m:+.4f}\n95% CI [{lo:+.4f}, {hi:+.4f}]\n"
           f"Wilcoxon p = {float(row['w_p']):.1e} {stars(float(row['w_p']))}\n"
           f"improved on {float(row['win_rate'])*100:.1f}% of images  (n={row['n']})")
    ax.text(.98, .97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
            zorder=6, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID, lw=.8))
    ax.legend(loc="lower left", frameon=False)   # 위쪽은 통계 박스 자리


def effect_panel(ax, rows):
    """설정×지표 대응표본 효과크기(dz). 지표별 스케일이 달라 % 대신 dz 로 통일.

    상대 % 로 그리면 cross-domain HD95(-91%)가 축을 지배해 in-domain(+1.3%)이
    보이지 않는다. dz 는 세 지표를 같은 축에 올릴 수 있고, 부호도 '양수=BRM 우세'.
    """
    settings = [("A_image_level", "in-domain\n(TN3K test)"),
                ("B_TN3K2DDTI", "cross-domain\nTN3K→DDTI"),
                ("B_DDTI2TN3K", "cross-domain\nDDTI→TN3K")]
    metrics = ["Dice", "BoundaryIoU", "HD95%"]
    colors = [C["orange"], C["aqua"], C["yellow"]]
    idx = {(r["analysis"], r["metric"]): r for r in rows}

    x = np.arange(len(settings))
    width = 0.26
    allv = []
    for j, met in enumerate(metrics):
        vals, labs = [], []
        for key, _ in settings:
            r = idx.get((key, met))
            vals.append(float(r["dz"]) if r else 0.0)
            labs.append(stars(float(r["w_p"])) if r else "")
        allv += vals
        pos = x + (j - 1) * width
        ax.bar(pos, vals, width * 0.9, color=colors[j], alpha=.9, label=met, zorder=3)
        for p, v, s in zip(pos, vals, labs):
            ax.text(p, v + (.025 if v >= 0 else -.06), s, ha="center",
                    fontsize=7.6, color=MUTED)

    hi, lo = max(allv), min(allv)
    ax.set_ylim(lo - .18, hi + .30)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([s for _, s in settings])
    ax.set_ylabel("paired effect size (Cohen's $d_z$)")
    ax.set_title("Effect by domain  (positive = BRM better)", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.text(.98, .97, "in-domain: significant gain\ncross-domain: small, within seed variance\n"
            "(stars = image-level test; see panel (d))",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID, lw=.8))


LODO_SEEDS = [42, 43, 44]
LODO_DIRS = [("TN3K2DDTI", "TN3K→DDTI"), ("DDTI2TN3K", "DDTI→TN3K")]


def seed_mean(cache, model, tag, seed, metric):
    key = f"B|{model}|{tag}" + ("" if seed == 42 else f"|s{seed}")
    return float(np.mean([v[IDX[metric]] for v in cache[key].values()]))


def seed_panel(ax, cache, rows, metric="Dice"):
    """LODO 시드별 평균을 그대로 찍어 '효과 vs 실행간 변동'을 보인다.

    이미지 수준 p 는 작게 나오지만 그건 '이 모델들'에 대한 진술이다. 기법을
    일반화하려면 시드가 오차 단위이고, 그 단위에서는 두 그룹이 겹친다.
    """
    idx = {(r["analysis"], r["metric"]): r for r in rows}
    xs, ticks, labels = [], [], []
    for gi, (tag, nice) in enumerate(LODO_DIRS):
        x0 = gi * 2.7
        for mi, (model, col, nm) in enumerate([("base", C["blue"], "baseline"),
                                               ("brm", C["orange"], "BRM")]):
            x = x0 + mi
            vals = [seed_mean(cache, model, tag, s, metric) for s in LODO_SEEDS]
            ax.scatter([x] * len(vals), vals, s=34, color=col, zorder=4,
                       label=nm if gi == 0 else None)
            ax.plot([x - .22, x + .22], [np.mean(vals)] * 2, color=col, lw=2.2, zorder=5)
            xs.append((x, vals))
            ticks.append(x)
            labels.append(nm)
        # 같은 시드끼리 연결 — 대응 구조를 보이게
        vb = [seed_mean(cache, "base", tag, s, metric) for s in LODO_SEEDS]
        vn = [seed_mean(cache, "brm", tag, s, metric) for s in LODO_SEEDS]
        for a, b in zip(vb, vn):
            ax.plot([x0, x0 + 1], [a, b], color=MUTED, lw=.7, alpha=.55, zorder=3)
        r = idx.get((f"B_{tag}_seed_level", metric))
        if r:  # y 는 축 비율(데이터 범위가 뒤에 바뀌어도 위치 고정)
            ax.text(x0 + .5, .96, f"paired-t p = {float(r['t_p']):.2f} ns",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.6, color=MUTED)
        ax.text(x0 + .5, -0.13, nice, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5)

    allv = [v for _, vals in xs for v in vals]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * .22          # 위: p 주석, 아래: 범례 자리
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", pad=1)
    ax.set_xlim(-.7, 4.4)
    ax.set_ylabel(f"{metric} on target domain")
    ax.set_title(f"Cross-domain {metric}, per seed (42/43/44)", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, loc="lower right")


def build(cache, rows, layout):
    """layout='1x4': 발표·가로 지면용, '2x2': 세로 쪽(학위논문)용 — 패널은 같고 배치만 다르다."""
    ind = {r["metric"]: r for r in rows if r["analysis"] == "A_image_level"}
    if layout == "1x4":
        fig = plt.figure(figsize=(16.8, 4.4), constrained_layout=True)
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.1, .9])
        axes = [fig.add_subplot(gs[i]) for i in range(4)]
    else:
        fig = plt.figure(figsize=(11.0, 8.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
        axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    hist_panel(axes[0], indomain_diff(cache, "Dice"), ind["Dice"], "Dice", "", True)
    hist_panel(axes[1], indomain_diff(cache, "HD95%"), ind["HD95%"], "HD95%", ", % of diagonal", False)
    effect_panel(axes[2], rows)
    seed_panel(axes[3], cache, rows)
    # 본문 캡션이 (a)~(d)로 패널을 가리키므로 글자를 붙인다
    for ax, t in zip(axes, "abcd"):
        ax.text(-0.02, 1.07, f"({t})", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold")
    if layout == "1x4":
        fig.suptitle("Fig 9  Statistical significance of boundary refinement and cross-dataset generalization",
                     fontsize=11.5, y=1.04, x=.01, ha="left")
    return fig


def main():
    for p in (CACHE, STATS):
        assert os.path.exists(p), f"없음: {p} — 먼저 scripts/stats_test.py 를 실행하세요"
    with open(CACHE) as f:
        cache = json.load(f)
    rows = load_stats()
    os.makedirs(OUT, exist_ok=True)
    for layout, name in [("1x4", "fig9_significance"), ("2x2", "fig9_significance_2x2")]:
        fig = build(cache, rows, layout)
        for ext in ("png", "pdf"):
            p = f"{OUT}/{name}.{ext}"
            fig.savefig(p)
            print("saved", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
