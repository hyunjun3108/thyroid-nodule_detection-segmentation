#!/usr/bin/env python
"""
논문용 그림 자동 생성 (Phase 2~4 결과 기반).

Fig1  boundary_improvement : Phase3 경계 정교화 효과 — baseline vs BRM, 3개 지표 small multiples
                             (fold별 점 + 평균 막대, per-fold paired 연결선)
Fig2  pareto               : 정확도 vs 효율 — CPU 지연 vs Dice, 마커 크기=모델 크기 (핵심 그림)
Fig3  quantization         : 정밀도별 크기/CPU지연/정확도 3패널
Fig4  qualitative          : 실제 초음파 영상 위 GT vs baseline vs BRM 경계 오버레이

출력: runs/figures/fig{1..4}_*.{png,pdf}  (PNG 300dpi + 벡터 PDF)

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/make_figures.py [--only 1,2,3,4]
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/HJ_thyroid"
OUT = f"{ROOT}/runs/figures"
CACHE = f"{ROOT}/runs/phase4_quant/final_table_cache.json"

# --- 검증된 categorical 팔레트 (CVD-safe, validate_palette.js PASS) ---
C = {
    "blue":    "#2a78d6",  # slot1 — baseline
    "orange":  "#eb6834",  # slot2 — BRM (제안 기법)
    "aqua":    "#1baf7a",  # slot3
    "yellow":  "#eda100",  # slot4
    "magenta": "#e87ba4",  # slot5
    "green":   "#008300",  # slot6
}
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


def load():
    with open(CACHE) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        p = f"{OUT}/{name}.{ext}"
        fig.savefig(p)
    plt.close(fig)
    print(f"  저장: {OUT}/{name}.png (+.pdf)")


# ---------------------------------------------------------------- Fig 1
def fig1_boundary(d):
    """Phase3 경계 개선: baseline vs BRM, fold별 paired 비교."""
    base, brm = d["baseline nano (P2)"], d["BRM nano (P3)"]
    metrics = [("dice", "Dice ↑", False), ("hd", "HD95 (%) ↓", True),
               ("biou", "Boundary IoU ↑", False)]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, (key, label, lower_better) in zip(axes, metrics):
        b = [r[key] for r in base]
        m = [r[key] for r in brm]
        # fold별 paired 연결선(개별 근거) + 평균 막대
        for i, (bv, mv) in enumerate(zip(b, m)):
            ax.plot([0, 1], [bv, mv], "-", color=GRID, lw=0.9, zorder=1)
            ax.plot(0, bv, "o", ms=4.5, color=C["blue"], mec="white", mew=0.8, zorder=3)
            ax.plot(1, mv, "o", ms=4.5, color=C["orange"], mec="white", mew=0.8, zorder=3)
        for x, vals, col in [(0, b, C["blue"]), (1, m, C["orange"])]:
            ax.hlines(np.mean(vals), x - 0.22, x + 0.22, color=col, lw=2.4, zorder=4)
        # 평균값 직접 라벨(대비 WARN 해소용 secondary encoding)
        dec = 3 if key != "hd" else 2
        ax.text(0, min(b + m) - (max(b + m) - min(b + m)) * 0.28,
                f"{np.mean(b):.{dec}f}", ha="center", color=INK, fontsize=8.5)
        ax.text(1, min(b + m) - (max(b + m) - min(b + m)) * 0.28,
                f"{np.mean(m):.{dec}f}", ha="center", color=INK, fontsize=8.5,
                fontweight="bold")
        ax.set_xlim(-0.45, 1.45)
        rng = max(b + m) - min(b + m)
        ax.set_ylim(min(b + m) - rng * 0.42, max(b + m) + rng * 0.18)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "BRM"])
        ax.set_title(label, color=INK)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    fig.suptitle("Boundary refinement module improves boundary quality "
                 "(held-out test, 5-fold)", y=1.04, fontsize=10.5)
    axes[0].plot([], [], "o", color=C["blue"], label="Baseline nano")
    axes[0].plot([], [], "o", color=C["orange"], label="BRM nano (ours)")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
    save(fig, "fig1_boundary_improvement")


# ---------------------------------------------------------------- Fig 2
def fig2_pareto(d):
    """정확도 vs CPU 지연 (마커 크기 = 모델 크기). 핵심 trade-off 그림."""
    rows = [
        ("ONNX FP32",      "BRM + ONNX-FP32",     C["blue"]),
        ("ONNX INT8",      "BRM + ONNX-INT8",     C["yellow"]),
        ("OpenVINO INT8",  "BRM + OpenVINO-INT8", C["orange"]),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for label, key, col in rows:
        rs = d[key]
        cpu = np.mean([r["cpu"] for r in rs if r.get("cpu")])
        dice = np.mean([r["dice"] for r in rs])
        size = np.mean([r["size"] for r in rs])
        ax.scatter(cpu, dice, s=size * 26, color=col, alpha=0.85,
                   edgecolor="white", linewidth=1.6, zorder=3)
        ax.annotate(f"{label}\n{size:.1f} MB · {cpu:.1f} ms",
                    (cpu, dice), textcoords="offset points",
                    xytext=(0, -32 if label == "ONNX INT8" else 20),
                    ha="center", fontsize=8.4, color=INK,
                    fontweight="bold" if "OpenVINO" in label else "normal")
    ax.set_xscale("log")
    ax.set_xticks([4, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("CPU latency per image (ms, batch = 1, log scale)  ↓ better")
    ax.set_ylabel("Dice ↑ better")
    ax.set_title("Accuracy vs CPU inference cost\n(marker area ∝ model size)",
                 color=INK)
    ax.grid(True, which="major"); ax.set_axisbelow(True)
    ax.margins(x=0.22, y=0.30)
    save(fig, "fig2_accuracy_vs_latency")


# ---------------------------------------------------------------- Fig 3
def fig3_quant(d):
    """정밀도별 크기 / CPU지연 / 정확도 — 3패널 (축이 다르므로 dual-axis 금지)."""
    labels = ["FP32", "FP16", "INT8\n(ORT)", "INT8\n(OV)"]
    keys = ["BRM + ONNX-FP32", "BRM + ONNX-FP16", "BRM + ONNX-INT8",
            "BRM + OpenVINO-INT8"]
    cols = [C["blue"], C["aqua"], C["yellow"], C["orange"]]

    def agg(key, field):
        vals = [r[field] for r in d[key] if r.get(field) is not None]
        return (np.mean(vals), np.std(vals)) if vals else (np.nan, 0)

    panels = [("size", "Model size (MB) ↓", "{:.2f}"),
              ("cpu", "CPU latency (ms) ↓", "{:.1f}"),
              ("dice", "Dice ↑", "{:.3f}")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), constrained_layout=True)
    for ax, (field, title, fmt) in zip(axes, panels):
        means = [agg(k, field)[0] for k in keys]
        errs = [agg(k, field)[1] for k in keys]
        x = np.arange(len(keys))
        bars = ax.bar(x, means, width=0.62, color=cols, edgecolor="white",
                      linewidth=1.4, zorder=3)
        if field == "dice":
            ax.errorbar(x, means, yerr=errs, fmt="none", ecolor=MUTED,
                        elinewidth=0.9, capsize=2.5, zorder=4)
            ax.set_ylim(0.80, 0.845)
        for xi, (m, b) in enumerate(zip(means, bars)):
            if np.isnan(m):
                ax.text(xi, ax.get_ylim()[0], "n/a", ha="center", va="bottom",
                        fontsize=8, color=MUTED)
            else:
                ax.text(xi, b.get_height(), fmt.format(m), ha="center",
                        va="bottom", fontsize=8.2, color=INK,
                        fontweight="bold" if xi == 3 else "normal")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, color=INK)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.margins(y=0.20)
    fig.suptitle("Quantization: OpenVINO INT8 is smallest and fastest "
                 "at near-FP32 accuracy   (ORT = onnxruntime, OV = OpenVINO)",
                 fontsize=10.5)
    save(fig, "fig3_quantization")


# ---------------------------------------------------------------- Fig 4
def fig4_qualitative(n=4, seed=7):
    """실제 영상 위 GT / baseline / BRM 경계 오버레이 (정성 비교)."""
    import cv2
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_boundary as EB
    from ultralytics import YOLO

    imgs = [l.strip() for l in open(EB.TEST_LIST) if l.strip()]
    rng = np.random.RandomState(seed)

    mb = YOLO(f"{ROOT}/runs/phase2_baseline/n_fold0/weights/best.pt", task="segment")
    mr = YOLO(f"{ROOT}/runs/phase3_brm_final/nbrm_fold0/weights/best.pt", task="segment")

    # HD95 개선폭이 큰 사례 우선 선택(대표성 + 효과 가시성)
    cand = list(rng.choice(imgs, size=min(40, len(imgs)), replace=False))
    scored = []
    for ip in cand:
        im = cv2.imread(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = EB.load_gt(ip, h, w)
        if gt.sum() < 200:
            continue
        pb = EB.pred_mask(mb.predict(ip, imgsz=640, conf=0.25, verbose=False,
                                     retina_masks=True, device="0")[0], h, w)
        pr = EB.pred_mask(mr.predict(ip, imgsz=640, conf=0.25, verbose=False,
                                     retina_masks=True, device="0")[0], h, w)
        if pb.sum() == 0 or pr.sum() == 0:
            continue
        scored.append((EB.dice(gt, pr) - EB.dice(gt, pb), ip, im, gt, pb, pr))
    # 체리피킹 방지: Dice 개선폭 분포 전체를 대표하도록 분위수에서 선택
    scored.sort(key=lambda t: -t[0])
    m = len(scored)
    idx = sorted({0, m // 4, m // 2, m - 1})[:n] if m >= n else list(range(m))
    row_tags = ["best case", "upper quartile", "median case", "worst case"][:len(idx)]
    picks = [scored[i] for i in idx]

    fig, axes = plt.subplots(len(picks), 3, figsize=(6.9, 2.25 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    col_titles = ["Ground truth", "Baseline nano", "BRM nano (ours)"]
    for r, (dgain, ip, im, gt, pb, pr) in enumerate(picks):
        axes[r, 0].set_ylabel(row_tags[r], fontsize=8.5, color=MUTED, labelpad=6)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        for c, (mask, col, tag) in enumerate([
                (gt, (0, 200, 0), None),
                (pb, tuple(int(C["blue"][i:i + 2], 16) for i in (5, 3, 1)), EB.dice(gt, pb)),
                (pr, tuple(int(C["orange"][i:i + 2], 16) for i in (5, 3, 1)), EB.dice(gt, pr))]):
            ax = axes[r, c]
            ax.imshow(rgb)
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for cn in cnts:
                pts = cn.reshape(-1, 2)
                ax.plot(np.append(pts[:, 0], pts[0, 0]),
                        np.append(pts[:, 1], pts[0, 1]),
                        color=np.array(col[::-1]) / 255, lw=1.6)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9.5, color=INK)
            if tag is not None:
                ax.text(0.03, 0.03, f"Dice {tag:.3f}", transform=ax.transAxes,
                        fontsize=8, color="white", va="bottom",
                        bbox=dict(fc="black", alpha=0.55, pad=1.6, lw=0))
    fig.suptitle("Qualitative boundary comparison on held-out test images\n"
                 f"(rows span the Dice-improvement distribution over {len(scored)} "
                 "sampled cases, best → worst)", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_qualitative")


# ---------------------------------------------------------------- Fig 10
def fig10_dataset(n=5, seed=3):
    """TN3K vs DDTI 표본 + GT 경계 — 두 데이터셋의 도메인 격차를 보이는 그림."""
    import cv2
    import glob
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_boundary as EB

    rng = np.random.RandomState(seed)
    sets = [("TN3K", f"{ROOT}/datasets/tn3k_trainval/images", 2879),
            ("DDTI", f"{ROOT}/datasets/ddti/images", 637)]

    fig, axes = plt.subplots(2, n, figsize=(2.05 * n, 4.5))
    for r, (name, d, cnt) in enumerate(sets):
        files = sorted(glob.glob(f"{d}/*"))
        picks, tries = [], 0
        while len(picks) < n and tries < 200:  # 결절이 실제로 있는 표본만
            ip = files[rng.randint(len(files))]
            tries += 1
            im = cv2.imread(ip)
            if im is None:
                continue
            gt = EB.load_gt(ip, *im.shape[:2])
            if gt.sum() > 300 and ip not in [p[0] for p in picks]:
                picks.append((ip, im, gt))
        for c, (ip, im, gt) in enumerate(picks):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cn in cnts:
                p = cn.reshape(-1, 2)
                ax.plot(np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1]),
                        color="#00c853", lw=1.5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(f"{name}\n({cnt:,} images)", fontsize=9.5, color=INK,
                              labelpad=8)
    fig.suptitle("Dataset samples with ground-truth nodule boundaries (green)\n"
                 "TN3K and DDTI differ in acquisition device, field of view, "
                 "annotation style and image quality", y=1.005, fontsize=10)
    fig.tight_layout()
    save(fig, "fig10_dataset")


# ------------------------------------------------------- Fig 11·12 (표본 스트립)
def _samples(imgdir, n, seed, min_area=300):
    """결절이 실제로 있는 표본 n장을 무작위로 고른다 -> [(경로, 영상, GT)]."""
    import cv2
    import glob
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_boundary as EB

    rng = np.random.RandomState(seed)
    files = sorted(glob.glob(f"{imgdir}/*"))
    out, tries = [], 0
    while len(out) < n and tries < 400:
        ip = files[rng.randint(len(files))]
        tries += 1
        im = cv2.imread(ip)
        if im is None or ip in [p[0] for p in out]:
            continue
        gt = EB.load_gt(ip, *im.shape[:2])
        if gt.sum() > min_area:
            out.append((ip, im, gt))
    return out


def _strip(name, imgdir, outname, n=4, seed=11, note=""):
    """데이터셋 표본 가로 스트립 + GT 경계.

    note 를 비우면 제목을 넣지 않는다 — 슬라이드에서 데이터셋 이름을 따로 쓰는 경우
    그림 안 제목이 중복되고 자리만 잡아먹는다.
    """
    import cv2
    picks = _samples(imgdir, n, seed)
    fig, axes = plt.subplots(1, n, figsize=(2.15 * n, 2.1 if note else 1.85))
    for ax, (ip, im, gt) in zip(np.atleast_1d(axes), picks):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cn in cnts:
            p = cn.reshape(-1, 2)
            ax.plot(np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1]),
                    color="#00c853", lw=1.6)
        h, w = im.shape[:2]
        ax.set_xlabel(f"{w}×{h}", fontsize=7.5, color=MUTED, labelpad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if note:
        fig.suptitle(f"{name} — {note}", y=1.02, fontsize=10.5)
    fig.tight_layout()
    save(fig, outname)


def fig11_tn3k():
    _strip("TN3K", f"{ROOT}/datasets/tn3k_trainval/images", "fig11_tn3k_samples",
           seed=11)


def fig12_ddti():
    _strip("DDTI", f"{ROOT}/datasets/ddti/images", "fig12_ddti_samples",
           seed=5)


# ---------------------------------------------------------------- Fig 13
def fig13_dataset_stats():
    """두 데이터셋의 정량적 차이 3패널 — 이 격차가 교차 데이터셋 실험의 전제다."""
    import glob
    from PIL import Image as PILImage

    def collect(imgdir):
        sizes, areas, counts = [], [], []
        for f in sorted(glob.glob(f"{imgdir}/*")):
            try:
                w, h = PILImage.open(f).size
            except Exception:
                continue
            sizes.append((w, h))
            lbl = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            k = 0
            if os.path.exists(lbl):
                for line in open(lbl):
                    v = line.split()
                    if len(v) > 6:
                        k += 1
                        p = np.array([float(x) for x in v[1:]]).reshape(-1, 2)
                        x, y = p[:, 0], p[:, 1]
                        areas.append(abs(np.dot(x, np.roll(y, -1)) -
                                         np.dot(y, np.roll(x, -1))) / 2 * 100)
            counts.append(k)
        return np.array(sizes), np.array(areas), np.array(counts)

    data = {"TN3K": collect(f"{ROOT}/datasets/tn3k_trainval/images"),
            "DDTI": collect(f"{ROOT}/datasets/ddti/images")}
    col = {"TN3K": C["blue"], "DDTI": C["orange"]}

    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.5), constrained_layout=True)

    # (a) 이미지 해상도 — DDTI 는 고유 해상도가 5종뿐이라 점이 겹친다.
    #     고유 (w,h)별 장수에 마커 넓이를 비례시켜 두 분포를 함께 보이게 한다.
    import collections
    nuniq = {}
    for k in ("TN3K", "DDTI"):
        s = data[k][0]
        cnt = collections.Counter(map(tuple, s))
        nuniq[k] = len(cnt)
        pts = np.array(list(cnt.keys()))
        num = np.array(list(cnt.values()))
        ax[0].scatter(pts[:, 0], pts[:, 1], s=8 + 3.2 * np.sqrt(num), alpha=.45,
                      color=col[k], label=k, edgecolors="none",
                      zorder=4 if k == "DDTI" else 3)
    ax[0].set_xlabel("image width (px)")
    ax[0].set_ylabel("image height (px)")
    ax[0].set_title("Image resolution  (marker area ∝ #images)", loc="left")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].grid(zorder=0)
    ax[0].text(.97, .04, f"DDTI: {nuniq['DDTI']} fixed sizes, height always 360\n"
                         f"TN3K: {nuniq['TN3K']:,} distinct sizes",
               transform=ax[0].transAxes, ha="right", va="bottom", fontsize=7.6,
               color=MUTED)

    # (b) 결절 상대 면적 — 표본 수가 5배 차이나므로 각 데이터셋 내 비율로 정규화
    bins = np.logspace(np.log10(0.04), np.log10(60), 34)
    for k, (_, a, _) in data.items():
        ax[1].hist(a, bins=bins, weights=np.full(len(a), 100 / len(a)),
                   alpha=.55, color=col[k], label=k, zorder=3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("nodule area (% of image, log scale)")
    ax[1].set_ylabel("% of that dataset's nodules")
    ax[1].set_title("Nodule size distribution", loc="left")
    ax[1].legend(frameon=False)
    ax[1].grid(axis="y", zorder=0)

    # (c) 이미지당 결절 수
    width = 0.36
    xs = np.array([1, 2, 3])
    for i, (k, (_, _, c_)) in enumerate(data.items()):
        frac = [(c_ == 1).mean(), (c_ == 2).mean(), (c_ >= 3).mean()]
        ax[2].bar(xs + (i - .5) * width, np.array(frac) * 100, width * .9,
                  color=col[k], alpha=.9, label=k, zorder=3)
        for x, v in zip(xs + (i - .5) * width, frac):
            if v > 0:
                ax[2].text(x, v * 100 + 1.5, f"{v*100:.1f}", ha="center", fontsize=7.5,
                           color=MUTED)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(["1 nodule", "2 nodules", "3+"])
    ax[2].set_ylabel("% of images")
    ax[2].set_ylim(0, 112)
    ax[2].set_title("Nodules per image", loc="left")
    ax[2].legend(frameon=False)
    ax[2].grid(axis="y", zorder=0)

    fig.suptitle("Quantitative gap between the two public datasets "
                 "(TN3K trainval vs DDTI)", fontsize=10.5, y=1.04, x=.005, ha="left")
    save(fig, "fig13_dataset_stats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2,3,4")
    args = ap.parse_args()
    want = set(args.only.split(","))
    d = load()
    print("그림 생성 중...")
    if "1" in want:
        fig1_boundary(d)
    if "2" in want:
        fig2_pareto(d)
    if "3" in want:
        fig3_quant(d)
    if "4" in want:
        fig4_qualitative()
    if "10" in want:
        fig10_dataset()
    if "11" in want:
        fig11_tn3k()
    if "12" in want:
        fig12_ddti()
    if "13" in want:
        fig13_dataset_stats()
    print(f"\n완료 → {OUT}/")


if __name__ == "__main__":
    main()
