#!/usr/bin/env python
"""
논문용 구조도/과정도 생성.

Fig5 architecture : 제안 모델(YOLO11n-seg + SegmentBRM) 구조도 —
                    backbone/neck/head, 경계 분기는 학습 전용(점선)
Fig6 dual_path    : 학습 경로 vs 추론 경로 (경계 분기가 추론에서 제거됨 → 추가 FLOPs 0)
Fig7 boundary_loss: 경계 가중 loss 동작 원리 — 실제 GT 마스크로 형태학적 경계밴드·가중맵 시각화
Fig8 pipeline     : 전체 연구 파이프라인(데이터 → 베이스라인 → 경계정교화 → 경량화 → 배포)

출력: runs/figures/fig{5..8}_*.{png,pdf}

사용: /data/my_envs/thyroid_seg/bin/python scripts/make_arch_figures.py [--only 5,6,7,8]
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = "/data/HJ_thyroid"
OUT = f"{ROOT}/runs/figures"

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300"}
INK, MUTED, GRID = "#1a1a19", "#5c5c5a", "#d8d8d5"
FILL = "#f2f4f7"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "text.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  저장: {OUT}/{name}.png (+.pdf)")


def box(ax, x, y, w, h, text, fc=FILL, ec=MUTED, fs=8.5, bold=False, dashed=False,
        tc=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       fc=fc, ec=ec, lw=1.6 if bold else 1.1,
                       ls="--" if dashed else "-", zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or INK, fontweight="bold" if bold else "normal", zorder=3,
            linespacing=1.35)
    return (x + w / 2, y, x + w / 2, y + h)  # cx, ybot, cx, ytop


def arrow(ax, p0, p1, color=MUTED, dashed=False, lw=1.3, text=None, tx=0, ty=0,
          fs=7.6, rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                        color=color, lw=lw, ls="--" if dashed else "-",
                        connectionstyle=f"arc3,rad={rad}", zorder=1,
                        shrinkA=1.5, shrinkB=2.5)
    ax.add_patch(a)
    if text:
        ax.text((p0[0] + p1[0]) / 2 + tx, (p0[1] + p1[1]) / 2 + ty, text,
                ha="center", va="center", fontsize=fs, color=MUTED, zorder=3)


def blank_ax(fig, rect=(0, 0, 1, 1)):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return ax


# ------------------------------------------------------------------ Fig 5
def fig5_architecture():
    fig = plt.figure(figsize=(7.8, 6.8))
    ax = blank_ax(fig)

    # --- 상단 backbone/neck 흐름 (세로 직선) ---
    inp = box(ax, 0.29, 0.905, 0.42, 0.052, "Input ultrasound image   640 × 640 × 3")
    bb = box(ax, 0.23, 0.795, 0.54, 0.070,
             "Backbone   Conv → C3k2 → SPPF → C2PSA\n(YOLO11n, width 0.25)")
    nk = box(ax, 0.23, 0.700, 0.54, 0.058,
             "Neck (PAN-FPN)   upsample / downsample + concat")
    ft = box(ax, 0.14, 0.610, 0.72, 0.055,
             "Multi-scale features   P3 80²  ·  P4 40²  ·  P5 20²",
             fc="#eef4fd", ec=C["blue"])
    arrow(ax, (inp[0], inp[1]), (bb[0], bb[3]))
    arrow(ax, (bb[0], bb[1]), (nk[0], nk[3]))
    arrow(ax, (nk[0], nk[1]), (ft[0], ft[3]))

    # --- head 컨테이너 ---
    hx, hy, hw, hh = 0.055, 0.245, 0.89, 0.320
    ax.add_patch(FancyBboxPatch((hx, hy), hw, hh,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#fbfbfa", ec=C["orange"], lw=1.8, zorder=1))
    ax.text(hx + hw / 2, hy + hh - 0.021, "SegmentBRM head  (proposed)",
            ha="center", fontsize=10, color=C["orange"], fontweight="bold")

    # 3개 분기 (한 줄) — features 에서 세로 직선으로 연결
    dt = box(ax, 0.095, 0.430, 0.245, 0.070, "Detect branch\ncv2 / cv3 → box, cls")
    cf = box(ax, 0.375, 0.430, 0.245, 0.070, "cv4 branch\n32 mask coefficients")
    pr = box(ax, 0.655, 0.430, 0.245, 0.070, "Proto(P3)\n32 prototypes @160²",
             fc="#eaf7f1", ec=C["aqua"])
    for t in (dt, cf, pr):
        arrow(ax, (t[0], ft[1]), (t[0], t[3]))

    # 경계 분기(신규) — Proto 바로 아래, 점선
    bd = box(ax, 0.632, 0.276, 0.303, 0.101,
             "Boundary branch  (new)\nConv3×3 (32→16) → Conv1×1 (16→1)\n"
             "1-ch edge map @160²\n+4.7 K params",
             fc="#fdeee7", ec=C["orange"], dashed=True, bold=True, fs=7.4)
    arrow(ax, (pr[0], pr[1]), (bd[0], bd[3]), color=C["orange"], dashed=True)

    # 마스크 조립 — cv4 아래
    ma = box(ax, 0.345, 0.290, 0.265, 0.070,
             "Mask assembly\nσ( Σ coeff · prototype )", fc="#eef4fd", ec=C["blue"])
    arrow(ax, (cf[0], cf[1]), (ma[0], ma[3]))
    arrow(ax, (pr[0] - 0.10, pr[1]), (ma[0] + 0.11, ma[3]), rad=0.18)

    # --- 출력 (head 밖) ---
    o1 = box(ax, 0.1025, 0.140, 0.23, 0.058, "Nodule boxes\n(detection)",
             fc="#eef4fd", ec=C["blue"], fs=8.2)
    o2 = box(ax, 0.3625, 0.140, 0.23, 0.058, "Nodule masks\n(segmentation)",
             fc="#eef4fd", ec=C["blue"], fs=8.2)
    arrow(ax, (dt[0], hy), (o1[0], o1[3]))
    arrow(ax, (ma[0], ma[1]), (o2[0], o2[3]))

    # --- 학습 전용 loss ---
    lo = box(ax, 0.635, 0.040, 0.32, 0.082,
             "Boundary-aware loss  (training)\nedge BCE  +\nboundary-weighted mask BCE",
             fc="#fdeee7", ec=C["orange"], dashed=True, fs=7.8)
    arrow(ax, (bd[0], bd[1]), (lo[0], lo[3]), dashed=True, color=C["orange"])
    arrow(ax, (o2[0] + 0.08, o2[1]), (lo[0] - 0.13, lo[3] - 0.01), dashed=True,
          color=C["orange"], rad=-0.15)

    ax.text(0.30, 0.012,
            "dashed = training only\n(removed at inference → +0 FLOPs)",
            ha="center", fontsize=8, color=MUTED, style="italic", linespacing=1.4)
    fig.suptitle("Proposed architecture: YOLO11n-seg with Boundary Refinement Module",
                 y=1.005, fontsize=11)
    save(fig, "fig5_architecture")


# ------------------------------------------------------------------ Fig 6
def fig6_dual_path():
    fig = plt.figure(figsize=(7.8, 3.5))

    for i, (title, col, is_train) in enumerate([
            ("Training path", C["orange"], True),
            ("Inference / deployment path", C["blue"], False)]):
        ax = blank_ax(fig, (0.02 + i * 0.5, 0.02, 0.46, 0.90))
        ax.text(0.5, 0.985, title, ha="center", fontsize=10.5, color=col,
                fontweight="bold")

        b1 = box(ax, 0.12, 0.845, 0.76, 0.072, "Backbone + Neck")
        b2 = box(ax, 0.12, 0.700, 0.76, 0.082,
                 "SegmentBRM head\n" + ("returns (det, coeff, proto, edge)"
                                        if is_train else "returns (det, coeff, proto)"),
                 ec=col)
        arrow(ax, (b1[0], b1[1]), (b2[0], b2[3]))

        if is_train:
            e = box(ax, 0.06, 0.545, 0.40, 0.082,
                    "edge map\n(1 ch @160²)", fc="#fdeee7", ec=col, dashed=True)
            m = box(ax, 0.54, 0.545, 0.40, 0.082, "predicted masks")
            arrow(ax, (b2[0] - 0.16, b2[1]), (e[0], e[3]), rad=0.1)
            arrow(ax, (b2[0] + 0.16, b2[1]), (m[0], m[3]), rad=-0.1)

            l1 = box(ax, 0.06, 0.385, 0.40, 0.082,
                     "edge BCE\nvs GT boundary", fc="#fdeee7", ec=col, fs=8.2)
            l2 = box(ax, 0.54, 0.385, 0.40, 0.082,
                     "mask BCE × (1+λ·B)\nboundary-weighted", fc="#fdeee7", ec=col,
                     fs=8.2)
            arrow(ax, (e[0], e[1]), (l1[0], l1[3]))
            arrow(ax, (m[0], m[1]), (l2[0], l2[3]))

            tot = box(ax, 0.14, 0.225, 0.72, 0.078,
                      "total loss = box + cls + dfl + seg\n(+ boundary terms)",
                      ec=col, bold=True)
            arrow(ax, (l1[0], l1[1]), (tot[0] - 0.14, tot[3]), rad=-0.1)
            arrow(ax, (l2[0], l2[1]), (tot[0] + 0.14, tot[3]), rad=0.1)
            bp = box(ax, 0.24, 0.085, 0.52, 0.062, "backpropagation", fc="white",
                     ec=col, dashed=True)
            arrow(ax, (tot[0], tot[1]), (bp[0], bp[3]))
            ax.text(0.5, 0.035, "boundary signal shapes the prototypes",
                    ha="center", fontsize=8, color=col, style="italic")
        else:
            g = box(ax, 0.12, 0.560, 0.76, 0.072,
                    "boundary branch NOT executed", fc="white", ec=MUTED,
                    dashed=True, tc=MUTED)
            arrow(ax, (b2[0], b2[1]), (g[0], g[3]), dashed=True, color=MUTED)
            n = box(ax, 0.12, 0.425, 0.76, 0.072, "NMS + mask assembly")
            arrow(ax, (g[0], g[1]), (n[0], n[3]))
            o = box(ax, 0.12, 0.290, 0.76, 0.072, "boxes + masks", ec=col)
            arrow(ax, (n[0], n[1]), (o[0], o[3]))
            k = box(ax, 0.08, 0.120, 0.84, 0.100,
                    "identical graph to the baseline\n9.73 GFLOPs  ·  +0 inference cost",
                    fc="#eef4fd", ec=col, bold=True, fs=8.6)
            arrow(ax, (o[0], o[1]), (k[0], k[3]))

    fig.suptitle("How the boundary refinement module works: supervision at training, "
                 "zero cost at inference", y=1.0, fontsize=10.5)
    save(fig, "fig6_dual_path")


# ------------------------------------------------------------------ Fig 7
def fig7_boundary_loss(lam=2.0):
    """실제 GT 마스크로 경계밴드/가중맵 시각화 (loss 동작 원리)."""
    import cv2
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_boundary as EB

    imgs = [l.strip() for l in open(EB.TEST_LIST) if l.strip()]
    pick = None
    for ip in imgs:
        im = cv2.imread(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = EB.load_gt(ip, h, w)
        if gt.sum() > 4000:
            pick = (ip, im, gt)
            break
    ip, im, gt = pick
    k = np.ones((9, 9), np.uint8)
    dil = cv2.dilate(gt, k)
    ero = cv2.erode(gt, k)
    band = np.clip(dil.astype(np.int16) - ero.astype(np.int16), 0, 1).astype(np.float32)
    wmap = 1.0 + lam * band

    fig, axes = plt.subplots(1, 4, figsize=(7.8, 2.45))
    panels = [
        (cv2.cvtColor(im, cv2.COLOR_BGR2RGB), "Ultrasound image", None),
        (gt, "GT mask  M", "gray"),
        (band, "boundary band\nB = dilate(M) − erode(M)", "magma"),
        (wmap, f"weight map\nw = 1 + λ·B   (λ = {lam:g})", "viridis"),
    ]
    for ax, (arr, title, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=8.8, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.suptitle("Boundary-aware loss: BCE on mask pixels is up-weighted near the "
                 "ground-truth boundary", y=1.04, fontsize=10.5)
    fig.text(0.5, -0.10,
             r"$\mathcal{L}_{mask}=\frac{1}{N}\sum_{p} w_p \cdot "
             r"\mathrm{BCE}(\hat{m}_p, m_p)$   ·   "
             r"$\mathcal{L}_{edge}=\mathrm{BCE}(\hat{e}, B)$   ·   "
             r"$\mathcal{L}_{seg}=\mathcal{L}_{mask}+\gamma\,\mathcal{L}_{edge}$"
             "   (γ = 1.0)", ha="center", fontsize=9.4, color=INK)
    fig.tight_layout()
    save(fig, "fig7_boundary_loss")


# ------------------------------------------------------------------ Fig 8
def fig8_pipeline():
    fig = plt.figure(figsize=(8.0, 4.6))
    ax = blank_ax(fig)

    stages = [
        ("Phase 1 — Data", "#f2f4f7", MUTED,
         "TN3K 3 493 + DDTI 637\nmasks → YOLO-seg polygons\n4 130 images · 4 457 nodules"),
        ("Phase 2 — Baseline", "#eef4fd", C["blue"],
         "YOLO11n-seg vs YOLO11s-seg\npooled 5-fold + LODO\n→ nano ≈ small, 3.5× lighter"),
        ("Phase 3 — Boundary refinement", "#fdeee7", C["orange"],
         "SegmentBRM + boundary loss\n(train-only branch)\n→ Dice +0.011, HD95 −13 %"),
        ("Phase 4 — Compression", "#eaf7f1", C["aqua"],
         "ONNX FP16 / INT8, OpenVINO INT8\nNNCF post-training quantization\n→ 3.71 MB, CPU 4.1 ms"),
    ]
    y0, hgt, gap = 0.775, 0.142, 0.041
    prev = None
    for i, (title, fc, ec, body) in enumerate(stages):
        y = y0 - i * (hgt + gap)
        ax.add_patch(FancyBboxPatch((0.05, y), 0.90, hgt,
                                    boxstyle="round,pad=0.012,rounding_size=0.02",
                                    fc=fc, ec=ec, lw=1.5, zorder=2))
        ax.text(0.085, y + hgt * 0.70, title, fontsize=9.6, color=ec,
                fontweight="bold", va="center")
        ax.text(0.085, y + hgt * 0.30, body, fontsize=8.4, color=INK, va="center",
                linespacing=1.4)
        if prev is not None:
            arrow(ax, (0.5, prev), (0.5, y + hgt), lw=1.5)
        prev = y

    # 배포 결론
    ax.add_patch(FancyBboxPatch((0.05, 0.030), 0.90, 0.112,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="white", ec=INK, lw=1.8, zorder=2))
    ax.text(0.5, 0.116, "Deployment", ha="center", fontsize=9.6,
            fontweight="bold", color=INK)
    ax.text(0.29, 0.066, "CPU / edge\nOpenVINO INT8 · 3.71 MB · 4.1 ms",
            ha="center", fontsize=8.4, color=C["aqua"], linespacing=1.4)
    ax.text(0.71, 0.066, "GPU\nONNX FP16 · 5.87 MB · 3.1 ms",
            ha="center", fontsize=8.4, color=C["blue"], linespacing=1.4)
    arrow(ax, (0.5, prev), (0.5, 0.142), lw=1.5)

    fig.suptitle("Research pipeline: from mixed public datasets to a deployable "
                 "boundary-refined lightweight model", y=0.965, fontsize=11)
    save(fig, "fig8_pipeline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="5,6,7,8")
    args = ap.parse_args()
    want = set(args.only.split(","))
    print("구조도 생성 중...")
    if "5" in want:
        fig5_architecture()
    if "6" in want:
        fig6_dual_path()
    if "7" in want:
        fig7_boundary_loss()
    if "8" in want:
        fig8_pipeline()
    print(f"\n완료 → {OUT}/")


if __name__ == "__main__":
    main()
