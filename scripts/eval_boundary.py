#!/usr/bin/env python
"""
경계 전용 진단: baseline vs BRM 의 경계 품질을 Boundary IoU / HD95 로 비교.

mAP 는 경계의 미세 개선을 잘 못 잡을 수 있어, held-out tn3k_test 에서
경계 특화 지표로 baseline nano vs BRM nano 를 비교한다.
각 모델의 5-fold best.pt 로 예측 → GT(폴리곤 라스터화)와 경계 지표 계산 → fold 평균±std.

지표(semantic, 단일 클래스 union):
  - Boundary IoU (Cheng 2021, dilation_ratio=0.02): 경계 밴드에서의 IoU (↑ 좋음)
  - HD95: 95퍼센타일 Hausdorff 거리, 이미지 대각선 대비 % (↓ 좋음)
  - Dice: 참고용 전역 겹침 (↑)

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/eval_boundary.py --device 0
"""
import os
import json
import argparse
import statistics as st
import numpy as np
import cv2

ROOT = "/data/HJ_thyroid"
TEST_LIST = f"{ROOT}/datasets/splits/heldout_test/tn3k_test.txt"
DIL_RATIO = 0.02
CONF = 0.25


def _erode(m, k):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.erode(m, ker)


def poly_to_mask(polys_norm, h, w):
    """정규화 폴리곤 리스트 -> (h,w) uint8 union 마스크."""
    m = np.zeros((h, w), np.uint8)
    for p in polys_norm:
        pts = np.array(p, np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    return m


def load_gt(img_path, h, w):
    lbl = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    polys = []
    if os.path.exists(lbl):
        for line in open(lbl):
            v = line.split()
            if len(v) > 6:
                polys.append([float(x) for x in v[1:]])
    return poly_to_mask(polys, h, w)


def pred_mask(result, h, w):
    """ultralytics Result -> (h,w) uint8 union 마스크 (예측 폴리곤 라스터화)."""
    m = np.zeros((h, w), np.uint8)
    if result.masks is None:
        return m
    for xy in result.masks.xy:  # 원본 좌표계 폴리곤
        if len(xy) >= 3:
            cv2.fillPoly(m, [xy.astype(np.int32)], 1)
    return m


def boundary_iou(gt, pred, ratio=DIL_RATIO):
    h, w = gt.shape
    d = max(1, int(round(ratio * np.hypot(h, w))))
    gb = gt - _erode(gt, d)
    pb = pred - _erode(pred, d)
    inter = np.logical_and(gb, pb).sum()
    union = np.logical_or(gb, pb).sum()
    return inter / union if union > 0 else 1.0


def hd95(gt, pred):
    h, w = gt.shape
    diag = np.hypot(h, w)
    gb = gt - _erode(gt, 1)
    pb = pred - _erode(pred, 1)
    if gb.sum() == 0 or pb.sum() == 0:
        return diag, diag / diag * 100  # 최대 패널티(예측/GT 경계 없음)
    dt_to_p = cv2.distanceTransform((1 - pb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_to_g = cv2.distanceTransform((1 - gb).astype(np.uint8), cv2.DIST_L2, 3)
    d_gp = np.percentile(dt_to_p[gb > 0], 95)
    d_pg = np.percentile(dt_to_g[pb > 0], 95)
    hd = max(d_gp, d_pg)
    return hd, hd / diag * 100


def dice(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    s = gt.sum() + pred.sum()
    return 2 * inter / s if s > 0 else 1.0


def eval_fold_per_image(weight, imgs, device, task="segment"):
    """이미지별 경계 지표를 그대로 반환 -> {img_path: (biou, hd95%, dice)}.

    통계 검정(대응표본)에 쓰려면 평균이 아닌 이미지 단위 값이 필요하다.
    """
    from ultralytics import YOLO
    model = YOLO(weight, task=task)  # .onnx는 task 추론이 안 돼 명시 필요
    out = {}
    for ip in imgs:
        img = cv2.imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(ip, h, w)
        r = model.predict(ip, imgsz=640, conf=CONF, iou=0.7, device=device,
                          verbose=False, retina_masks=True)[0]
        pr = pred_mask(r, h, w)
        out[ip] = (boundary_iou(gt, pr), hd95(gt, pr)[1], dice(gt, pr))
    return out


def eval_fold(weight, imgs, device, task="segment"):
    per = eval_fold_per_image(weight, imgs, device, task=task)
    v = np.array(list(per.values()), dtype=float)  # (N,3) = biou, hd, dice
    return v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean()


def read_test_map(base, fold):
    p = f"{ROOT}/{base}_fold{fold}_test/test_metrics.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return float(json.load(f).get("metrics/mAP50-95(M)", 0) or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", nargs="+",
                    default=["baseline=runs/phase2_baseline/n", "BRM=runs/phase3_brm/nbrm"],
                    help='"label=PROJECT/PREFIX" 목록')
    ap.add_argument("--folds", default="0,1,2,3,4", help="쉼표구분 fold 인덱스 (예: 0,2)")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    runs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.runs]
    imgs = [l.strip() for l in open(TEST_LIST) if l.strip()]
    print(f"held-out test images: {len(imgs)}  folds={folds}  (conf={CONF}, dilation={DIL_RATIO})")

    summary = {}
    for label, base in runs:
        per_fold = []
        for i in folds:
            w = f"{ROOT}/{base}_fold{i}/weights/best.pt"
            if not os.path.exists(w):
                print(f"  [skip] {label} fold{i} 가중치 없음")
                continue
            biou, hd, dc = eval_fold(w, imgs, args.device)
            mp = read_test_map(base, i)
            per_fold.append((biou, hd, dc, mp))
            mps = f"{mp:.4f}" if mp is not None else "n/a"
            print(f"  {label} fold{i}: BoundaryIoU={biou:.4f}  HD95={hd:.3f}%  Dice={dc:.4f}  testMask50-95={mps}")
        summary[label] = per_fold

    def ms(vals):
        vals = [v for v in vals if v is not None]
        return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), 0.0)

    labels = [l for l, _ in runs]
    W, C = 20, 20
    print("\n" + "=" * (W + C * len(labels)))
    print(f"  경계 진단 (held-out tn3k_test, folds={folds})")
    print("=" * (W + C * len(labels)))
    print(f"  {'metric':<{W}}" + "".join(f"{l:<{C}}" for l in labels))
    names = [("BoundaryIoU↑", 0), ("HD95%↓", 1), ("Dice↑", 2), ("testMask50-95↑", 3)]
    for nm, idx in names:
        cells = []
        for l in labels:
            m, s = ms([f[idx] for f in summary[l]])
            cells.append(f"{m:.4f}±{s:.4f}")
        print(f"  {nm:<{W}}" + "".join(f"{c:<{C}}" for c in cells))
    print("\n주의: BoundaryIoU·Dice·mAP는 ↑ 좋음, HD95는 ↓ 좋음.")


if __name__ == "__main__":
    main()
