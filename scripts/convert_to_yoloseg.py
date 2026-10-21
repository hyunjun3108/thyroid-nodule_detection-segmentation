#!/usr/bin/env python
"""
갑상선 결절 seg 데이터셋(TN3K, DDTI) -> YOLO-seg polygon 라벨 통합 변환기.

- 마스크(JPG/PNG, 압축잡음 포함)를 임계값 이진화(>127) + 모폴로지 open으로 정제
- 외곽 컨투어를 정규화 polygon 으로 변환, 단일 클래스 'nodule'(class 0)
- 이미지에는 symlink 생성(원본 datasets_raw 유지), 라벨은 txt 생성
- 소스 이름을 파일명에 prefix -> 나중에 pool 합칠 때 충돌 방지
- 변환 요약(이미지 수 / 라벨 있는 수 / 빈 마스크 / 총 폴리곤) 출력

사용: python convert_to_yoloseg.py
"""
import os
import glob
import cv2
import numpy as np
from pathlib import Path

RAW = "/data/HJ_thyroid/datasets_raw/tn3k_raw/Thyroid Dataset"
DDTI = f"{RAW}/DDTI dataset/DDTI_ext/1_or_data"
OUT_ROOT = "/data/HJ_thyroid/datasets"

# (소스명, 이미지 폴더, 마스크 폴더)
SOURCES = [
    ("tn3k_trainval", f"{RAW}/tn3k/trainval-image", f"{RAW}/tn3k/trainval-mask"),
    ("tn3k_test",     f"{RAW}/tn3k/test-image",     f"{RAW}/tn3k/test-mask"),
    ("ddti",          f"{DDTI}/image",              f"{DDTI}/mask"),
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP")
THRESH = 127            # 이진화 임계값
MIN_AREA_FRAC = 0.0005  # 이미지 대비 최소 결절 면적 비율(잡티 제거)
MIN_AREA_ABS = 15       # 최소 절대 면적(px)
APPROX_EPS_FRAC = 0.002 # 폴리곤 단순화 강도(둘레 대비)


def find_mask(mask_dir, stem):
    for e in IMG_EXTS:
        p = os.path.join(mask_dir, stem + e)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(mask_dir, stem + ".*"))
    return hits[0] if hits else None


def mask_to_polygons(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None, (0, 0)
    h, w = m.shape[:2]
    _, b = cv2.threshold(m, THRESH, 255, cv2.THRESH_BINARY)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    min_area = max(MIN_AREA_ABS, MIN_AREA_FRAC * w * h)
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = APPROX_EPS_FRAC * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        norm = c.astype(np.float32)
        norm[:, 0] = np.clip(norm[:, 0] / w, 0, 1)
        norm[:, 1] = np.clip(norm[:, 1] / h, 0, 1)
        polys.append(norm.reshape(-1))
    return polys, (w, h)


def main():
    total = {"img": 0, "labeled": 0, "empty": 0, "polys": 0}
    for name, img_dir, mask_dir in SOURCES:
        out_img = os.path.join(OUT_ROOT, name, "images")
        out_lbl = os.path.join(OUT_ROOT, name, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        imgs = sorted(
            p for p in glob.glob(os.path.join(img_dir, "*")) if p.endswith(IMG_EXTS)
        )
        s = {"img": 0, "labeled": 0, "empty": 0, "polys": 0, "nomask": 0}
        for ip in imgs:
            stem = Path(ip).stem
            mp = find_mask(mask_dir, stem)
            s["img"] += 1
            if mp is None:
                s["nomask"] += 1
                continue
            polys, _ = mask_to_polygons(mp)
            if not polys:  # 빈 마스크(결절 없음/변환 실패) -> 스킵
                s["empty"] += 1
                continue
            new = f"{name}_{stem}"
            link = os.path.join(out_img, new + Path(ip).suffix.lower())
            if not os.path.exists(link):
                os.symlink(os.path.abspath(ip), link)
            with open(os.path.join(out_lbl, new + ".txt"), "w") as f:
                for poly in polys:
                    f.write("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")
            s["labeled"] += 1
            s["polys"] += len(polys)
        print(f"[{name}] imgs={s['img']} labeled={s['labeled']} "
              f"empty={s['empty']} nomask={s['nomask']} polys={s['polys']}")
        for k in total:
            total[k] += s[k]
    print(f"\n[TOTAL] imgs={total['img']} labeled={total['labeled']} "
          f"empty={total['empty']} polys={total['polys']}")
    print(f"출력 위치: {OUT_ROOT}/<source>/images , /labels")


if __name__ == "__main__":
    main()
