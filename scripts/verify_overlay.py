#!/usr/bin/env python
"""변환된 YOLO-seg 폴리곤을 이미지 위에 그려 정합성 육안 확인용 미리보기 생성."""
import os
import glob
import random
import cv2
import numpy as np
from pathlib import Path

OUT_ROOT = "/data/HJ_thyroid/datasets"
PREVIEW = "/data/HJ_thyroid/datasets/_preview"
N_PER_SOURCE = 6

os.makedirs(PREVIEW, exist_ok=True)
for name in ("tn3k_trainval", "tn3k_test", "ddti"):
    lbls = sorted(glob.glob(os.path.join(OUT_ROOT, name, "labels", "*.txt")))
    if not lbls:
        print(f"[{name}] 라벨 없음(스킵)")
        continue
    for lp in random.sample(lbls, min(N_PER_SOURCE, len(lbls))):
        stem = Path(lp).stem
        imgs = glob.glob(os.path.join(OUT_ROOT, name, "images", stem + ".*"))
        if not imgs:
            continue
        img = cv2.imread(imgs[0])
        h, w = img.shape[:2]
        for line in open(lp):
            v = line.split()
            pts = np.array(v[1:], dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            cv2.polylines(img, [pts.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(PREVIEW, f"{stem}.png"), img)
    print(f"[{name}] 미리보기 생성 완료")
print(f"\n미리보기 폴더: {PREVIEW} (IDE에서 PNG 열어 확인)")
