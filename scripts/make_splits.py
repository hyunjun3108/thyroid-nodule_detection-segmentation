#!/usr/bin/env python
"""
Phase 1-E: YOLO-seg 학습용 분할 구성 + data.yaml 생성.

설계:
- tn3k_test(614) = held-out 고정 test (모든 실험 공통 최종 test)
- Pooled 5-fold CV: tn3k_trainval + ddti(3516) 를 소스별 stratified 5분할
    각 fold -> train(4/5) + val(1/5), test=tn3k_test
- LODO(양방향):
    tn3k2ddti: train=TN3K(trainval+test), val/test=DDTI
    ddti2tn3k: train=DDTI, val/test=TN3K(trainval+test)

출력:
- datasets/splits/pooled_5fold/fold{0..4}/{train.txt,val.txt,data.yaml}
- datasets/splits/lodo/{tn3k2ddti,ddti2tn3k}/{train.txt,val.txt,data.yaml}
- datasets/splits/heldout_test/tn3k_test.txt

사용: python make_splits.py
"""
import os
import glob
import random
from pathlib import Path

ROOT = "/data/HJ_thyroid/datasets"
SPLITS = os.path.join(ROOT, "splits")
SEED = 42
K = 5

SOURCES = ["tn3k_trainval", "tn3k_test", "ddti"]


def list_images(source):
    """라벨(.txt)이 존재하는 이미지 절대경로만 반환."""
    img_dir = os.path.join(ROOT, source, "images")
    lbl_dir = os.path.join(ROOT, source, "labels")
    out = []
    for ip in sorted(glob.glob(os.path.join(img_dir, "*"))):
        stem = Path(ip).stem
        if os.path.exists(os.path.join(lbl_dir, stem + ".txt")):
            out.append(os.path.abspath(ip))
    return out


def write_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(items) + "\n")


def write_yaml(path, train_rel, val_rel, test_rel=None):
    lines = [
        f"path: {ROOT}",
        f"train: {train_rel}",
        f"val: {val_rel}",
    ]
    if test_rel:
        lines.append(f"test: {test_rel}")
    lines += ["names:", "  0: nodule"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def stratified_folds(source_to_imgs, k, seed):
    """소스별로 균등하게 k개 fold에 분배. folds[i] = 이미지 경로 리스트."""
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for src, imgs in source_to_imgs.items():
        imgs = imgs[:]
        rng.shuffle(imgs)
        for i, ip in enumerate(imgs):
            folds[i % k].append(ip)
    for f in folds:
        rng.shuffle(f)
    return folds


def main():
    imgs = {s: list_images(s) for s in SOURCES}
    for s in SOURCES:
        print(f"[{s}] labeled images = {len(imgs[s])}")

    # ---- held-out test ----
    test_rel = "splits/heldout_test/tn3k_test.txt"
    write_list(os.path.join(ROOT, test_rel), imgs["tn3k_test"])
    print(f"\n[held-out test] {test_rel}  ({len(imgs['tn3k_test'])} imgs)")

    # ---- Pooled 5-fold (tn3k_trainval + ddti) ----
    pool = {"tn3k_trainval": imgs["tn3k_trainval"], "ddti": imgs["ddti"]}
    folds = stratified_folds(pool, K, SEED)
    print("\n[pooled 5-fold] fold sizes:", [len(f) for f in folds])
    for i in range(K):
        val = folds[i]
        train = [ip for j in range(K) if j != i for ip in folds[j]]
        d = os.path.join(SPLITS, "pooled_5fold", f"fold{i}")
        write_list(os.path.join(d, "train.txt"), train)
        write_list(os.path.join(d, "val.txt"), val)
        write_yaml(
            os.path.join(d, "data.yaml"),
            f"splits/pooled_5fold/fold{i}/train.txt",
            f"splits/pooled_5fold/fold{i}/val.txt",
            test_rel,
        )
        # 소스 구성 확인
        n_tn = sum("tn3k_trainval_" in Path(p).name for p in val)
        n_dd = sum("ddti_" in Path(p).name for p in val)
        print(f"  fold{i}: train={len(train)} val={len(val)} "
              f"(val tn3k={n_tn}, ddti={n_dd})")

    # ---- LODO (양방향) ----
    tn3k_all = imgs["tn3k_trainval"] + imgs["tn3k_test"]
    ddti_all = imgs["ddti"]
    lodo = {
        "tn3k2ddti": (tn3k_all, ddti_all),   # train TN3K -> eval DDTI
        "ddti2tn3k": (ddti_all, tn3k_all),   # train DDTI -> eval TN3K
    }
    print("\n[LODO]")
    for name, (train, target) in lodo.items():
        d = os.path.join(SPLITS, "lodo", name)
        write_list(os.path.join(d, "train.txt"), train)
        write_list(os.path.join(d, "val.txt"), target)  # val==test (left-out 도메인)
        write_yaml(
            os.path.join(d, "data.yaml"),
            f"splits/lodo/{name}/train.txt",
            f"splits/lodo/{name}/val.txt",
            f"splits/lodo/{name}/val.txt",
        )
        print(f"  {name}: train={len(train)} val/test={len(target)}")

    print(f"\n완료. data.yaml 위치: {SPLITS}/(pooled_5fold|lodo)/**/data.yaml")


if __name__ == "__main__":
    main()
