#!/usr/bin/env python
"""
Phase 2: YOLOv11-seg 베이스라인 학습 스크립트.

data.yaml / 모델 / 이름 / device 를 인자로 받아 학습하고,
--test 지정 시 data.yaml 의 test 스플릿(held-out tn3k_test)으로 추가 평가.

예)
  python train_seg.py \
    --data datasets/splits/pooled_5fold/fold0/data.yaml \
    --model /data/HJ_thyroid/_phase0_check/yolo11n-seg.pt \
    --name n_fold0 --device 0 --test
"""
import argparse
import json
import os
from ultralytics import YOLO

DEFAULT_MODEL = "/data/HJ_thyroid/_phase0_check/yolo11n-seg.pt"
DEFAULT_PROJECT = "/data/HJ_thyroid/runs/phase2_baseline"


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv11-seg thyroid nodule training")
    ap.add_argument("--data", required=True, help="data.yaml 경로")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="pretrained 가중치(.pt) 또는 모델 yaml (기본: 로컬 yolo11n-seg.pt)")
    ap.add_argument("--weights", default=None,
                    help="--model 이 yaml 일 때 이식할 pretrained 가중치(.pt). "
                         "지정 시 YOLO(yaml).load(weights)")
    ap.add_argument("--name", required=True, help="run 이름 (예: n_fold0)")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42,
                    help="난수 시드. 시드 반복(예: LODO 3회)으로 실행간 변동을 분리할 때 변경")
    ap.add_argument("--test", action="store_true",
                    help="학습 후 data.yaml 의 test 스플릿으로 held-out 평가")
    return ap.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    if args.weights:  # yaml 아키텍처에 pretrained 가중치 이식(새 분기만 랜덤 초기화)
        model = model.load(args.weights)
    model.train(
        data=args.data,
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        deterministic=True,
        overlap_mask=True,
        mask_ratio=4,
        plots=True,
        exist_ok=True,
    )
    if args.test:
        tm = model.val(
            data=args.data,
            split="test",
            project=args.project,
            name=args.name + "_test",
            device=args.device,
            plots=True,
            exist_ok=True,
        )
        # test 지표를 JSON으로 저장(나중에 집계용)
        out = os.path.join(args.project, args.name + "_test", "test_metrics.json")
        with open(out, "w") as f:
            json.dump({k: float(v) for k, v in tm.results_dict.items()}, f, indent=2)
        print(f"[test metrics] saved -> {out}")


if __name__ == "__main__":
    main()
