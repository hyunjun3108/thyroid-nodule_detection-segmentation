#!/usr/bin/env python
"""
논문용 Phase 2~4 통합 결과표.

행: baseline nano(Phase2) / BRM nano(Phase3) / BRM+ONNX-FP16 / BRM+ONNX-INT8 / BRM+OpenVINO-INT8
열: held-out test Mask mAP50-95, Dice, HD95%, BoundaryIoU, Params(M), GFLOPs, 크기(MB), CPU(ms), GPU(ms)

- Phase2/3(pytorch) 정확도·경계: results.csv / test_metrics.json / eval_boundary 재계산
- Phase4(양자화) 수치: eval_quant.py 로직 재사용(경로별 평가)
- 5-fold 평균±std

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/final_table.py [--folds 0,1,2,3,4] [--fast]
  --fast : 경계 지표 재계산 생략(이미 있는 캐시 JSON 사용)
"""
import os
import sys
import json
import csv
import argparse
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_boundary as EB
import eval_quant as EQ

ROOT = "/data/HJ_thyroid"
CACHE = f"{ROOT}/runs/phase4_quant/final_table_cache.json"

# (표시명, 경로템플릿, 정확도 device, 지연 providers)
ROWS = [
    ("baseline nano (P2)",   f"{ROOT}/runs/phase2_baseline/n_fold{{f}}/weights/best.pt",              "0",   []),
    ("BRM nano (P3)",        f"{ROOT}/runs/phase3_brm_final/nbrm_fold{{f}}/weights/best.pt",          "0",   []),
    ("BRM + ONNX-FP32",      f"{ROOT}/runs/phase4_quant/fold{{f}}/fp32.onnx",                          "0",   ["CPUExecutionProvider", "CUDAExecutionProvider"]),
    ("BRM + ONNX-FP16",      f"{ROOT}/runs/phase4_quant/fold{{f}}/fp16.onnx",                          "0",   ["CUDAExecutionProvider"]),
    ("BRM + ONNX-INT8",      f"{ROOT}/runs/phase4_quant/fold{{f}}/int8.onnx",                          "cpu", ["CPUExecutionProvider"]),
    ("BRM + OpenVINO-INT8",  f"{ROOT}/runs/phase4_quant/fold{{f}}/openvino_int8_openvino_model",       "cpu", ["OpenVINO"]),
]


def torch_params_flops(pt_path):
    """PyTorch 체크포인트의 Params(M), GFLOPs@640."""
    try:
        from ultralytics import YOLO
        from ultralytics.utils import torch_utils as tu
        m = YOLO(pt_path).model
        return tu.get_num_params(m) / 1e6, tu.get_flops(m, imgsz=640)
    except Exception as e:
        print(f"  [warn] params/flops 실패: {str(e)[:70]}")
        return None, None


def read_test_json(base_dir):
    p = os.path.join(base_dir + "_test", "test_metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return float(json.load(f).get("metrics/mAP50-95(M)", 0) or 0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--fast", action="store_true", help="캐시가 있으면 재계산 생략")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",")]
    imgs = [l.strip() for l in open(EB.TEST_LIST) if l.strip()]

    cache = {}
    if args.fast and os.path.exists(CACHE):
        with open(CACHE) as f:
            cache = json.load(f)
        print(f"캐시 사용: {CACHE}")

    print(f"held-out test images: {len(imgs)}  folds={folds}\n")
    results = {}
    for name, tmpl, dev, provs in ROWS:
        if name in cache:
            results[name] = cache[name]
            print(f"[cache] {name}")
            continue
        per = []
        for f in folds:
            path = tmpl.format(f=f)
            if not os.path.exists(path):
                print(f"  [skip] {name} fold{f}: 없음")
                continue
            r = EQ.eval_one(name, path, dev, provs, imgs)
            # pytorch 행은 test_metrics.json 이 있으면 그 mAP 사용(학습 시 기록값)
            if path.endswith(".pt"):
                mp = read_test_json(os.path.dirname(os.path.dirname(path)))
                if mp is not None:
                    r["map"] = mp
                pm, fl = torch_params_flops(path)
                r["params"], r["flops"] = pm, fl
            per.append(r)
            print(f"  {name} fold{f}: mAP={r['map'] and round(r['map'],4)} Dice={r['dice']:.4f} "
                  f"HD95={r['hd']:.3f} size={r['size']:.2f}MB")
        results[name] = per

    with open(CACHE, "w") as f:
        json.dump(results, f, indent=2)

    def agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)

    # ---- 표 출력 ----
    cols = [("mAP50-95", "map", 4), ("Dice", "dice", 4), ("HD95%", "hd", 3),
            ("BIoU", "biou", 4), ("Params(M)", "params", 2), ("GFLOPs", "flops", 2),
            ("Size(MB)", "size", 2), ("CPU(ms)", "cpu", 2), ("GPU(ms)", "gpu", 2)]
    W = 22
    line = "=" * (W + 13 * len(cols))
    print("\n" + line)
    print("  Phase 2~4 통합 결과표 (held-out tn3k_test, 5-fold 평균)")
    print(line)
    print(f"  {'model':<{W}}" + "".join(f"{c[0]:<13}" for c in cols))
    csv_rows = [["model"] + [c[0] for c in cols]]
    for name, *_ in ROWS:
        rows = results.get(name) or []
        if not rows:
            continue
        cells, raw = [], [name]
        for _, key, dec in cols:
            m, s = agg(rows, key)
            cells.append(f"{m:.{dec}f}" if m is not None else "n/a")
            raw.append(f"{m:.{dec}f}" if m is not None else "")
        print(f"  {name:<{W}}" + "".join(f"{c:<13}" for c in cells))
        csv_rows.append(raw)
    print("\n  ↑좋음: mAP/Dice/BIoU | ↓좋음: HD95/Size/CPU/GPU")
    print("  CPU=AMD EPYC 9654(멀티스레드), GPU=A100 / bs=1 순수 forward")

    out_csv = f"{ROOT}/runs/phase4_quant/final_table.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"\nCSV 저장: {out_csv}")


if __name__ == "__main__":
    main()
