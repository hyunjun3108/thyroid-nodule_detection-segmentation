#!/usr/bin/env python
"""
Phase 4: 양자화 정밀도별 평가 — 크기 / CPU·GPU 지연 / 정확도(mAP + 경계).

정밀도: pytorch(FP32 원본), onnx fp32, onnx fp16, onnx int8(head 제외 PTQ).
정확도는 held-out tn3k_test 기준. INT8은 CPU provider에서 평가/측정(그게 배포 타깃).

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/eval_quant.py --folds 0
  /data/my_envs/thyroid_seg/bin/python scripts/eval_quant.py --folds 0,1,2,3,4
"""
import os
import time
import argparse
import statistics as st
import sys
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_boundary as EB  # boundary 지표/전처리 재사용 (scripts/ 를 path에 추가)

ROOT = "/data/HJ_thyroid"
TEST_LIST = f"{ROOT}/datasets/splits/heldout_test/tn3k_test.txt"
QDIR = f"{ROOT}/runs/phase4_quant"
BRM = f"{ROOT}/runs/phase3_brm_final"

# (라벨, 파일템플릿, 정확도 device, 지연 providers)
PRECS = [
    ("pytorch",  BRM + "/nbrm_fold{f}/weights/best.pt", "0",   []),
    ("onnx_fp32", QDIR + "/fold{f}/fp32.onnx",          "0",   ["CPUExecutionProvider", "CUDAExecutionProvider"]),
    ("onnx_fp16", QDIR + "/fold{f}/fp16.onnx",          "0",   ["CUDAExecutionProvider"]),
    ("onnx_int8", QDIR + "/fold{f}/int8.onnx",          "cpu", ["CPUExecutionProvider"]),
    ("openvino_int8", QDIR + "/fold{f}/openvino_int8_openvino_model",  "cpu", ["OpenVINO"]),
]


def ov_latency(ov_dir, iters=100, warmup=15, imgsz=640):
    """OpenVINO CPU 순수 forward 지연(ms/img, bs=1)."""
    import glob as _g
    try:
        import openvino as ov
        xml = _g.glob(os.path.join(ov_dir, "*.xml"))[0]
        core = ov.Core()
        cm = core.compile_model(core.read_model(xml), "CPU")
        ireq = cm.create_infer_request()
        x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        for _ in range(warmup):
            ireq.infer({0: x})
        t0 = time.perf_counter()
        for _ in range(iters):
            ireq.infer({0: x})
        return (time.perf_counter() - t0) / iters * 1000.0
    except Exception as e:
        print(f"    [warn] ov_latency 실패: {str(e)[:80]}")
        return None


def path_size_mb(path):
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(d, f))
                   for d, _, fs in os.walk(path) for f in fs) / 1e6
    return os.path.getsize(path) / 1e6


def ort_latency(onnx_path, provider, iters=100, warmup=15, imgsz=640):
    if provider == "CUDAExecutionProvider" and provider not in ort.get_available_providers():
        return None
    try:
        sess = ort.InferenceSession(onnx_path, providers=[provider])
    except Exception:
        return None
    inp = sess.get_inputs()[0]
    x = np.random.rand(1, 3, imgsz, imgsz).astype(
        np.float16 if "float16" in inp.type else np.float32)
    fd = {inp.name: x}
    for _ in range(warmup):
        sess.run(None, fd)
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, fd)
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def test_map(model_path, device):
    """held-out test Mask50-95 (ultralytics val)."""
    try:
        r = YOLO(model_path, task="segment").val(
            data=f"{ROOT}/datasets/splits/pooled_5fold/fold0/data.yaml",  # test 스플릿 공통(tn3k_test)
            split="test", device=device, verbose=False, plots=False)
        return float(r.results_dict.get("metrics/mAP50-95(M)", 0) or 0)
    except Exception as e:
        print(f"    [warn] mAP 실패({os.path.basename(model_path)}): {str(e)[:80]}")
        return None


def eval_one(label, path, acc_dev, providers, imgs):
    size_mb = path_size_mb(path)
    cpu_lat = gpu_lat = None
    for p in providers:
        if p == "OpenVINO":
            cpu_lat = ov_latency(path)
        elif p == "CPUExecutionProvider":
            cpu_lat = ort_latency(path, p)
        elif p == "CUDAExecutionProvider":
            gpu_lat = ort_latency(path, p)
    biou, hd, dice = EB.eval_fold(path, imgs, acc_dev, task="segment")  # predict 기반 경계 지표
    mp = test_map(path, acc_dev)
    return dict(size=size_mb, cpu=cpu_lat, gpu=gpu_lat, biou=biou, hd=hd, dice=dice, map=mp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--cache", default=f"{QDIR}/eval_cache.json",
                    help="(fold,precision)별 결과 캐시 — 크래시 후 재실행 시 완료분 건너뜀")
    ap.add_argument("--no-cache", action="store_true", help="캐시 무시하고 전부 재계산")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",")]
    imgs = [l.strip() for l in open(TEST_LIST) if l.strip()]
    print(f"held-out test images: {len(imgs)}  folds={folds}")

    import json
    cache = {}
    if not args.no_cache and os.path.exists(args.cache):
        with open(args.cache) as fh:
            cache = json.load(fh)
        print(f"캐시 로드: {args.cache} ({len(cache)}건)")

    # NOTE: precision-major 순서. OpenVINO 백엔드를 로드하면 프로세스의 CUDA가
    # 비활성화(CUDA_VISIBLE_DEVICES=-1)되므로, GPU를 쓰는 정밀도를 모두 끝낸 뒤
    # openvino_int8(PRECS 마지막)을 실행해야 한다.
    acc = {lbl: [] for lbl, *_ in PRECS}
    for lbl, tmpl, acc_dev, provs in PRECS:
        print(f"\n=== {lbl} ===")
        for f in folds:
            path = tmpl.format(f=f)
            if not os.path.exists(path):
                print(f"  [skip] {lbl}: 파일 없음 {path}")
                continue
            ckey = f"{f}|{lbl}"
            if ckey in cache:
                r = cache[ckey]
                acc[lbl].append(r)
                print(f"  [cache] fold{f} mAP={r['map'] and round(r['map'],4)} Dice={r['dice']:.4f}")
                continue
            r = eval_one(lbl, path, acc_dev, provs, imgs)
            cache[ckey] = r
            with open(args.cache, "w") as fh:  # 매 항목마다 저장(크래시 대비)
                json.dump(cache, fh, indent=1)
            acc[lbl].append(r)
            print(f"  fold{f} size={r['size']:.2f}MB cpu={r['cpu'] and round(r['cpu'],2)}ms "
                  f"gpu={r['gpu'] and round(r['gpu'],2)}ms mAP={r['map'] and round(r['map'],4)} "
                  f"Dice={r['dice']:.4f} HD95={r['hd']:.3f} BIoU={r['biou']:.4f}")

    def ms(vals):
        vals = [v for v in vals if v is not None]
        return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0) if vals else (None, 0.0)

    print("\n" + "=" * 96)
    print(f"  Phase 4 양자화 비교 (held-out test, folds={folds})")
    print("=" * 96)
    hdr = ["precision", "size(MB)", "CPU(ms)", "GPU(ms)", "mAP50-95", "Dice", "HD95%", "BIoU"]
    print("  " + "".join(f"{h:<12}" for h in hdr))
    for lbl, *_ in PRECS:
        rows = acc[lbl]
        if not rows:
            continue
        def g(k): return ms([r[k] for r in rows])[0]
        def cell(v, dec=2): return f"{v:.{dec}f}" if v is not None else "n/a"
        print("  " + f"{lbl:<12}" + f"{cell(g('size')):<12}" + f"{cell(g('cpu')):<12}" +
              f"{cell(g('gpu')):<12}" + f"{cell(g('map'),4):<12}" + f"{cell(g('dice'),4):<12}" +
              f"{cell(g('hd'),3):<12}" + f"{cell(g('biou'),4):<12}")
    print("\n주의: size/CPU/GPU/HD95 ↓ 좋음, mAP/Dice/BIoU ↑ 좋음. INT8은 CPU 기준.")


if __name__ == "__main__":
    main()
