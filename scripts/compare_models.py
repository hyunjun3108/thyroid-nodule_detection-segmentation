#!/usr/bin/env python
"""
YOLOv11-seg 모델 비교(정확도 + 효율). 임의의 run 세트를 나란히 비교.

각 run 세트는 "label=BASE" 로 지정하며, BASE 는 project/prefix 형태:
  - pooled fold : {BASE}_fold{0..4}(+_test)
  - LODO        : {BASE}_tn3k2ddti, {BASE}_ddti2tn3k
예)
  # 기본: Phase2 nano vs small
  python compare_models.py
  # Phase3: baseline nano vs BRM nano
  python compare_models.py --runs "baseline=runs/phase2_baseline/n" "BRM=runs/phase3_brm/nbrm" \
                           --out runs/phase3_brm

정확도: pooled 5-fold VAL(best mAP50-95(M), 평균±std), held-out tn3k_test, LODO 양방향.
효율  : Params(M), GFLOPs@640, weight(MB), GPU/CPU 추론 지연(bs=1)+FPS.
출력  : 콘솔 표 + {out}/comparison.csv (+ matplotlib 있으면 comparison.png).

반드시 thyroid_seg 환경으로 실행:
  /data/my_envs/thyroid_seg/bin/python scripts/compare_models.py ...
"""
import os
import glob
import json
import csv
import argparse
import statistics as st

METRICS = [
    ("metrics/mAP50(B)", "Box50"),
    ("metrics/mAP50-95(B)", "Box50-95"),
    ("metrics/mAP50(M)", "Mask50"),
    ("metrics/mAP50-95(M)", "Mask50-95"),
]


# ---------- 정확도 읽기 ----------
def read_best_row(run_dir):
    p = os.path.join(run_dir, "results.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get("metrics/mAP50-95(M)", 0) or 0))


def read_test(run_dir):
    p = os.path.join(run_dir + "_test", "test_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def agg(rows):
    good = [r for r in rows if r]
    out = {}
    for k, lbl in METRICS:
        vals = [float(r.get(k, 0) or 0) for r in good]
        out[lbl] = (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0, len(vals)) if vals else (None, None, 0)
    return out


def collect_accuracy(base):
    val = agg([read_best_row(f"{base}_fold{i}") for i in range(5)])
    test = agg([read_test(f"{base}_fold{i}") for i in range(5)])
    lodo = {
        "TN3K->DDTI": read_best_row(f"{base}_tn3k2ddti"),
        "DDTI->TN3K": read_best_row(f"{base}_ddti2tn3k"),
    }
    return val, test, lodo


# ---------- 효율 ----------
def measure_latency(weight, device="0", imgsz=640, warmup=30, iters=200):
    import time
    import torch
    from ultralytics import YOLO
    dev = torch.device("cpu") if str(device) == "cpu" else (
        torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu"))
    is_cuda = dev.type == "cuda"
    model = YOLO(weight).model.to(dev).eval()
    x = torch.randn(1, 3, imgsz, imgsz, device=dev)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms = (t1 - t0) / iters * 1000.0
    return {"latency_ms": ms, "fps": 1000.0 / ms, "device": str(dev), "threads": torch.get_num_threads()}


def model_efficiency(base, device="0", measure_speed=True, iters=200, cpu_iters=20):
    cands = sorted(glob.glob(f"{base}_fold*/weights/best.pt"))
    if not cands:
        return None
    w = cands[0]
    out = {"weight": w, "size_mb": os.path.getsize(w) / 1e6, "params_m": None, "flops": None,
           "gpu_latency_ms": None, "gpu_fps": None, "gpu_dev": None,
           "cpu_latency_ms": None, "cpu_fps": None, "cpu_threads": None}
    try:
        from ultralytics import YOLO
        from ultralytics.utils import torch_utils as tu
        m = YOLO(w).model
        out["params_m"] = tu.get_num_params(m) / 1e6
        out["flops"] = tu.get_flops(m, imgsz=640)
    except Exception as e:
        print(f"  [warn] {base} 효율 계산 실패: {e}")
    if measure_speed:
        try:
            g = measure_latency(w, device=device, iters=iters)
            out["gpu_latency_ms"], out["gpu_fps"], out["gpu_dev"] = g["latency_ms"], g["fps"], g["device"]
        except Exception as e:
            print(f"  [warn] {base} GPU 지연 실패: {e}")
        try:
            c = measure_latency(w, device="cpu", warmup=5, iters=cpu_iters)
            out["cpu_latency_ms"], out["cpu_fps"], out["cpu_threads"] = c["latency_ms"], c["fps"], c["threads"]
        except Exception as e:
            print(f"  [warn] {base} CPU 지연 실패: {e}")
    return out


# ---------- 출력 ----------
def cell(t):
    m, s, n = t
    return "  n/a" if m is None else f"{m:.3f}±{s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["nano=runs/phase2_baseline/n", "small=runs/phase2_baseline/s"],
                    help='"label=PROJECT/PREFIX" 목록')
    ap.add_argument("--out", default="runs/phase2_baseline", help="CSV/PNG 저장 폴더")
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cpu-iters", type=int, default=20)
    args = ap.parse_args()

    # 환경 사전 점검
    try:
        import ultralytics  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:
        import sys
        print("\n" + "!" * 66)
        print("  [환경 오류] ultralytics/torch 임포트 실패 → Params/GFLOPs/지연 n/a.")
        print(f"    현재 python : {sys.executable}\n    원인 : {e}")
        print("  /data/my_envs/thyroid_seg/bin/python 로 실행하세요.")
        print("!" * 66 + "\n")

    runs = []
    for spec in args.runs:
        label, base = spec.split("=", 1)
        runs.append((label, base))

    acc = {lbl: collect_accuracy(base) for lbl, base in runs}
    eff = {lbl: model_efficiency(base, device=args.device, measure_speed=not args.no_speed,
                                 iters=args.iters, cpu_iters=args.cpu_iters) for lbl, base in runs}
    labels = [lbl for lbl, _ in runs]
    W, C = 22, 18

    print("\n" + "=" * (W + C * len(labels) + 12))
    print("  YOLOv11-seg 비교: " + "  vs  ".join(labels))
    print("=" * (W + C * len(labels) + 12))

    def header():
        print(f"  {'metric':<{W}}" + "".join(f"{l:<{C}}" for l in labels) +
              ("Δ(2-1)" if len(labels) == 2 else ""))

    def section(title, idx):
        print(f"\n[{title}]")
        header()
        for _, lbl_m in METRICS:
            cells = [acc[l][idx].get(lbl_m) for l in labels]
            line = f"  {lbl_m:<{W}}" + "".join(f"{cell(c):<{C}}" for c in cells)
            if len(labels) == 2 and cells[0][0] is not None and cells[1][0] is not None:
                line += f"{cells[1][0]-cells[0][0]:+.3f}"
            print(line)

    section("VAL — pooled 5-fold (mean±std)", 0)
    section("TEST — held-out tn3k_test (mean±std)", 1)

    print("\n[LODO — cross-dataset Mask50-95]")
    header()
    for setting in ("TN3K->DDTI", "DDTI->TN3K"):
        def g(l):
            r = acc[l][2][setting]
            return f"{float(r.get('metrics/mAP50-95(M)',0) or 0):.3f}" if r else "n/a"
        print(f"  {setting:<{W}}" + "".join(f"{g(l):<{C}}" for l in labels))

    print("\n[EFFICIENCY]")
    print(f"  {'':<{W}}" + "".join(f"{l:<{C}}" for l in labels))
    for key, lbl, unit in [("params_m", "Params", "M"), ("flops", "GFLOPs", ""),
                           ("size_mb", "Weight size", "MB"),
                           ("gpu_latency_ms", "GPU Latency(bs=1)", "ms"), ("gpu_fps", "GPU FPS(bs=1)", ""),
                           ("cpu_latency_ms", "CPU Latency(bs=1)", "ms"), ("cpu_fps", "CPU FPS(bs=1)", "")]:
        vals = [(eff[l].get(key) if eff[l] else None) for l in labels]
        cells = [f"{v:.2f}{unit}" if v is not None else "n/a" for v in vals]
        print(f"  {lbl:<{W}}" + "".join(f"{c:<{C}}" for c in cells))
    e0 = eff[labels[0]] or {}
    print(f"  (GPU={e0.get('gpu_dev')}, CPU threads={e0.get('cpu_threads')}, batch=1, 순수 forward)")

    # CSV
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "comparison.csv")
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "metric"] + labels)
        for split_name, idx in [("val", 0), ("test", 1)]:
            for _, lbl_m in METRICS:
                wr.writerow([split_name, lbl_m] + [
                    (acc[l][idx].get(lbl_m)[0]) for l in labels])
        for key in ("params_m", "flops", "size_mb", "gpu_latency_ms", "gpu_fps", "cpu_latency_ms", "cpu_fps"):
            wr.writerow(["efficiency", key] + [(eff[l].get(key) if eff[l] else None) for l in labels])
    print(f"\nCSV 저장: {out_csv}")

    # 산점도(정확도 vs 파라미터)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = []
        for l in labels:
            e = eff[l]
            v = acc[l][0].get("Mask50-95")
            if e and e.get("params_m") and v[0] is not None:
                pts.append((l, e["params_m"], v[0]))
        if len(pts) >= 2:
            fig, ax = plt.subplots(figsize=(5, 4))
            for name, x, y in pts:
                ax.scatter(x, y, s=80)
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6))
            ax.set_xlabel("Params (M)")
            ax.set_ylabel("VAL Mask mAP50-95 (5-fold mean)")
            ax.set_title("Accuracy vs Model Size")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_png = os.path.join(args.out, "comparison.png")
            fig.savefig(out_png, dpi=120)
            print(f"산점도 저장: {out_png}")
    except Exception as e:
        print(f"(plot 생략: {e})")


if __name__ == "__main__":
    main()
