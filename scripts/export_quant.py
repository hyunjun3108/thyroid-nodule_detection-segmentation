#!/usr/bin/env python
"""
Phase 4: BRM nano 를 ONNX FP32 / FP16 / INT8 로 export/양자화.

- FP32/FP16: ultralytics YOLO.export(format="onnx")
- INT8      : onnxruntime 정적 PTQ (quantize_static), tn3k_trainval 이미지로 캘리브레이션
산출물: runs/phase4_quant/fold{N}/{fp32,fp16,int8}.onnx

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/export_quant.py --fold 0 [--calib 150]
"""
import os
import glob
import shutil
import random
import argparse
import numpy as np
import cv2
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

ROOT = "/data/HJ_thyroid"
CALIB_DIR = f"{ROOT}/datasets/tn3k_trainval/images"
IMGSZ = 640


def letterbox(img, new=IMGSZ, color=114):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess(path, imgsz=IMGSZ):
    img = cv2.imread(path)  # BGR
    img = letterbox(img, imgsz)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # RGB, CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[None]  # (1,3,H,W)


class CalibReader(CalibrationDataReader):
    def __init__(self, files, input_name):
        self.files = files
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x = preprocess(self.files[self.i])
        self.i += 1
        return {self.input_name: x}


def export_onnx(pt, half, device):
    """ultralytics onnx export -> 생성된 onnx 경로 반환."""
    return YOLO(pt).export(format="onnx", imgsz=IMGSZ, opset=13,
                           simplify=True, half=half, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--calib", type=int, default=150, help="캘리브레이션 이미지 수")
    ap.add_argument("--device", default="0")
    ap.add_argument("--only", choices=["all", "onnx", "openvino"], default="all",
                    help="생성 대상: all(기본) / onnx(FP32·FP16·onnx-INT8) / openvino(OpenVINO INT8만)")
    args = ap.parse_args()
    do_onnx = args.only in ("all", "onnx")
    do_ov = args.only in ("all", "openvino")

    pt = f"{ROOT}/runs/phase3_brm_final/nbrm_fold{args.fold}/weights/best.pt"
    assert os.path.exists(pt), f"가중치 없음: {pt}"
    outdir = f"{ROOT}/runs/phase4_quant/fold{args.fold}"
    os.makedirs(outdir, exist_ok=True)

    fp32 = f"{outdir}/fp32.onnx"
    fp16 = f"{outdir}/fp16.onnx"
    int8 = f"{outdir}/int8.onnx"
    ovdst = f"{outdir}/openvino_int8_openvino_model"

    if do_onnx:
        # --- FP32 ---
        print("== ONNX FP32 export ==")
        p = export_onnx(pt, half=False, device=args.device)
        shutil.copy(p, fp32)
        print("  ->", fp32, "%.2f MB" % (os.path.getsize(fp32) / 1e6))

        # --- FP16 ---
        print("== ONNX FP16 export ==")
        p = export_onnx(pt, half=True, device=args.device)
        shutil.copy(p, fp16)
        print("  ->", fp16, "%.2f MB" % (os.path.getsize(fp16) / 1e6))

        # --- INT8 (정적 PTQ) ---
        print("== ONNX INT8 (static PTQ) ==")
        input_name = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]).get_inputs()[0].name
        files = sorted(glob.glob(f"{CALIB_DIR}/*"))
        random.Random(42).shuffle(files)
        files = files[:args.calib]
        print(f"  캘리브레이션 이미지: {len(files)}장, input='{input_name}'")
        prep = f"{outdir}/_fp32_prep.onnx"
        quant_pre_process(fp32, prep)
        # 검출 head(/model.23/)는 양자화 제외 — INT8화 시 cls/box 출력이 파괴됨(진단 확인).
        import onnx
        head = [n.name for n in onnx.load(prep).graph.node if n.name and "/model.23/" in n.name]
        print(f"  양자화 제외 head 노드: {len(head)}개 (/model.23/)")
        quantize_static(
            prep, int8, CalibReader(files, input_name),
            quant_format=QuantFormat.QDQ, per_channel=True,
            weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
            nodes_to_exclude=head,
        )
        os.remove(prep)
        print("  ->", int8, "%.2f MB" % (os.path.getsize(int8) / 1e6))

    if do_ov:
        # --- OpenVINO INT8 (NNCF PTQ, YOLO-aware) ---
        print("== OpenVINO INT8 (NNCF PTQ) ==")
        data_yaml = f"{ROOT}/datasets/splits/pooled_5fold/fold{args.fold}/data.yaml"
        ovsrc = YOLO(pt).export(format="openvino", int8=True, imgsz=IMGSZ, data=data_yaml)
        if os.path.exists(ovdst):
            shutil.rmtree(ovdst)
        shutil.copytree(ovsrc, ovdst)
        ovsz = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(ovdst) for f in fs)
        print("  ->", ovdst, "%.2f MB" % (ovsz / 1e6))

    def _sz(p):
        if not os.path.exists(p):
            return None
        if os.path.isdir(p):
            return sum(os.path.getsize(os.path.join(d, f))
                       for d, _, fs in os.walk(p) for f in fs) / 1e6
        return os.path.getsize(p) / 1e6

    print("\n요약(MB):", " ".join(
        f"{n}={v:.2f}" for n, v in
        [("FP32", _sz(fp32)), ("FP16", _sz(fp16)), ("INT8", _sz(int8)), ("OV_INT8", _sz(ovdst))]
        if v is not None))


if __name__ == "__main__":
    main()
