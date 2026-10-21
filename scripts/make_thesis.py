#!/usr/bin/env python
"""
학위논문 LaTeX 패키지(thesis/)의 '생성물'을 만든다 — 표(tables/*.tex)와 그림(figures/*.pdf).

본문 장(chapters/*.tex)은 사람이 고치는 파일이라 이 스크립트는 건드리지 않는다.
실험을 다시 돌려 수치가 바뀌면 이 스크립트만 다시 실행하면 표가 따라 갱신된다.

원자료:
  runs/phase4_quant/final_table_cache.json   Phase 2~4 fold별 정확도·효율
  runs/stats/paired_stats.csv                대응표본 검정
  runs/stats/per_image_cache.json            이미지별 경계 지표(LODO 시드별 평균 계산용)
  runs/phase3_sweep/sweep_table.csv          경계 gain 스윕 (scripts/sweep_eval.py)
  runs/phase2_baseline/{n,s}_fold*_test/     nano·small held-out 지표
  runs/**/results.csv                        LODO 체크포인트 선택 epoch 지표
  datasets/**/labels, datasets/splits        데이터셋 통계·분할 구성

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/make_thesis.py [--zip]
"""
import os
import re
import csv
import glob
import json
import shutil
import argparse
import collections
import numpy as np

ROOT = "/data/HJ_thyroid"
TH = f"{ROOT}/thesis"
TAB = f"{TH}/tables"
FIGD = f"{TH}/figures"
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [42, 43, 44]


# ------------------------------------------------------------------ 공통
def write(name, body):
    os.makedirs(TAB, exist_ok=True)
    with open(f"{TAB}/{name}.tex", "w") as f:
        f.write("% 자동 생성 파일 — scripts/make_thesis.py 가 덮어쓴다. 직접 고치지 말 것.\n")
        f.write(body)
    print(f"  tables/{name}.tex")


def ms(vals, d=4):
    """평균 ± 표본표준편차(ddof=1) 문자열."""
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}"


def mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "n.s."


def pfmt(p):
    if p < 1e-3:
        return f"{p:.2e}".replace("e-0", "e-").replace("e-", "\\times10^{-") + "}"
    return f"{p:.4f}" if p < 1e-2 else f"{p:.3f}"


def table(caption, label, colspec, header, rows, note=None, size="\\small", wide=False, colsep=None):
    env = "table*" if wide else "table"
    # 표 목차에는 첫 문장만 싣는다(소수점 뒤의 '.'은 문장 끝으로 보지 않음)
    m = re.search(r"(?<!\d)\.(?=\s|$)", caption)
    short = caption[:m.end()] if m and m.end() < len(caption.rstrip()) else None
    cap = f"\\caption[{short}]{{{caption}}}" if short else f"\\caption{{{caption}}}"
    out = [f"\\begin{{{env}}}[htbp]", "\\centering", size, cap, f"\\label{{{label}}}"]
    if colsep:
        out.append(f"\\setlength{{\\tabcolsep}}{{{colsep}}}")
    # 쪽 폭보다 넓을 때만 줄이고, 좁은 표는 그대로 둔다
    out += ["\\begin{adjustbox}{max width=\\linewidth}",
            f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    for r in rows:
        # 규칙선만 그대로 두고, \multirow 로 시작하는 행을 포함한 나머지 행에는 줄 끝을 붙인다
        out.append(r if r.startswith(("\\midrule", "\\cmidrule", "\\addlinespace")) else r + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        out += ["\\par\\vspace{3pt}", f"\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}"]
    out.append(f"\\end{{{env}}}\n")
    return "\n".join(out)


def load_json(p):
    with open(p) as f:
        return json.load(f)


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def fitness(row):
    # ultralytics SegmentMetrics.fitness: box·mask 각각 0.1·mAP50 + 0.9·mAP50-95 의 합
    return (0.1 * row["metrics/mAP50(B)"] + 0.9 * row["metrics/mAP50-95(B)"]
            + 0.1 * row["metrics/mAP50(M)"] + 0.9 * row["metrics/mAP50-95(M)"])


def best_epoch_metrics(run_dir):
    """best.pt 가 저장된 epoch(fitness 최대)의 검증 지표. 열 최댓값이 아니라 이 값을 쓴다."""
    import pandas as pd
    d = pd.read_csv(f"{run_dir}/results.csv")
    d.columns = [c.strip() for c in d.columns]
    i = int(np.argmax([fitness(r) for _, r in d.iterrows()]))
    return d.iloc[i], int(d["epoch"].iloc[i]), len(d)


# ------------------------------------------------------------------ 그림
FIGS = ["fig1_boundary_improvement", "fig2_accuracy_vs_latency", "fig3_quantization",
        "fig4_qualitative", "fig5_architecture", "fig6_dual_path", "fig7_boundary_loss",
        "fig8_pipeline", "fig9_significance_2x2", "fig10_dataset", "fig11_tn3k_samples",
        "fig12_ddti_samples", "fig13_dataset_stats"]


def copy_figures():
    os.makedirs(FIGD, exist_ok=True)
    for f in FIGS:
        src = f"{ROOT}/runs/figures/{f}.pdf"
        assert os.path.exists(src), f"그림 없음: {src}"
        shutil.copy(src, f"{FIGD}/{f}.pdf")
    print(f"  figures/ {len(FIGS)}종 복사")


# ------------------------------------------------------------------ 표: 데이터
def dataset_stats():
    from PIL import Image
    sets = [("TN3K (trainval)", "tn3k_trainval"), ("TN3K (test)", "tn3k_test"), ("DDTI", "ddti")]
    stat = {}
    for name, d in sets:
        sizes, areas, counts = [], [], []
        for f in sorted(glob.glob(f"{ROOT}/datasets/{d}/images/*")):
            try:
                sizes.append(Image.open(f).size)
            except Exception:
                continue
            lbl = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            k = 0
            if os.path.exists(lbl):
                for line in open(lbl):
                    v = line.split()
                    if len(v) > 6:
                        k += 1
                        p = np.array([float(x) for x in v[1:]]).reshape(-1, 2)
                        x, y = p[:, 0], p[:, 1]
                        areas.append(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2 * 100)
            counts.append(k)
        s, a, c = np.array(sizes), np.array(areas), np.array(counts)
        ext = os.path.splitext(glob.glob(f"{ROOT}/datasets/{d}/images/*")[0])[1].lstrip(".").upper()
        stat[name] = dict(
            n=len(s), nod=int(c.sum()), per=c.mean(), multi=(c >= 2).mean() * 100,
            wmin=s[:, 0].min(), wmax=s[:, 0].max(), hmin=s[:, 1].min(), hmax=s[:, 1].max(),
            wmed=int(np.median(s[:, 0])), hmed=int(np.median(s[:, 1])),
            uniq=len(collections.Counter(map(tuple, s))), ext=ext,
            amed=np.median(a), a5=np.percentile(a, 5), a95=np.percentile(a, 95), amin=a.min())

    names = [n for n, _ in sets]
    rows = []
    def r(label, fn):
        rows.append(f"{label} & " + " & ".join(fn(stat[n]) for n in names))
    r("영상 수", lambda x: f"{x['n']:,}")
    r("결절 수", lambda x: f"{x['nod']:,}")
    r("영상당 평균 결절 수", lambda x: f"{x['per']:.2f}")
    r("결절 2개 이상 영상 비율 (\\%)", lambda x: f"{x['multi']:.1f}")
    r("파일 형식", lambda x: x["ext"])
    r("너비 범위 (px)", lambda x: f"{x['wmin']}--{x['wmax']}")
    r("높이 범위 (px)", lambda x: f"{x['hmin']}--{x['hmax']}")
    r("해상도 중앙값 (px)", lambda x: f"{x['wmed']}$\\times${x['hmed']}")
    r("고유 해상도 종류 수", lambda x: f"{x['uniq']:,}")
    r("결절 면적 중앙값 (영상 대비 \\%)", lambda x: f"{x['amed']:.2f}")
    r("결절 면적 5--95 백분위 (\\%)", lambda x: f"{x['a5']:.2f}--{x['a95']:.2f}")
    r("결절 면적 최솟값 (\\%)", lambda x: f"{x['amin']:.3f}")
    write("tab_dataset_stats", table(
        "사용한 두 공개 데이터셋의 통계. 결절 면적은 정답 폴리곤 면적을 영상 면적으로 나눈 값이다.",
        "tab:dataset_stats", "lrrr",
        "항목 & TN3K (trainval) & TN3K (test) & DDTI", rows))
    return stat


def split_table():
    def src(paths):
        t = sum("/tn3k" in p for p in paths)
        return t, len(paths) - t
    rows = []
    for f in FOLDS:
        d = f"{ROOT}/datasets/splits/pooled_5fold/fold{f}"
        tr = [l.strip() for l in open(f"{d}/train.txt") if l.strip()]
        va = [l.strip() for l in open(f"{d}/val.txt") if l.strip()]
        (tt, td), (vt, vd) = src(tr), src(va)
        rows.append(f"fold {f} & {len(tr):,} & {tt:,} / {td:,} & {len(va):,} & {vt:,} / {vd:,} & 614")
    rows.append("\\midrule")
    for name, slug, tgt in [("LODO TN3K$\\rightarrow$DDTI", "tn3k2ddti", "DDTI"),
                            ("LODO DDTI$\\rightarrow$TN3K", "ddti2tn3k", "TN3K")]:
        d = f"{ROOT}/datasets/splits/lodo/{slug}"
        tr = [l.strip() for l in open(f"{d}/train.txt") if l.strip()]
        va = [l.strip() for l in open(f"{d}/val.txt") if l.strip()]
        (tt, td), (vt, vd) = src(tr), src(va)
        rows.append(f"{name} & {len(tr):,} & {tt:,} / {td:,} & {len(va):,} & {vt:,} / {vd:,} & "
                    f"(= 검증, {tgt})")
    write("tab_splits", table(
        "데이터 분할 구성. 괄호 안은 TN3K / DDTI 영상 수이다. pooled 5-fold의 평가는 모두 "
        "학습에 쓰이지 않은 TN3K test 614장에서 수행하였다.",
        "tab:splits", "lrrrrr",
        "분할 & 학습 & (TN3K / DDTI) & 검증 & (TN3K / DDTI) & 평가", rows))


# ------------------------------------------------------------------ 표: 실험 설정
def hparam_table():
    import yaml
    a = yaml.safe_load(open(f"{ROOT}/runs/phase3_brm_final/nbrm_fold0/args.yaml"))
    b = yaml.safe_load(open(f"{ROOT}/runs/phase2_baseline/n_fold0/args.yaml"))
    keys = ["epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum",
            "weight_decay", "warmup_epochs", "cos_lr", "close_mosaic", "amp", "box", "cls", "dfl",
            "overlap_mask", "mask_ratio", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
            "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
            "seed", "deterministic"]
    diff = [k for k in keys if a.get(k) != b.get(k)]
    assert not diff, f"베이스라인과 제안 모델의 학습 설정이 다름: {diff}"
    groups = [
        ("최적화", ["epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum",
                 "weight_decay", "warmup_epochs", "cos_lr", "amp"]),
        ("손실 가중치", ["box", "cls", "dfl", "overlap_mask", "mask_ratio"]),
        ("데이터 증강", ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                    "perspective", "flipud", "fliplr", "mosaic", "close_mosaic", "mixup",
                    "copy_paste"]),
        ("재현성", ["seed", "deterministic"]),
    ]
    rows = []
    for g, ks in groups:
        cells = ", ".join(f"\\texttt{{{k.replace('_', chr(92) + '_')}}}={a[k]}" for k in ks)
        rows.append(f"{g} & \\parbox[t]{{0.72\\linewidth}}{{\\raggedright {cells}}}")
    rows.append(f"경계 손실 (제안) & \\texttt{{head\\_gain}}=1.0, $\\lambda$=2.0 (\\ref{{sec:sweep}}절 스윕으로 결정)")
    write("tab_hparams", table(
        "학습 설정. 베이스라인과 제안 모델은 경계 손실 계수를 제외한 모든 값이 같다"
        "(생성 시 두 모델의 \\texttt{args.yaml}이 일치하는지 자동 확인함).",
        "tab:hparams", "lp{0.74\\linewidth}", "구분 & 설정값", rows))


def env_table():
    import platform
    import subprocess
    vers = {}
    for mod in ["torch", "ultralytics", "onnxruntime", "openvino", "nncf", "numpy", "scipy", "cv2"]:
        try:
            m = __import__(mod)
            vers[mod] = m.__version__.split("-")[0]
        except Exception:
            vers[mod] = "--"
    try:
        import torch
        cuda = torch.version.cuda
    except Exception:
        cuda = "--"
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return "--"
    cpu = sh("lscpu | grep 'Model name' | sed 's/.*: *//'")
    ncpu = sh("nproc --all")
    gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1")
    ngpu = sh("nvidia-smi -L | wc -l")
    rows = [
        f"GPU & {gpu.replace('_', ' ')} $\\times$ {ngpu}",
        f"CPU & {cpu} ({ncpu} 논리 코어)",
        f"Python / PyTorch / CUDA & {platform.python_version()} / {vers['torch']} / {cuda}",
        f"Ultralytics & {vers['ultralytics']} (커스텀 패치 적용)",
        f"ONNX Runtime & {vers['onnxruntime']}",
        f"OpenVINO / NNCF & {vers['openvino']} / {vers['nncf']}",
        f"NumPy / SciPy / OpenCV & {vers['numpy']} / {vers['scipy']} / {vers['cv2']}",
    ]
    write("tab_env", table("실험 환경.", "tab:env", "ll", "항목 & 사양", rows))


# ------------------------------------------------------------------ 표: Phase 2
def nano_small_table():
    keys = [("metrics/mAP50-95(M)", "Mask mAP$_{50:95}$"), ("metrics/mAP50(M)", "Mask mAP$_{50}$"),
            ("metrics/mAP50-95(B)", "Box mAP$_{50:95}$"), ("metrics/mAP50(B)", "Box mAP$_{50}$"),
            ("metrics/precision(M)", "Mask 정밀도"), ("metrics/recall(M)", "Mask 재현율")]
    res = {}
    for m in ("n", "s"):
        js = [load_json(f"{ROOT}/runs/phase2_baseline/{m}_fold{f}_test/test_metrics.json") for f in FOLDS]
        res[m] = {k: [j[k] for j in js] for k, _ in keys}
    comp = {(r["split"], r["metric"]): r for r in read_csv(f"{ROOT}/runs/phase2_baseline/comparison.csv")}
    rows = []
    for k, lab in keys:
        n, s = np.mean(res["n"][k]), np.mean(res["s"][k])
        bn, bs = ("\\textbf{%s}" % ms(res["n"][k]), ms(res["s"][k])) if n >= s else \
                 (ms(res["n"][k]), "\\textbf{%s}" % ms(res["s"][k]))
        rows.append(f"{lab} & {bn} & {bs}")
    rows.append("\\midrule")
    for key, lab, d, unit in [("params_m", "파라미터", 2, " M"), ("flops", "GFLOPs", 2, ""),
                              ("size_mb", "가중치 파일 크기", 2, " MB")]:
        n = float(comp[("efficiency", key)]["nano"])
        s = float(comp[("efficiency", key)]["small"])
        rows.append(f"{lab} & \\textbf{{{n:.{d}f}{unit}}} & {s:.{d}f}{unit} ({s / n:.1f}배)")
    write("tab_nano_small", table(
        "YOLOv11-seg nano와 small의 비교 (held-out TN3K test, 5-fold 평균 $\\pm$ 표준편차). "
        "굵은 글씨는 더 나은 값이다.",
        "tab:nano_small", "lcc", "지표 & YOLOv11n-seg (nano) & YOLOv11s-seg (small)", rows))
    return res


# ------------------------------------------------------------------ 표: Phase 3
def sweep_table(split="val"):
    """split=val: 각 fold 검증 분할 기준(계수 선택의 근거, 본문). split=test: held-out 기준(부록 참고용)."""
    p = f"{ROOT}/runs/phase3_sweep/sweep_table{'_val' if split == 'val' else ''}.csv"
    if not os.path.exists(p):
        print(f"  [건너뜀] {os.path.basename(p)} 없음 — scripts/sweep_eval.py --split {split} 먼저 실행")
        return None
    rows_in = read_csv(p)
    base = next(r for r in rows_in if r["config"] == "baseline")
    # 굵게 표시하는 행 = 실제로 채택한 구성(g=1.0, λ=2.0). 표가 '무엇이 최선인지'를 판정하지 않고
    # 채택 구성을 보여 주며, 그것이 최선인지는 본문이 숫자를 보고 서술한다.
    best = next(r for r in rows_in if r["head_gain"] and float(r["head_gain"]) == 1.0
                and float(r["lam"]) == 2.0)
    rows = []
    for r in rows_in:
        is_base = r["config"] == "baseline"
        g = "--" if is_base else f"{float(r['head_gain']):g}"
        l = "--" if is_base else f"{float(r['lam']):g}"
        name = "베이스라인" if is_base else ("초기값" if r["config"].startswith("initial") else "스윕")
        hd = float(r["hd95"])
        dh = "" if is_base else f" ({(hd / float(base['hd95']) - 1) * 100:+.1f}\\%)"
        cells = [name, g, l, f"{float(r['dice']):.4f}", f"{hd:.3f}{dh}", f"{float(r['biou']):.4f}",
                 f"{float(r['map']):.4f}"]
        if r is best:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        rows.append(" & ".join(cells))
        if is_base or r["config"].startswith("initial"):
            rows.append("\\midrule")
    where = ("각 fold의 검증 분할(fold 0: 704장, fold 2: 703장)" if split == "val"
             else "held-out TN3K test 614장")
    # 실제 경위: 스윕 당시에는 held-out(test) 결과로 계수를 골랐고, 검증 분할 평가는 그 뒤의 사후 점검이다.
    use = ("계수 선택 이후에 같은 모델을 다시 평가한 사후 점검 결과이다." if split == "val"
           else "스윕 당시 이 표를 근거로 계수를 선택하였다(\\secref{sec:sweep}의 한계 참조).")
    write(f"tab_sweep_{split}", table(
        f"경계 손실 계수 탐색 결과 (fold 0·2 평균, {where}). $g$는 보조 경계 손실 가중치"
        "(\\texttt{head\\_gain}), $\\lambda$는 경계 밴드 강조 계수이며, 굵은 글씨가 채택한 구성이다. " + use,
        f"tab:sweep_{split}", "lcccccc",
        "구분 & $g$ & $\\lambda$ & Dice$\\uparrow$ & HD95 (\\%)$\\downarrow$ & BIoU$\\uparrow$ & "
        "Mask mAP$_{50:95}\\uparrow$", rows))
    return rows_in


def epochs_table():
    rows = []
    for tag, lab, pre in [("n", "베이스라인 nano", f"{ROOT}/runs/phase2_baseline/n_fold"),
                          ("s", "small", f"{ROOT}/runs/phase2_baseline/s_fold"),
                          ("b", "제안 (BRM nano)", f"{ROOT}/runs/phase3_brm_final/nbrm_fold")]:
        cells = [lab]
        for f in FOLDS:
            _, ep, ne = best_epoch_metrics(f"{pre}{f}")
            cells.append(f"{ne} / {ep}")
        rows.append(" & ".join(cells))
    write("tab_epochs", table(
        "fold별 학습 epoch (총 학습 epoch / 체크포인트 epoch). 최대 100 epoch, 검증 fitness가 "
        "20 epoch 동안 개선되지 않으면 조기 종료하였으므로, 총 epoch이 100보다 작은 경우 체크포인트 epoch보다 정확히 20 크다.",
        "tab:epochs", "lccccc", "모델 & fold 0 & fold 1 & fold 2 & fold 3 & fold 4", rows))


def brm_main_table(c):
    b, m = c["baseline nano (P2)"], c["BRM nano (P3)"]
    rows = []
    for key, lab, hb, d in [("map", "Mask mAP$_{50:95}\\uparrow$", True, 4), ("dice", "Dice$\\uparrow$", True, 4),
                            ("hd", "HD95 (\\%)$\\downarrow$", False, 3), ("biou", "Boundary IoU$\\uparrow$", True, 4)]:
        xb = np.array([r[key] for r in b]); xm = np.array([r[key] for r in m])
        delta = xm.mean() - xb.mean()
        rel = delta / xb.mean() * 100
        better = ((xm > xb) if hb else (xm < xb)).sum()
        rows.append(f"{lab} & {ms(xb, d)} & {ms(xm, d)} & {delta:+.{d}f} ({rel:+.1f}\\%) & {better}/5")
    rows.append("\\midrule")
    rows.append(f"파라미터 & {b[0]['params'] * 1e6:,.0f} & {m[0]['params'] * 1e6:,.0f} & "
                f"+{(m[0]['params'] - b[0]['params']) * 1e6:,.0f} & --")
    rows.append(f"추론 GFLOPs & {b[0]['flops']:.2f} & {m[0]['flops']:.2f} & "
                f"{m[0]['flops'] - b[0]['flops']:+.2f} & --")
    write("tab_brm_main", table(
        "경계 정교화 모듈의 효과 (held-out TN3K test 614장, 5-fold 평균 $\\pm$ 표준편차). "
        "마지막 열은 제안 모델이 베이스라인보다 나은 fold 수이다.",
        "tab:brm_main", "lcccc",
        "지표 & 베이스라인 & 제안 (BRM) & $\\Delta$ (상대) & 개선 fold", rows))


def perfold_table(c):
    b, m = c["baseline nano (P2)"], c["BRM nano (P3)"]
    rows = []
    for f in FOLDS:
        cells = [f"fold {f}"]
        for key, d in [("map", 4), ("dice", 4), ("hd", 3), ("biou", 4)]:
            cells += [f"{b[f][key]:.{d}f}", f"{m[f][key]:.{d}f}"]
        rows.append(" & ".join(cells))
    rows.append("\\midrule")
    cells = ["평균"]
    for key, d in [("map", 4), ("dice", 4), ("hd", 3), ("biou", 4)]:
        cells += [f"{np.mean([r[key] for r in b]):.{d}f}", f"{np.mean([r[key] for r in m]):.{d}f}"]
    rows.append(" & ".join(cells))
    hdr = ("\\multirow{2}{*}{fold} & \\multicolumn{2}{c}{Mask mAP$_{50:95}$} & "
           "\\multicolumn{2}{c}{Dice} & \\multicolumn{2}{c}{HD95 (\\%)} & \\multicolumn{2}{c}{BIoU} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
           " & 기준 & 제안 & 기준 & 제안 & 기준 & 제안 & 기준 & 제안")
    write("tab_perfold", table(
        "fold별 상세 결과 (held-out TN3K test). 기준 = 베이스라인 nano, 제안 = BRM nano.",
        "tab:perfold", "l" + "cc" * 4, hdr, rows, size="\\footnotesize"))


def stats_table():
    s = read_csv(f"{ROOT}/runs/stats/paired_stats.csv")
    idx = {(r["analysis"], r["metric"]): r for r in s}
    rows = []
    for ana, lab in [("A_image_level", "영상 단위 (n=614)"), ("A_fold_level", "fold 단위 (n=5)")]:
        for k, (met, mlab) in enumerate([("Dice", "Dice"), ("BoundaryIoU", "Boundary IoU"), ("HD95%", "HD95 (\\%)")]):
            r = idx[(ana, met)]
            ci = (f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}]" if r.get("ci_lo") else "--")
            first = f"\\multirow{{3}}{{*}}{{{lab}}}" if k == 0 else ""
            wp = float(r["w_p"])
            rows.append(f"{first} & {mlab} & {float(r['delta']):+.4f} & {ci} & ${pfmt(wp)}$ ({stars(wp)}) "
                        f"& {float(r['dz']):.2f} & {float(r['win_rate']) * 100:.1f}")
        if ana == "A_image_level":
            rows.append("\\midrule")
    write("tab_stats", table(
        "경계 정교화 효과의 대응표본 검정 (held-out TN3K test). 영상 단위 분석은 각 영상에 대해 "
        "5개 fold 모델의 예측 지표를 평균한 뒤 비교한다. 95\\% 신뢰구간은 대응 차이의 부트스트랩"
        "(10,000회) 백분위 구간이며, $d_z$는 대응표본 효과크기로 양수가 제안 모델의 우세를 뜻한다. "
        "n=5에서 양측 Wilcoxon 검정이 도달할 수 있는 최소 $p$는 0.0625이다. 대응 $t$ 검정의 $p$는 "
        + "영상 단위 " + ", ".join(f"{float(idx[('A_image_level', m)]['t_p']):.4f}" for m in ("Dice", "BoundaryIoU", "HD95%"))
        + ", fold 단위 " + ", ".join(f"{float(idx[('A_fold_level', m)]['t_p']):.3f}" for m in ("Dice", "BoundaryIoU", "HD95%"))
        + "(Dice, Boundary IoU, HD95 순)로 결론이 같다.",
        "tab:stats", "llccccc",
        "분석 단위 & 지표 & $\\Delta$ & 95\\% CI & Wilcoxon $p$ & $d_z$ & 개선 비율 (\\%)",
        rows, size="\\footnotesize", wide=True, colsep="4pt"))
    return idx


def lodo_tables(cache):
    dirs = [("TN3K$\\rightarrow$DDTI", "tn3k2ddti", 637), ("DDTI$\\rightarrow$TN3K", "ddti2tn3k", 3493)]
    def run_dir(model, slug, seed):
        if seed == 42:
            return f"{ROOT}/runs/phase2_baseline/n_{slug}" if model == "base" else \
                   f"{ROOT}/runs/phase3_brm_final/nbrm_{slug}"
        return f"{ROOT}/runs/phase5_lodo_seeds/{'n' if model == 'base' else 'nbrm'}_{slug}_s{seed}"
    def ckey(model, slug, seed):
        return f"B|{model}|{slug.upper()}" + ("" if seed == 42 else f"|s{seed}")

    stats = read_csv(f"{ROOT}/runs/stats/paired_stats.csv")
    sidx = {(r["analysis"], r["metric"]): r for r in stats}

    main_rows, seed_rows, out = [], [], {}
    for nice, slug, n in dirs:
        vals = {}
        for model in ("base", "brm"):
            per = []
            for s in SEEDS:
                row, ep, ne = best_epoch_metrics(run_dir(model, slug, s))
                img = cache[ckey(model, slug, s)].values()
                v = np.array(list(img))
                per.append(dict(map=float(row["metrics/mAP50-95(M)"]), dice=v[:, 2].mean(),
                                hd=v[:, 1].mean(), biou=v[:, 0].mean(), ep=ep, ne=ne, seed=s))
            vals[model] = per
        out[slug] = vals
        for k, (key, lab, d) in enumerate([("map", "Mask mAP$_{50:95}$", 4), ("dice", "Dice", 4),
                                           ("hd", "HD95 (\\%)", 3), ("biou", "Boundary IoU", 4)]):
            xb = [r[key] for r in vals["base"]]; xm = [r[key] for r in vals["brm"]]
            from scipy import stats as S
            tp = S.ttest_rel(xm, xb).pvalue
            first = f"\\multirow{{4}}{{*}}{{{nice}}}" if k == 0 else ""
            main_rows.append(f"{first} & {lab} & {ms(xb, d)} & {ms(xm, d)} & "
                             f"{np.mean(xm) - np.mean(xb):+.{d}f} & {tp:.3f} ({stars(tp)})")
        main_rows.append("\\midrule")
        for i, s in enumerate(SEEDS):
            b, m = vals["base"][i], vals["brm"][i]
            first = f"\\multirow{{3}}{{*}}{{{nice}}}" if i == 0 else ""
            seed_rows.append(f"{first} & {s} & {b['map']:.4f} / {m['map']:.4f} & "
                             f"{b['dice']:.4f} / {m['dice']:.4f} & {b['hd']:.2f} / {m['hd']:.2f} & "
                             f"{b['ep']} / {m['ep']} & {b['ne']} / {m['ne']}")
        seed_rows.append("\\midrule")
    main_rows.pop(); seed_rows.pop()
    write("tab_lodo", table(
        "교차 데이터셋(LODO) 결과 — 3개 시드(42, 43, 44) 평균 $\\pm$ 표준편차. Mask mAP는 "
        "저장된 체크포인트(검증 fitness 최대 epoch)의 값이며, 경계 지표는 타깃 도메인 전체 영상에 대한 "
        "평균이다. $p$는 시드 단위 대응 $t$ 검정(n=3)이다.",
        "tab:lodo", "llcccc",
        "방향 & 지표 & 베이스라인 & 제안 (BRM) & $\\Delta$ & 시드 단위 $p$", main_rows))
    write("tab_lodo_seed", table(
        "LODO 시드별 상세 결과 (베이스라인 / 제안). 체크포인트 epoch은 검증 fitness가 최대인 epoch, "
        "총 epoch은 조기 종료까지 학습한 epoch 수이다.",
        "tab:lodo_seed", "llccccc",
        "방향 & 시드 & Mask mAP$_{50:95}$ & Dice & HD95 (\\%) & 체크포인트 epoch & 총 epoch",
        seed_rows, size="\\footnotesize"))
    return out


# ------------------------------------------------------------------ 표: Phase 4
QROWS = [("BRM nano (P3)", "PyTorch FP32"), ("BRM + ONNX-FP32", "ONNX FP32"),
         ("BRM + ONNX-FP16", "ONNX FP16"), ("BRM + ONNX-INT8", "ONNX Runtime INT8"),
         ("BRM + OpenVINO-INT8", "OpenVINO INT8")]


def quant_table(c):
    def sd(k):
        v = [np.std([r[k] for r in c[key]], ddof=1) for key, _ in QROWS
             if all(r.get(k) is not None for r in c[key])]
        return min(v), max(v)
    (cmin, cmax), (_, gmax) = sd("cpu"), sd("gpu")
    rows = []
    for key, lab in QROWS:
        rs = c[key]
        def g(k, d):
            v = [r.get(k) for r in rs]
            return ms(v, d) if any(x is not None for x in v) else "--"
        def m_(k, d):
            v = mean([r.get(k) for r in rs])
            return f"{v:.{d}f}" if v is not None else "--"
        cells = [lab, m_("size", 2), m_("cpu", 2), m_("gpu", 2),
                 g("map", 4), g("dice", 4), g("hd", 3), g("biou", 4)]
        if "OpenVINO" in lab:
            cells = [f"\\textbf{{{x}}}" for x in cells]
        rows.append(" & ".join(cells))
    write("tab_quant", table(
        "정밀도별 경량화 결과 (제안 모델, held-out TN3K test, 5-fold 평균 $\\pm$ 표준편차). "
        "지연 시간은 batch 1, 640$\\times$640 입력, 워밍업 15회 후 100회 평균의 fold 평균이며 "
        f"fold 간 표준편차는 CPU {cmin:.2f}--{cmax:.2f} ms, GPU {gmax:.2f} ms 이하였다. "
        "INT8 모델의 정확도는 CPU에서 측정했다.",
        "tab:quant", "lccccccc",
        "정밀도 & 크기 (MB) & CPU (ms) & GPU (ms) & Mask mAP$_{50:95}$ & Dice & HD95 (\\%) & BIoU",
        rows, size="\\footnotesize", wide=True, colsep="3.5pt"))


def final_table(c):
    b = c["baseline nano (P2)"]
    order = [("baseline nano (P2)", "베이스라인 nano"), ("BRM nano (P3)", "+ 경계 정교화"),
             ("BRM + ONNX-FP16", "+ ONNX FP16"), ("BRM + ONNX-INT8", "+ ONNX Runtime INT8"),
             ("BRM + OpenVINO-INT8", "+ OpenVINO INT8 (최종)")]
    rows = []
    for key, lab in order:
        rs = c[key]
        cpu = mean([r.get("cpu") for r in rs]); gpu = mean([r.get("gpu") for r in rs])
        cells = [lab, f"{mean([r['map'] for r in rs]):.4f}", f"{mean([r['dice'] for r in rs]):.4f}",
                 f"{mean([r['hd'] for r in rs]):.3f}", f"{mean([r['biou'] for r in rs]):.4f}",
                 f"{mean([r['size'] for r in rs]):.2f}",
                 f"{cpu:.2f}" if cpu else "--", f"{gpu:.2f}" if gpu else "--"]
        if "최종" in lab:
            cells = [f"\\textbf{{{x}}}" for x in cells]
        rows.append(" & ".join(cells))
    write("tab_final", table(
        "단계별 누적 결과 (held-out TN3K test, 5-fold 평균). PyTorch 모델의 지연 시간은 "
        "배포 형식과 측정 조건이 달라 비교에서 제외했다.",
        "tab:final", "lccccccc",
        "모델 & Mask mAP$_{50:95}$ & Dice & HD95 (\\%) & BIoU & 크기 (MB) & CPU (ms) & GPU (ms)",
        rows, size="\\footnotesize", wide=True))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="thesis_source.zip 도 만든다")
    args = ap.parse_args()
    print("그림 복사")
    copy_figures()
    print("표 생성")
    c = load_json(f"{ROOT}/runs/phase4_quant/final_table_cache.json")
    cache = load_json(f"{ROOT}/runs/stats/per_image_cache.json")
    dataset_stats(); split_table(); hparam_table(); env_table()
    nano_small_table(); sweep_table("val"); sweep_table("test"); epochs_table()
    brm_main_table(c); perfold_table(c); stats_table()
    lodo_tables(cache); quant_table(c); final_table(c)
    if args.zip:
        import zipfile
        out = f"{ROOT}/thesis_source.zip"
        skip = (".log", ".aux", ".bbl", ".blg", ".out", ".toc", ".lof", ".lot", ".synctex.gz", ".fls",
                ".fdb_latexmk", ".xdv")
        n = 0
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for d, _, fs in os.walk(TH):
                for f in sorted(fs):
                    rel = os.path.relpath(os.path.join(d, f), ROOT)
                    # 빌드 산출물과 미리보기 PDF(thesis/main.pdf)는 넣지 않는다 — 업로드한 곳에서 다시 컴파일한다
                    if f.endswith(skip) or rel == "thesis/main.pdf":
                        continue
                    z.write(os.path.join(d, f), rel)
                    n += 1
        print(f"압축: {out} ({n}개 파일, {os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
