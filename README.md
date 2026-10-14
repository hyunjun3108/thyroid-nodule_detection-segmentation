# 갑상선 결절 초음파 영상의 경량 검출·분할 모델

추론 비용 없는 경계 정교화 모듈(SegmentBRM)과 INT8 양자화를 중심으로 한 석사학위 연구 코드입니다.
YOLOv11-seg를 기반으로 ① 두 공개 데이터셋을 혼합해 학습하고, ② 학습 전용 경계 분기와 경계 가중 손실로
경계 품질을 높이고, ③ 사후 양자화로 CPU 배포가 가능한 수준까지 경량화합니다.

## 주요 결과 (held-out TN3K test 614장, 5-fold 평균)

| 모델 | Mask mAP50-95 | Dice | HD95 (%) | Boundary IoU | GFLOPs | 크기 | CPU 지연 |
|---|---|---|---|---|---|---|---|
| 베이스라인 YOLOv11n-seg | 0.6272 | 0.8195 | 9.628 | 0.4358 | 9.73 | 6.00 MB | – |
| + 경계 정교화 (SegmentBRM) | 0.6267 | **0.8300** | **8.378** | **0.4414** | 9.73 | 6.01 MB | – |
| + ONNX FP32 | 0.6188 | 0.8300 | 8.529 | 0.4402 | – | 11.63 MB | 62.82 ms |
| + OpenVINO INT8 (최종) | 0.6161 | 0.8290 | 8.497 | 0.4379 | – | **3.71 MB** | **4.10 ms** |

- 경계 정교화: Dice +0.0104, HD95 −13.0%, 추론 연산량 증가 없음. 이미지 단위 대응표본 검정에서
  세 지표 모두 유의(Wilcoxon, n=614 — Dice p=0.014, HD95 p=0.0045, Boundary IoU p=0.011).
- 경량화: OpenVINO INT8로 FP32 대비 CPU 15.3배 가속, Dice 손실 0.001 수준.
- 교차 데이터셋(LODO): 개선이 유지되지 않으며, 3시드 반복 결과 악화 역시 유의하지 않다(전 지표 ns).

## 저장소 구성

```
scripts/                 학습·평가·그림·발표자료 생성 스크립트 전부
ultralytics_patches/     커스텀 ultralytics 변경분(패치) — SegmentBRM 헤드, 경계 손실 등
datasets/splits/         5-fold / LODO / held-out 분할 목록 (이미지 자체는 미포함)
runs/figures/            논문 그림 (PNG 300dpi + 벡터 PDF)
runs/**/                 학습 곡선(results.csv), 평가 지표(json/csv) — 가중치는 미포함
runs/stats/              통계 검정 결과 (paired_stats.csv)
*.pptx, *.docx           발표자료, 연구계획서
```

가중치(`*.pt`), ONNX/OpenVINO 산출물, 원본 데이터셋은 용량과 재배포 문제로 올리지 않습니다.

## 환경 구성

```bash
# 1) 기반 ultralytics 체크아웃 후 패치 적용
git clone https://github.com/ultralytics/ultralytics.git ultralytics_thyroid
cd ultralytics_thyroid
git checkout 4dcc98a6be09d058c206f2756f3ee2966d1c4b71
git apply ../ultralytics_patches/thyroid.patch
pip install -e .
cd ..

# 2) 부가 의존성
pip install onnx onnxruntime-gpu openvino nncf scipy python-pptx python-docx
```

## 데이터 준비

TN3K(Gong et al., 2021·2023)와 DDTI(Pedraza et al., 2015)를 각 배포처에서 내려받아
`datasets_raw/` 아래에 둔 뒤 변환합니다.

```bash
python scripts/convert_to_yoloseg.py   # 마스크 → YOLO-seg 폴리곤, datasets/ 생성
python scripts/make_splits.py          # 5-fold / LODO / held-out 분할 (seed 42)
```

## 실행 순서

| 단계 | 명령 |
|---|---|
| Phase 2 베이스라인 | `bash scripts/run_baseline_rest.sh` |
| Phase 3 경계 정교화 | `bash scripts/run_phase3_final.sh` (gain 탐색은 `run_phase3_sweep.sh`) |
| 경계 지표 평가 | `python scripts/eval_boundary.py --runs baseline=runs/phase2_baseline/n BRM=runs/phase3_brm_final/nbrm` |
| Phase 4 양자화 | `python scripts/export_quant.py --fold N` → `python scripts/eval_quant.py --folds 0,1,2,3,4` |
| 통합 결과표 | `python scripts/final_table.py` |
| LODO 시드 반복 | `bash scripts/run_lodo_seeds.sh` |
| 통계 검정 | `python scripts/stats_test.py --part all` |
| 그림 | `python scripts/make_figures.py --only 1,2,3,4,10,11,12,13` · `make_arch_figures.py` · `make_stats_figure.py` |
| 발표자료 | `python scripts/make_ppt.py` |

학습 스크립트는 `exist_ok=True`로 동작하므로 같은 실행을 두 번 띄우면 결과가 섞입니다.
재실행 전 `pgrep -af train_seg.py`로 기존 실행이 없는지 확인하고, 드라이버 로그의 `ALL DONE`을 확인한 뒤 평가하세요.
