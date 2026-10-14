# ultralytics 변경분

`thyroid.patch` 하나에 이 연구에서 ultralytics에 가한 변경이 전부 들어 있습니다.

- 기반 커밋: `4dcc98a6be09d058c206f2756f3ee2966d1c4b71` (ultralytics 8.3.181 계열)
- 적용: 해당 커밋을 체크아웃한 뒤 `git apply thyroid.patch`

| 파일 | 내용 |
|---|---|
| `ultralytics/nn/modules/head.py` | `SegmentBRM` — Proto 출력에서 갈라지는 학습 전용 경계 분기 |
| `ultralytics/utils/loss.py` | `v8SegmentationBoundaryLoss` — 경계 가중 마스크 BCE + 보조 경계 손실 |
| `ultralytics/nn/tasks.py` | `SegmentBRM` 파싱 등록, 손실 함수 선택 |
| `ultralytics/nn/modules/__init__.py` | 모듈 export |
| `ultralytics/cfg/models/11/yolo11-seg-brm.yaml` | 제안 모델 구성 |
| `ultralytics/nn/modules/lska.py`, `cfg/models/11/yolov11-lska.yaml`, `utils/loss(focal).py` | 포크에 있던 별도 실험 파일(본 연구 결과에는 사용하지 않음) |

`loss.py`의 diff가 길어 보이는 것은 포크에서 기존 클래스 docstring에 빈 줄이 삽입되었기 때문입니다.
docstring을 제외하고 AST를 비교하면 기존 클래스(`v8SegmentationLoss` 등)의 동작 변경은 없으며,
실질적으로 추가된 것은 `v8SegmentationBoundaryLoss` 하나입니다.

경계 손실 계수는 환경변수로 바꿀 수 있습니다: `BRM_HEAD_GAIN`(기본 1.0), `BRM_LAMBDA`(기본 2.0).

`scripts/auto_sync.sh`가 동기화할 때마다 이 패치를 포크의 현재 상태로 다시 생성하므로,
`ultralytics_thyroid/`를 수정해도 별도 작업 없이 반영됩니다.
