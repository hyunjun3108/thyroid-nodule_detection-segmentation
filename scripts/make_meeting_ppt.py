#!/usr/bin/env python
"""
연구 제안 요약 발표자료(.pptx) 생성. 학위논문 제안서의 논문개요를 기준으로,
실험 과정 전체가 아니라 '무엇을, 왜, 어떻게 하려는지'와 예비 결과만 짧게 담는다.

서식·도우미 함수는 make_ppt.py 것을 그대로 쓴다. 각 슬라이드의 발표 대본은 슬라이드 노트에 들어간다.

사용:
  /data/my_envs/thyroid_seg/bin/python scripts/make_meeting_ppt.py
출력: /data/HJ_thyroid/갑상선결절_연구제안_요약.pptx
"""
import io
import os
import sys

from PIL import Image
from pptx.util import Inches as In, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ppt import (Deck, textbox, para, rect, picture, table, FIG, ROOT, W, H, ML, MR,
                      INK, MUTED, FAINT, BLUE, ORANGE, RULE, PANEL, WHITE, TITLE, SUBTITLE)

from pptx.dml.color import RGBColor

OUT = f"{ROOT}/갑상선결절_연구제안_요약.pptx"
RGBGreen = RGBColor(0x1E, 0x9E, 0x4A)           # 정성 비교 그림의 정답 윤곽 색
CW = W - ML - MR                     # 본문 폭
DATE = "2026. 11. 03."


# ----------------------------------------------------------------- 도우미
def notes(slide, text):
    """발표 대본을 슬라이드 노트에 넣는다. 빈 줄로 문단을 나눈다."""
    slide.notes_slide.notes_text_frame.text = text.strip()


def box(slide, x, y, w, h, text, sub=None, fill=PANEL, line=None, color=INK, size=13,
        bold=True, dash=False):
    """가운데 정렬 글상자가 든 사각형(구조도용)."""
    sh = rect(slide, x, y, w, h, fill=fill, line=line, lw=1.25)
    if dash and line is not None:
        ln = sh.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
    tf = textbox(slide, x + 0.08, y, w - 0.16, h, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, text, size, bold=bold, color=color, align=PP_ALIGN.CENTER, first=True,
         space_after=0, line=1.15)
    if sub:
        para(tf, sub, size - 2.5, color=MUTED, align=PP_ALIGN.CENTER, space_before=2,
             space_after=0, line=1.15)
    return sh


def arrow(slide, x1, y1, x2, y2, color=FAINT):
    c = slide.shapes.add_connector(1, In(x1), In(y1), In(x2), In(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def crop_png(path, box_frac):
    """그림의 일부(비율 좌표 l,t,r,b)만 잘라 메모리 PNG로 돌려준다. 원본 파일은 건드리지 않는다."""
    im = Image.open(path)
    w, h = im.size
    l, t, r, b = box_frac
    buf = io.BytesIO()
    im.crop((int(l * w), int(t * h), int(r * w), int(b * h))).save(buf, "PNG")
    buf.seek(0)
    return buf


def picture_buf(slide, buf, x, y, w, h):
    iw, ih = Image.open(buf).size
    buf.seek(0)
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    return slide.shapes.add_picture(buf, In(x + (w - pw) / 2), In(y + (h - ph) / 2),
                                    In(pw), In(ph))


# ----------------------------------------------------------------- 슬라이드
def s_title(d):
    s = d._new()
    rect(s, 0, 0, 0.09, H, fill=ORANGE)
    rect(s, ML, 2.32, 1.5, 0.028, fill=INK)
    tf = textbox(s, ML, 1.72, 10.4, 0.4)
    para(tf, "학위논문 연구 제안", 12.5, bold=True, color=MUTED, first=True, space_after=0)
    tf = textbox(s, ML, 2.62, 11.6, 1.4)
    para(tf, TITLE, 36, bold=True, first=True, space_after=10, line=1.18)
    para(tf, SUBTITLE, 17, color=MUTED, line=1.3)
    tf = textbox(s, ML, 5.5, 11.2, 0.9)
    para(tf, "○○○", 13.5, first=True, space_after=4)
    para(tf, DATE, 12, color=MUTED)
    d.n += 1
    notes(s, """
오늘은 학위논문 제안서에 적은 연구 내용을 간단히 말씀드리겠습니다.
주제는 갑상선 결절 초음파 영상에서 결절을 찾고 경계를 그리는 모델을, 경계는 더 정확하게, 모델은 더 가볍게 만드는 것입니다.
""")


def s_onepage(d):
    s, top = d.content("연구 요약", "한 문장으로 말씀드리면")
    tf = textbox(s, ML, top + 0.15, CW, 1.0)
    para(tf, "초음파 영상에서 갑상선 결절의 경계를 더 정확하게 그리면서,",
         21, bold=True, first=True, space_after=2, line=1.3)
    para(tf, "GPU 없이 일반 CPU에서도 빠르게 동작하는 검출·분할 모델을 만드는 것",
         21, bold=True, color=ORANGE, space_after=0, line=1.3)

    items = [
        ("01", "데이터", "공개 데이터셋 두 개(TN3K, DDTI)를 합쳐 4,130장으로 학습하고, "
                       "학습에 쓰지 않은 영상으로 평가한다"),
        ("02", "경계 정교화", "학습할 때만 경계를 따로 가르치고, 실제로 쓰는 모델에는 "
                          "아무것도 더하지 않는다"),
        ("03", "경량화", "학습이 끝난 모델을 8비트 정수(INT8)로 바꿔 크기와 CPU 처리 시간을 "
                       "줄인다"),
    ]
    gap = 0.5
    bw = (CW - 2 * gap) / 3
    y = top + 1.75
    for i, (num, t, b) in enumerate(items):
        x = ML + i * (bw + gap)
        rect(s, x, y, bw, 2.55, fill=PANEL)
        rect(s, x, y, bw, 0.045, fill=ORANGE if i == 1 else BLUE)
        tf = textbox(s, x + 0.3, y + 0.32, bw - 0.6, 2.1)
        para(tf, num, 13, bold=True, color=ORANGE if i == 1 else BLUE, first=True,
             space_after=6)
        para(tf, t, 18, bold=True, space_after=10)
        para(tf, b, 13, color=MUTED, space_after=0, line=1.4)
        if i < 2:
            arrow(s, x + bw + 0.08, y + 1.27, x + bw + gap - 0.08, y + 1.27)
    notes(s, """
연구를 한 문장으로 정리하면, 결절 경계를 더 정확하게 그리면서 GPU 없이 CPU에서도 빠르게 도는 모델을 만드는 것입니다.

이를 위해 세 단계로 진행합니다.
첫째, 공개 데이터셋 두 개를 합쳐 학습하고 평가합니다.
둘째, 학습할 때만 경계를 따로 가르치는 장치를 붙여서, 실제로 쓰는 모델의 계산량은 그대로 두고 경계만 좋아지게 합니다.
셋째, 학습이 끝난 모델을 8비트 정수로 바꿔서 CPU에서 빠르게 돌아가게 합니다.
""")


def s_why(d):
    s, top = d.content("연구 배경", "왜 '경계'가 중요한가",
                       foot="TI-RADS: Thyroid Imaging Reporting and Data System (ACR, 2017)")
    tf = textbox(s, ML, top + 0.2, 5.5, 4.6)
    for i, (t, b) in enumerate([
        ("결절은 흔하고, 대부분은 양성이다",
         "그래서 검사할 때마다 결절의 모양을 보고 악성 가능성을 따져야 한다."),
        ("경계는 판독 점수에 직접 들어간다",
         "ACR TI-RADS에서 경계(매끄러움, 분엽상, 불규칙)는 별도 채점 항목이고, "
         "이 점수가 세침흡인검사 여부를 정하는 데 쓰인다."),
        ("그런데 초음파에서는 경계 자체가 흐리다",
         "스페클 잡음과 낮은 대비 때문에 경계가 모호하고, 판독자마다 다르게 그리기도 한다. "
         "자동 분할 모델도 영역은 대체로 맞혀도 경계에서 많이 틀린다."),
    ]):
        para(tf, t, 15, bold=True, first=(i == 0), space_before=0 if i == 0 else 12,
             space_after=5)
        para(tf, b, 13, color=MUTED, space_after=0, line=1.4)

    x = 6.55
    picture_buf(s, crop_png(f"{FIG}/fig10_dataset.png", (0, 0.11, 1, 1)),
                x, top + 0.25, W - MR - x, 3.4)
    tf = textbox(s, x, top + 3.8, W - MR - x, 0.6)
    para(tf, "실제 데이터 예시. 위는 TN3K, 아래는 DDTI이며 초록 선이 전문가가 그린 결절 경계다. "
             "두 데이터셋은 장비와 화면 구성이 서로 다르다.",
         11, color=MUTED, first=True, space_after=0, line=1.35)
    notes(s, """
경계에 집중하는 이유입니다.
갑상선 결절은 아주 흔하고 대부분 양성이라, 결국 모양을 보고 악성 가능성을 판단하게 됩니다.
ACR TI-RADS에서는 결절의 경계가 매끄러운지, 분엽상인지, 불규칙한지를 따로 점수로 매기고, 이 점수가 세침흡인검사를 할지 말지에 반영됩니다.

그래서 자동 분할 모델이 영역만 대략 맞히는 것으로는 부족하고, 경계를 정확하게 그려야 임상에서 의미가 있다고 생각했습니다.
오른쪽은 이번에 사용한 실제 데이터이고, 초록 선이 정답 경계입니다.
""")


def s_problem(d):
    s, top = d.content("문제와 접근", "해결하려는 세 가지 문제")
    items = [
        ("문제 1", "경계 오차가 잘 드러나지 않는다",
         "빠른 분할 모델(YOLO 계열)은 마스크를 낮은 해상도로 만들어 경계가 거칠다. "
         "흔히 쓰는 지표(mAP)는 몇 픽셀의 경계 오차를 거의 반영하지 못한다.",
         "경계만 따로 가르치는 학습 신호를 추가하고, 경계 지표로 평가한다", ORANGE),
        ("문제 2", "현장에는 GPU가 없다",
         "판독 PC나 휴대형 초음파 장비는 GPU 없이 CPU로 동작하는 경우가 많다. "
         "정확한 모델이라도 느리면 현장에서 쓰기 어렵다.",
         "작은 모델(nano)에서 출발하고, INT8로 변환해 CPU 속도를 높인다", BLUE),
        ("문제 3", "데이터셋마다 영상이 다르다",
         "공개 데이터셋은 장비, 화면 구성, 주석 방식이 제각각이다. "
         "데이터셋 하나의 결과만으로는 다른 병원 영상에서도 통할지 알 수 없다.",
         "두 데이터셋을 함께 학습하고, 서로 바꿔 가며 교차 평가한다", BLUE),
    ]
    gap = 0.25
    cw = (CW - 2 * gap) / 3
    for i, (tag, t, body, how, col) in enumerate(items):
        x = ML + i * (cw + gap)
        rect(s, x, top + 0.2, cw, 4.55, fill=PANEL)
        rect(s, x, top + 0.2, cw, 0.045, fill=col)
        tf = textbox(s, x + 0.3, top + 0.5, cw - 0.6, 2.6)
        para(tf, tag, 12, bold=True, color=col, first=True, space_after=6)
        para(tf, t, 16.5, bold=True, space_after=10, line=1.25)
        para(tf, body, 12.5, color=MUTED, space_after=0, line=1.4)
        rect(s, x + 0.3, top + 3.35, cw - 0.6, 0.012, fill=RULE)
        tf = textbox(s, x + 0.3, top + 3.5, cw - 0.6, 1.1)
        para(tf, "접근", 11.5, bold=True, color=col, first=True, space_after=4)
        para(tf, how, 13, bold=True, space_after=0, line=1.35)
    notes(s, """
구체적으로는 세 가지 문제를 다룹니다.

첫째, 경계 오차가 잘 안 보인다는 점입니다. YOLO처럼 빠른 분할 모델은 마스크를 낮은 해상도로 만들기 때문에 경계가 거칠고, 보통 쓰는 mAP 같은 지표로는 경계가 몇 픽셀 틀린 것이 잘 드러나지 않습니다. 그래서 경계를 따로 가르치는 학습 신호를 넣고, 경계 전용 지표로 평가합니다.

둘째, 현장에는 GPU가 없는 경우가 많습니다. 그래서 작은 모델에서 시작하고, 8비트 정수로 변환해 CPU 속도를 높입니다.

셋째, 공개 데이터셋끼리도 영상이 많이 다릅니다. 그래서 두 데이터셋을 함께 쓰고, 한쪽으로 학습해 다른 쪽에서 평가하는 교차 실험도 합니다.
""")


def s_questions(d):
    s, top = d.content("연구 질문", "이 연구에서 확인하려는 것",
                       "제안서의 세 가설을 질문 형태로 정리했다")
    items = [
        ("Q1", "작은 모델로도 충분한가?",
         "YOLOv11 nano와 small을 같은 조건으로 학습해 정확도를 비교한다"),
        ("Q2", "추론 비용을 늘리지 않고 경계를 개선할 수 있는가?",
         "경계 정교화 전후의 Dice, Boundary IoU, HD95를 비교하고, 연산량이 같은지 확인한다"),
        ("Q3", "8비트로 줄여도 정확도가 유지되는가?",
         "정밀도별(FP32, FP16, INT8)로 정확도와 CPU 처리 시간을 함께 비교한다"),
    ]
    y = top + 0.25
    for tag, q, how in items:
        col = ORANGE if tag == "Q2" else BLUE
        rect(s, ML, y, CW, 1.35, fill=WHITE, line=RULE)
        rect(s, ML, y, 0.05, 1.35, fill=col)
        tf = textbox(s, ML + 0.35, y, 1.0, 1.35, anchor=MSO_ANCHOR.MIDDLE)
        para(tf, tag, 22, bold=True, color=col, first=True, space_after=0)
        tf = textbox(s, ML + 1.5, y, CW - 1.8, 1.35, anchor=MSO_ANCHOR.MIDDLE)
        para(tf, q, 17.5, bold=True, first=True, space_after=6)
        para(tf, "확인 방법   " + how, 12.5, color=MUTED, space_after=0)
        y += 1.55
    notes(s, """
그래서 이 연구에서 확인하려는 것은 세 가지입니다.

첫째, 가장 작은 nano 모델로도 충분한지.
둘째, 실제 사용하는 모델의 계산량을 늘리지 않고도 경계를 개선할 수 있는지.
셋째, 모델을 8비트로 줄여도 정확도가 유지되는지입니다.

제안서의 가설 H1, H2, H3이 각각 이 세 질문에 해당합니다.
""")


def s_data(d):
    s, top = d.content("데이터와 평가", "무엇으로, 어떻게 평가하는가")
    lw = 5.9
    tf = textbox(s, ML, top + 0.12, lw, 0.3)
    para(tf, "데이터  공개 데이터셋 2종, 4,130장 (결절 4,457개)", 13, bold=True, color=ORANGE,
         first=True, space_after=0)
    y = top + 0.6
    for name, n, fig, desc in [
        ("TN3K", "3,493장", "fig11_tn3k_samples",
         "중국 공개 데이터. 병변 주변을 확대한 영상"),
        ("DDTI", "637장", "fig12_ddti_samples",
         "콜롬비아 공개 데이터. 장비 화면을 그대로 캡처"),
    ]:
        tf = textbox(s, ML, y, lw, 0.6)
        para(tf, f"{name}   {n}", 14.5, bold=True, first=True, space_after=2)
        para(tf, desc, 11.5, color=MUTED, space_after=0)
        picture(s, f"{FIG}/{fig}.png", ML, y + 0.62, lw, 1.3)
        y += 2.2

    x = 7.15
    tf = textbox(s, x, top + 0.12, W - MR - x, 4.8)
    para(tf, "평가 방법", 13, bold=True, color=ORANGE, first=True, space_after=8)
    for t, b in [
        ("5겹 교차검증", "두 데이터셋을 섞어 다섯 번 나눠 학습해 결과가 우연이 아닌지 본다."),
        ("별도 평가 집합", "TN3K test 614장은 학습과 모델 선택에 전혀 쓰지 않고 최종 평가에만 쓴다."),
        ("교차 데이터셋", "한 데이터셋으로만 학습하고 다른 데이터셋에서 평가해, "
                       "처음 보는 장비 영상에서도 통하는지 본다."),
    ]:
        para(tf, t, 13.5, bold=True, space_before=2, space_after=2)
        para(tf, b, 12, color=MUTED, space_after=8, line=1.35)
    para(tf, "평가 지표", 13, bold=True, color=ORANGE, space_before=8, space_after=8)
    para(tf, "mAP (검출·분할 전반)  +  Dice, Boundary IoU, HD95 (경계 정확도)",
         12.5, bold=True, space_after=4, line=1.35)
    para(tf, "개선이 우연인지는 같은 영상끼리 짝지어 비교하는 대응표본 검정으로 확인한다.",
         12, color=MUTED, space_after=0, line=1.35)
    notes(s, """
데이터는 공개 데이터셋 두 개, TN3K와 DDTI를 사용합니다. 합쳐서 4,130장이고 결절은 4,457개입니다.
보시는 것처럼 TN3K는 병변 주변을 확대한 영상이고, DDTI는 장비 화면을 그대로 캡처한 영상이라 생김새가 꽤 다릅니다.

평가는 세 가지로 합니다.
5겹 교차검증으로 결과가 우연이 아닌지 보고, TN3K 테스트 614장은 학습에 전혀 쓰지 않고 최종 평가에만 씁니다.
그리고 한쪽 데이터셋으로만 학습해서 다른 쪽에서 평가하는 교차 실험으로, 처음 보는 장비 영상에서도 통하는지 확인합니다.

지표는 일반적인 mAP에 더해, 경계를 직접 보는 Dice, Boundary IoU, HD95를 함께 씁니다.
""")


def s_brm(d):
    s, top = d.content("제안 방법 ①", "경계 정교화: 학습할 때만 경계를 가르친다")
    bh = 0.72
    xs = [ML, ML + 1.85, ML + 4.25, ML + 6.15]      # 영상, 모델, 마스크, 손실 열
    ws = [1.45, 2.0, 1.5, 1.85]

    def row(y, last_text, last_sub, last_fill, last_line, last_color):
        box(s, xs[0], y, ws[0], bh, "초음파 영상", size=12.5)
        box(s, xs[1], y, ws[1], bh, "YOLOv11-seg", "nano", size=13)
        box(s, xs[2], y, ws[2], bh, "결절 마스크", size=12.5)
        box(s, xs[3], y, ws[3], bh, last_text, last_sub, fill=last_fill, line=last_line,
            color=last_color, size=12.5)
        for i in range(3):
            arrow(s, xs[i] + ws[i] + 0.04, y + bh / 2, xs[i + 1] - 0.04, y + bh / 2)

    # 학습할 때
    y1 = top + 0.5
    tf = textbox(s, ML, top + 0.08, 3, 0.3)
    para(tf, "학습할 때", 13, bold=True, color=ORANGE, first=True, space_after=0)
    row(y1, "경계 가중 손실", "경계 부근 오차 ×3", WHITE, ORANGE, ORANGE)
    yb = y1 + 1.05
    box(s, xs[1], yb, ws[1], bh, "경계 분기", "+4,657 파라미터", fill=WHITE, line=ORANGE,
        color=ORANGE, size=13, dash=True)
    box(s, xs[3], yb, ws[3], bh, "경계 손실", "정답 경계와 비교", fill=WHITE, line=ORANGE,
        color=ORANGE, size=12.5)
    arrow(s, xs[1] + ws[1] / 2, y1 + bh + 0.04, xs[1] + ws[1] / 2, yb - 0.04)
    arrow(s, xs[1] + ws[1] + 0.04, yb + bh / 2, xs[3] - 0.04, yb + bh / 2)

    # 배포할 때
    y2 = top + 3.05
    rect(s, ML, top + 2.75, xs[3] + ws[3] - ML, 0.012, fill=RULE)
    tf = textbox(s, ML, y2 - 0.02, 3, 0.3)
    para(tf, "실제로 쓸 때", 13, bold=True, color=BLUE, first=True, space_after=0)
    y2 += 0.42
    row(y2, "결과 출력", None, PANEL, None, INK)
    yb2 = y2 + 1.05
    box(s, xs[1], yb2, ws[1], bh, "경계 분기 떼어냄", fill=WHITE, line=FAINT, color=FAINT,
        size=12, bold=False, dash=True)
    box(s, xs[3] - 0.25, yb2, ws[3] + 0.25, bh, "9.73 GFLOPs", "기존 모델과 같은 연산량",
        fill=WHITE, line=BLUE, color=BLUE, size=13)

    x = 9.15
    tf = textbox(s, x, top + 0.15, W - MR - x, 4.8)
    para(tf, "핵심 아이디어", 13, bold=True, color=ORANGE, first=True, space_after=9)
    for t, b in [
        ("작은 경계 분기", "분할 헤드 옆에 합성곱 두 층짜리 분기를 달아 결절의 테두리만 "
                        "따로 맞히게 한다. 전체 파라미터의 0.16%."),
        ("경계 가중 손실", "정답 마스크의 테두리 띠에 있는 픽셀은 틀렸을 때 3배로 "
                        "벌점을 준다. 내부보다 경계를 더 신경 쓰게 만든다."),
        ("쓸 때는 떼어낸다", "경계를 보는 능력은 학습 중 모델 가중치에 남는다. "
                          "분기는 떼어내므로 실제 계산량은 기존 모델과 같다."),
    ]:
        para(tf, t, 13.5, bold=True, space_before=2, space_after=3)
        para(tf, b, 12, color=MUTED, space_after=10, line=1.38)
    notes(s, """
첫 번째 방법인 경계 정교화입니다.

기본 모델은 YOLOv11 분할 모델의 가장 작은 nano 버전입니다.
학습할 때는 여기에 작은 경계 분기를 하나 달아서, 결절의 테두리만 따로 맞히도록 가르칩니다. 이 분기는 합성곱 두 층, 파라미터 4,657개로 전체의 0.16%밖에 안 됩니다.
그리고 마스크 손실을 계산할 때 정답 경계 주변의 픽셀은 틀리면 3배로 벌점을 줘서, 모델이 내부보다 경계를 더 신경 쓰게 만듭니다.

중요한 점은 아래쪽입니다. 실제로 쓸 때는 이 경계 분기를 떼어냅니다. 경계를 보는 능력은 학습 과정에서 모델 가중치에 이미 남아 있기 때문에, 분기를 떼도 효과는 유지되고 계산량은 기존 모델과 똑같습니다.
""")


def s_quant(d):
    s, top = d.content("제안 방법 ②", "경량화: 학습이 끝난 모델을 8비트로 바꾼다")
    bh = 0.95
    y = top + 0.55
    bw = 2.35
    gap = 0.55
    labels = [("학습된 모델", "PyTorch FP32 · 6.0 MB", PANEL, None, INK),
              ("ONNX 변환", "표준 형식으로 내보내기", PANEL, None, INK),
              ("INT8 사후 양자화", "재학습 없이, 학습 영상\n일부로 범위 보정", WHITE, ORANGE, ORANGE)]
    for i, (t, sub, fill, line, col) in enumerate(labels):
        x = ML + i * (bw + gap)
        box(s, x, y, bw, bh, t, sub.replace("\n", " "), fill=fill, line=line, color=col,
            size=13.5)
        if i < 2:
            arrow(s, x + bw + 0.05, y + bh / 2, x + bw + gap - 0.05, y + bh / 2)
    x4 = ML + 3 * (bw + gap)
    w4 = W - MR - x4
    for j, (t, sub) in enumerate([("ONNX Runtime", "범용 추론 엔진"),
                                  ("OpenVINO", "인텔의 CPU 추론 엔진")]):
        yy = y - 0.5 + j * 1.25
        box(s, x4, yy, w4, 0.72, t, sub, fill=WHITE, line=BLUE, color=BLUE, size=13)
        arrow(s, x4 - gap + 0.05, y + bh / 2, x4 - 0.05, yy + 0.36)

    yt = top + 2.4
    rect(s, ML, yt, CW, 0.012, fill=RULE)
    tf = textbox(s, ML, yt + 0.25, 5.9, 2.4)
    para(tf, "INT8 양자화란", 13, bold=True, color=ORANGE, first=True, space_after=8)
    para(tf, "32비트 실수로 된 가중치와 연산을 8비트 정수로 바꾸는 것이다. 모델 크기가 줄고, "
             "CPU의 정수 연산을 쓸 수 있어 빨라진다.", 13, space_after=8, line=1.4)
    para(tf, "다시 학습하지 않고 변환만 하므로 비용이 거의 들지 않는다. 대신 정확도가 떨어질 수 "
             "있어, 원본과 같은 평가 집합에서 다시 측정해 확인한다.", 12.5, color=MUTED,
         space_after=0, line=1.4)
    x = 7.15
    tf = textbox(s, x, yt + 0.25, W - MR - x, 2.4)
    para(tf, "비교 항목", 13, bold=True, color=ORANGE, first=True, space_after=8)
    para(tf, "정밀도  PyTorch FP32, ONNX FP32 / FP16 / INT8, OpenVINO INT8", 12.5,
         space_after=6, line=1.35)
    para(tf, "측정  파일 크기, CPU 처리 시간(영상 1장), 정확도(mAP, Dice, HD95)", 12.5,
         space_after=10, line=1.35)
    para(tf, "같은 INT8이라도 변환 도구에 따라 결과가 달라질 수 있어 두 도구를 모두 비교한다.",
         12.5, color=MUTED, space_after=0, line=1.4)
    notes(s, """
두 번째 방법은 경량화입니다.

학습이 끝난 모델을 ONNX라는 표준 형식으로 내보낸 다음, 8비트 정수로 양자화합니다.
다시 학습하는 것이 아니라 변환만 하는 사후 양자화라서 비용이 거의 들지 않습니다. 학습 영상 일부를 넣어서 값의 범위만 맞춰 줍니다.

8비트로 바꾸면 모델이 작아지고 CPU의 정수 연산을 쓸 수 있어 빨라지지만, 정확도가 떨어질 수 있습니다. 그래서 원본과 같은 평가 집합에서 정확도를 다시 재서 확인합니다.
변환 도구로는 범용 엔진인 ONNX Runtime과 인텔이 만든 OpenVINO 두 가지를 비교했습니다.
""")


def s_result_boundary(d):
    s, top = d.content("예비 결과 ①", "경계 정교화: 계산량은 그대로, 경계는 개선",
                       "TN3K 평가 집합 614장, 5겹 교차검증 평균")
    tiles = [
        ("Dice", "0.8195 → 0.8300", "결절 영역 겹침 정도", ORANGE),
        ("HD95  경계의 큰 오차", "−13.0%", "9.63 → 8.38 (영상 대각선 대비 %)", ORANGE),
        ("Boundary IoU", "0.4358 → 0.4414", "경계 부근만 본 겹침 정도", ORANGE),
        ("연산량", "9.73 GFLOPs", "기존 모델과 같음", BLUE),
    ]
    tw, th, g = 2.75, 1.5, 0.2
    for i, (lab, val, sub, col) in enumerate(tiles):
        x = ML + (i % 2) * (tw + g)
        y = top + 0.2 + (i // 2) * (th + g)
        rect(s, x, y, tw, th, fill=PANEL)
        tf = textbox(s, x + 0.25, y + 0.2, tw - 0.4, th - 0.3)
        para(tf, lab, 11.5, bold=True, color=MUTED, first=True, space_after=6)
        para(tf, val, 20, bold=True, color=col, space_after=5, line=1.1)
        para(tf, sub, 10.5, color=MUTED, space_after=0, line=1.25)
    tf = textbox(s, ML, top + 3.65, 2 * tw + g, 1.0)
    para(tf, "같은 영상끼리 짝지어 비교한 검정에서 세 지표 모두 유의했다 (p < 0.05). "
             "검출 정확도(mAP)는 0.627로 그대로 유지되었다.",
         12, color=MUTED, first=True, space_after=0, line=1.4)

    x = ML + 2 * tw + g + 0.55
    w = W - MR - x
    labs = ["정답", "기존 모델", "제안 모델"]
    cols = [RGBGreen, BLUE, ORANGE]
    for i, (lab, col) in enumerate(zip(labs, cols)):
        tf = textbox(s, x + i * w / 3, top + 0.25, w / 3, 0.3)
        para(tf, lab, 12, bold=True, color=col, align=PP_ALIGN.CENTER, first=True,
             space_after=0)
    # fig4 의 두 번째 행(상위 4분위 사례). 최상 사례만 보여주면 효과가 과장되므로 이 행을 쓴다.
    picture_buf(s, crop_png(f"{FIG}/fig4_qualitative.png", (0.035, 0.348, 0.995, 0.555)),
                x, top + 0.6, w, 1.75)
    tf = textbox(s, x, top + 2.55, w, 1.6)
    para(tf, "제안 모델의 윤곽이 정답 경계에 더 가깝게 붙는다 (Dice 0.909 → 0.958). "
             "개선 폭 상위 25% 지점의 사례이다.", 12, first=True, space_after=8, line=1.4)
    para(tf, "반대로 주변 조직을 결절로 잘못 포함한 실패 사례도 있어, 남은 과제로 분석할 예정이다.",
         12, color=MUTED, space_after=0, line=1.4)
    notes(s, """
예비 실험 결과입니다. 학습에 쓰지 않은 TN3K 테스트 614장에서, 5겹 교차검증 평균으로 비교했습니다.

경계 정교화를 붙이니 Dice가 0.8195에서 0.8300으로 올랐고, 경계의 큰 오차를 보는 HD95는 13% 줄었습니다. 경계 부근만 보는 Boundary IoU도 올랐습니다.
같은 영상끼리 짝지어 검정했을 때 세 지표 모두 통계적으로 유의했습니다.
그러면서 연산량은 9.73 GFLOPs로 기존 모델과 똑같습니다.

오른쪽은 실제 예시입니다. 가장 좋은 사례만 고르면 과장이 될 것 같아, 개선 폭이 상위 25% 정도인 사례를 가져왔습니다. 제안 모델의 윤곽이 정답에 더 붙어 있습니다.
다만 주변 조직을 결절로 잘못 잡는 실패 사례도 있어서, 이 부분은 뒤에서 말씀드릴 남은 과제에 넣었습니다.
""")


def s_result_quant(d):
    s, top = d.content("예비 결과 ②", "경량화: CPU에서 15배 빨라졌다",
                       "영상 1장 기준, 서버 CPU에서 측정")
    picture(s, f"{FIG}/fig2_accuracy_vs_latency.png", ML, top + 0.1, 5.7, 4.25)
    x = 6.85
    rows = [
        ["형식", "크기", "CPU 처리 시간", "Dice"],
        ["ONNX FP32 (변환 전)", "11.63 MB", "62.82 ms", "0.8300"],
        ["ONNX Runtime INT8", "5.45 MB", "44.00 ms", "0.8184"],
        ["OpenVINO INT8", "3.71 MB", "4.10 ms", "0.8290"],
    ]
    table(s, rows, x, top + 0.25, W - MR - x, col_w=[2.5, 1.3, 1.6, 1.1], size=12,
          head_size=11.5, row_h=0.46, highlight={3})
    tf = textbox(s, x, top + 2.35, W - MR - x, 2.4)
    para(tf, "OpenVINO INT8", 13.5, bold=True, color=ORANGE, first=True, space_after=4)
    para(tf, "변환 전보다 15.3배 빠르고 크기는 3.71 MB로 가장 작다. Dice 손실은 0.001로 "
             "거의 없다.", 12.5, space_after=12, line=1.4)
    para(tf, "ONNX Runtime INT8", 13.5, bold=True, color=MUTED, space_after=4)
    para(tf, "검출 헤드까지 8비트로 바꾸면 모델이 작동하지 않아 그 부분은 실수 연산으로 "
             "남겨야 했다. 그래서 속도 이득이 작고 Dice도 0.012 떨어졌다.",
         12.5, color=MUTED, space_after=0, line=1.4)
    notes(s, """
경량화 결과입니다. 영상 한 장을 CPU로 처리하는 시간을 쟀습니다.

OpenVINO로 8비트 변환한 모델은 크기 3.71MB, 처리 시간 4.1ms로, 변환 전 62.8ms보다 15배 이상 빨라졌습니다. 그러면서 Dice는 0.8300에서 0.8290으로 거의 그대로입니다.

반면 ONNX Runtime으로 변환했을 때는 검출 헤드까지 8비트로 바꾸면 모델이 아예 작동하지 않아서, 그 부분을 실수 연산으로 남겨야 했습니다. 그래서 44ms로 속도 이득이 작고 정확도도 조금 더 떨어졌습니다.
같은 8비트라도 변환 도구에 따라 결과가 꽤 다르다는 점도 하나의 결과로 정리하려고 합니다.
""")


def s_answers(d):
    s, top = d.content("예비 결과 정리", "세 질문에 대한 현재까지의 답")
    rows = [
        ["질문", "예비 결과", "현재 판단"],
        ["Q1  작은 모델로 충분한가", "nano 0.6272, small 0.6239 (Mask mAP). "
                                "nano가 파라미터 1/3.5로 동등", "그렇다"],
        ["Q2  비용 없이 경계 개선", "Dice +0.0104, HD95 −13.0%, 연산량 동일",
         "그렇다 (계수 재검토 필요)"],
        ["Q3  8비트로도 정확도 유지", "OpenVINO는 Dice −0.001, ONNX Runtime은 −0.012",
         "도구에 따라 다르다"],
        ["다른 데이터셋에서도 유지되나", "3회 반복 결과 개선도 악화도 유의하지 않음",
         "아직 판단할 수 없다"],
    ]
    table(s, rows, ML, top + 0.25, CW, col_w=[3.3, 5.9, 2.8], size=13, head_size=12,
          row_h=0.68, align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
    y = top + 3.95
    rect(s, ML, y, CW, 0.82, fill=PANEL)
    rect(s, ML, y, 0.045, 0.82, fill=ORANGE)
    tf = textbox(s, ML + 0.3, y, CW - 0.6, 0.82, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, "교차 데이터셋 실험은 한 번만 돌렸을 때 크게 나빠진 것처럼 보였지만, 시드를 바꿔 "
             "3회 반복하니 실행마다의 차이가 더 컸다. 그래서 결론을 유보하고 추가 검증을 계획했다.",
         12.5, first=True, space_after=0, line=1.4)
    notes(s, """
지금까지의 결과를 세 질문에 맞춰 정리하면 이렇습니다.

첫째, 작은 nano 모델로 충분했습니다. small 모델보다 파라미터가 3.5배 적은데 정확도는 같은 수준이었습니다.
둘째, 계산량을 늘리지 않고 경계를 개선할 수 있었습니다. 다만 손실의 계수를 정한 방식에 보완할 점이 있어서, 이건 다음 장에서 말씀드리겠습니다.
셋째, 8비트 변환은 OpenVINO에서는 정확도가 유지됐고, ONNX Runtime에서는 조금 떨어졌습니다.

그리고 다른 데이터셋에서도 효과가 유지되는지는 아직 판단할 수 없었습니다. 처음 한 번 돌렸을 때는 크게 나빠진 것처럼 보였는데, 시드를 바꿔 세 번 반복해 보니 실행마다의 차이가 효과보다 커서, 지금 데이터로는 좋아진다고도 나빠진다고도 말하기 어렵습니다.
""")


def s_next(d):
    s, top = d.content("향후 계획", "앞으로 보완할 부분")
    items = [
        ("01", "경계 손실 계수 다시 정하기",
         "지금 쓰는 계수는 평가 집합 결과를 보고 골라서 개선 폭이 실제보다 좋게 나왔을 수 있다. "
         "검증 집합만으로 다시 고르고 해당 모델을 재학습해 수치를 확정한다."),
        ("02", "다른 데이터셋에서의 효과 판정",
         "반복 횟수를 늘리고, 가능하면 제3의 데이터셋을 추가해 일반화 여부를 판단할 수 있는 "
         "수준까지 검증한다."),
        ("03", "실패 사례 분석",
         "주변 조직을 결절로 잘못 잡는 사례의 원인을 분석하고, 이를 줄이는 방법을 검토한다."),
        ("04", "실제 장비에서 속도 측정",
         "지금의 처리 시간은 서버 CPU 기준이다. 일반 PC나 저전력 장비에서 다시 측정한다."),
    ]
    y = top + 0.2
    for num, t, b in items:
        tf = textbox(s, ML, y, 0.6, 0.4)
        para(tf, num, 14, bold=True, color=ORANGE, first=True, space_after=0)
        tf = textbox(s, ML + 0.75, y - 0.03, CW - 0.75, 1.0)
        para(tf, t, 16, bold=True, first=True, space_after=4)
        para(tf, b, 12.5, color=MUTED, space_after=0, line=1.38)
        y += 1.18
    notes(s, """
앞으로 보완할 부분은 네 가지입니다.

가장 먼저 할 것은 경계 손실의 계수를 다시 정하는 일입니다. 지금 계수는 평가 집합 결과를 보고 골랐기 때문에, 개선 폭이 실제보다 좋게 나왔을 가능성이 있습니다. 검증 집합만 보고 다시 고른 뒤 재학습해서 최종 수치를 확정하겠습니다. 미리 확인해 보니 검증 집합 기준으로도 모든 계수 설정이 기존 모델보다는 나았습니다.

두 번째로, 다른 데이터셋에서도 효과가 있는지 판정하기 위해 반복 횟수를 늘리고, 가능하면 데이터셋을 하나 더 추가하려고 합니다.
세 번째는 실패 사례 분석이고, 네 번째는 서버가 아닌 일반 PC나 저전력 장비에서 속도를 다시 재는 것입니다.
""")


def s_discuss(d):
    s, top = d.content("논의", "상의드리고 싶은 점")
    items = [
        ("외부 데이터 확보",
         "다른 데이터셋에서의 효과를 판정하려면 제3의 데이터가 필요합니다. "
         "연구실이나 협력 병원 데이터를 쓸 수 있는지 여쭙고 싶습니다."),
        ("임상적 의미 검증의 범위",
         "예측한 경계로 TI-RADS 경계 범주를 자동 판정해 전문의 판독과 비교하는 것까지 "
         "학위논문 범위에 넣을지 의견을 듣고 싶습니다."),
        ("결과 발표 계획",
         "학위논문과 별도로 학회나 학술지 투고를 준비하는 것이 좋을지 여쭙고 싶습니다."),
    ]
    y = top + 0.25
    for i, (t, b) in enumerate(items):
        rect(s, ML, y, CW, 1.15, fill=PANEL)
        rect(s, ML, y, 0.045, 1.15, fill=ORANGE)
        tf = textbox(s, ML + 0.35, y, CW - 0.7, 1.15, anchor=MSO_ANCHOR.MIDDLE)
        para(tf, t, 15.5, bold=True, first=True, space_after=4)
        para(tf, b, 12.5, color=MUTED, space_after=0, line=1.38)
        y += 1.35
    tf = textbox(s, ML, y + 0.12, CW, 0.4)
    para(tf, "감사합니다.", 14, bold=True, color=MUTED, first=True, space_after=0)
    notes(s, """
마지막으로 상의드리고 싶은 점이 세 가지 있습니다.

첫째, 다른 데이터셋에서의 효과를 제대로 판정하려면 데이터가 하나 더 필요한데, 연구실이나 협력 병원 데이터를 쓸 수 있을지 여쭙고 싶습니다.
둘째, 예측한 경계로 TI-RADS 경계 범주를 자동으로 판정해서 전문의 판독과 비교하는 것까지 학위논문 범위에 넣을지 의견을 듣고 싶습니다.
셋째, 학위논문과 별도로 학회나 학술지 투고를 준비하는 게 좋을지도 여쭙고 싶습니다.

이상입니다. 감사합니다.
""")


def main():
    d = Deck()
    s_title(d)
    s_onepage(d)
    s_why(d)
    s_problem(d)
    s_questions(d)
    s_data(d)
    s_brm(d)
    s_quant(d)
    s_result_boundary(d)
    s_result_quant(d)
    s_answers(d)
    s_next(d)
    s_discuss(d)

    cp = d.prs.core_properties
    cp.title, cp.subject = TITLE, SUBTITLE
    cp.author = cp.last_modified_by = cp.comments = cp.keywords = cp.category = ""
    d.prs.save(OUT)
    print(f"저장: {OUT}  ({len(d.prs.slides)} slides)")


if __name__ == "__main__":
    main()
