"""정확도 평가 탭의 Streamlit 렌더링.

ui.py 와의 순환 import를 피하기 위해 자동 라벨링 함수는 인자로 받는다.
평가는 인증 평가셋(nuScenes, KITTI, BDD100K, TuSimple)에 대해서만 한다. 데이터셋을 고르면
이미지 폴더, GT, 평가 항목이 정해진다(DATASETS, TASKS 표).
절차는 1) 평가 데이터셋 선택 -> 2) 자동 라벨링 -> 3) GT 확인 -> 4) 정확도 측정 -> 5) 결과 시각화 순이다.
"""

import datetime
import json
import os
import random
import re
import traceback

import streamlit as st

import evaluation as E
import visualization as V

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_EVAL_DIR = os.environ.get("EVAL_DIR", "/mnt/d/02_Dataset/04_Bigdata/Eval")
IOU_THRESHOLD = 0.5

# 세션 키. Streamlit은 위젯 조작마다 스크립트를 다시 돌리므로 중간 결과를 세션에 둔다.
K_PRED = "eval_predictions"      # {stem: {"detection": [...], "weather":..., "time":..., "lanes":[...]}}
K_ERRORS = "eval_errors"
K_REPORT = "eval_report"
K_PREVIEW = "eval_preview"       # 미리보기로 그린 이미지. '다른 이미지 보기'를 누를 때만 다시 그린다
K_ZIP = "eval_vis_zips"          # {"random": {...}, "all": {...}} 마지막으로 저장한 zip 정보
RESULT_KEYS = (K_PRED, K_ERRORS, K_REPORT, K_PREVIEW, K_ZIP)

PREVIEW_COUNT = 5
RANDOM_ZIP_COUNT = 100
DOWNLOAD_LIMIT = 200 * 1024 * 1024   # 이보다 큰 zip은 메모리를 아끼려고 다운로드 버튼 없이 파일로만 둔다

# 인증 기준(생성 데이터셋 정확도 85% 이상)을 판정하는 지표. 항목마다 맨 앞 지표에 붙인다
PRECISION_HELP = ("정밀도(Precision) = 정답인 생성 라벨 / 판정한 생성 라벨. "
                  "생성 데이터셋 정확도의 인증 기준(85% 이상)을 판정하는 값입니다.")


# --------------------------------------------------------------------------
# 보조
# --------------------------------------------------------------------------

def _clean_path(text):
    return os.path.normpath(text.strip(' "\''))


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _clear(keys):
    for key in keys:
        st.session_state.pop(key, None)


def _reset_results():
    """데이터셋이나 이미지 범위가 바뀌면 이전 자동 라벨링·평가·시각화 결과를 지운다."""
    _clear(RESULT_KEYS)


def _pct(x):
    return f"{x * 100:.2f}%"


def _ratio(x, defined=True):
    """표의 비율 칸. 정의되지 않는 칸('-')과 섞여도 열 형식이 하나가 되도록 문자열로 쓴다."""
    return f"{x:.4f}" if defined else "-"


# 폴더 목록과 GT는 위젯 조작마다 스크립트가 다시 돌아도 한 번만 읽도록 캐시한다.
# 캐시 키에 mtime을 넣어 파일이 바뀌면 다시 읽는다(밑줄로 시작하는 인자는 Streamlit이 키에서 빼므로 쓰지 않는다).
@st.cache_data(show_spinner=False)
def _list_images(folder, mtime):
    # os.scandir은 디렉터리 항목의 종류를 함께 돌려줘 /mnt/d에서도 파일마다 stat하지 않는다
    return sorted(e.path for e in os.scandir(folder) if e.name.lower().endswith(IMAGE_EXTS) and e.is_file())


# 아래 GT 로더는 읽기 전용으로만 쓰므로 복사 없이 같은 객체를 돌려주는 cache_resource를 쓴다
@st.cache_resource(show_spinner="nuScenes GT 로딩 중…")
def _load_nuscenes(coco_path, mtime):
    return E.NuScenesGT(coco_path)


@st.cache_resource(show_spinner="KITTI GT 로딩 중…")
def _load_kitti(label_dir, mtime):
    stems = [n[:-4] for n in os.listdir(label_dir) if n.endswith(".txt")]
    return E.load_kitti_gt(label_dir, stems)


@st.cache_resource(show_spinner="BDD100K GT 로딩 중…")
def _load_csv(csv_path, mtime):
    return E.load_classification_csv(csv_path)


@st.cache_resource(show_spinner="TuSimple GT 로딩 중…")
def _load_tusimple(lane_path, mtime):
    return E.load_tusimple_lanes(lane_path)


@st.cache_resource(max_entries=2, show_spinner=False)
def _zip_bytes(path, mtime):
    return _read_bytes(path)


# --------------------------------------------------------------------------
# 평가셋 정의: 데이터셋별 GT 로더
# --------------------------------------------------------------------------
# 각 로더는 (GT 경로, 자동 라벨링한 stem 목록)을 받아 GT 묶음 중 자기 항목을 채운 dict를 돌려준다.
# GT 묶음: {"detection": {stem: [...]}, "weather": {stem: str}, "time": {stem: str}, "lanes": {stem: TuSimple 레코드}}

def _gt_nuscenes(gt_path, stems):
    nusc = _load_nuscenes(gt_path, os.path.getmtime(gt_path))
    det = {s: nusc.detections(s) for s in stems if nusc.covers(s)}
    st.info(f"GT 매칭: {len(det):,} / {len(stems):,}장")
    with st.expander("nuScenes → KITTI 클래스 매핑"):
        st.dataframe([{"nuScenes": k, "KITTI": v or "평가 제외 (무시 영역)"} for k, v in E.NUSCENES_TO_KITTI.items()],
                     use_container_width=True)
        st.caption("공식 평가 필터나 가시도 60% 조건을 통과하지 못한 객체도 무시 영역입니다.")
    return {"detection": det}


def _gt_kitti(gt_path, stems):
    labels = _load_kitti(gt_path, os.path.getmtime(gt_path))
    det = {s: labels[s] for s in stems if s in labels}
    st.info(f"GT 매칭: {len(det):,} / {len(stems):,}장 (DontCare는 무시 영역)")
    return {"detection": det}


def _gt_bdd100k(gt_path, stems):
    w, t = _load_csv(gt_path, os.path.getmtime(gt_path))
    gt = {"weather": {s: w[s] for s in stems if s in w}, "time": {s: t[s] for s in stems if s in t}}
    st.info(f"GT 매칭: 날씨 {len(gt['weather']):,}장, 시간 {len(gt['time']):,}장 / {len(stems):,}장")
    st.caption("GT 칸이 빈 이미지(날씨 foggy·undefined, 시간 dawn/dusk·undefined)는 해당 항목 판정에서 제외됩니다.")
    return gt


def _gt_tusimple(gt_path, stems):
    records = _load_tusimple(gt_path, os.path.getmtime(gt_path))
    lanes = {s: records[s] for s in stems if s in records}
    st.info(f"GT 매칭: {len(lanes):,} / {len(stems):,}장")
    return {"lanes": lanes}


# 인증 평가셋. 경로는 평가 데이터 폴더(EVAL_DIR) 기준이고, tasks는 그 데이터셋으로 판정하는 항목(TASKS의 키)이다.
DATASETS = {
    "nuScenes": {"label": "nuScenes – 객체 인지", "images": "nuScenes/CAM_FRONT",
                 "gt": "nuScenes/annotations/nuscenes_val_CAM_FRONT_devkit2d_official_vis60.coco.json",
                 "load_gt": _gt_nuscenes, "tasks": ["Detection"]},
    "KITTI": {"label": "KITTI – 객체 인지", "images": "KITTI/image_2", "gt": "KITTI/label_2",
              "load_gt": _gt_kitti, "tasks": ["Detection"]},
    "BDD100K": {"label": "BDD100K – 날씨·시간 분류", "images": "BDD100K/images",
                "gt": "BDD100K/weather_time_gt.csv", "load_gt": _gt_bdd100k, "tasks": ["Weather", "Time"]},
    "TuSimple": {"label": "TuSimple – 차선 인지", "images": "TuSimple/images", "gt": "TuSimple/gt_lanes.json",
                 "load_gt": _gt_tusimple, "tasks": ["Lane"]},
}


def _paths(root, dataset):
    spec = DATASETS[dataset]
    return os.path.join(root, spec["images"]), os.path.join(root, spec["gt"])


# --------------------------------------------------------------------------
# 1단계: 평가 데이터셋 선택
# --------------------------------------------------------------------------

def _step_dataset():
    """평가 데이터셋을 고른다. (데이터셋 이름, 평가 데이터 폴더, [(stem, 파일명, 경로)])를 돌려준다."""
    st.subheader("1. 평가 데이터셋 선택")
    name = st.radio("평가 데이터셋", list(DATASETS), format_func=lambda k: DATASETS[k]["label"],
                    horizontal=True, key="eval_dataset", on_change=_reset_results)
    root = _clean_path(st.text_input("평가 데이터 폴더 (EVAL_DIR)", value=DEFAULT_EVAL_DIR, key="eval_root",
                                     on_change=_reset_results))
    image_dir, gt_path = _paths(root, name)
    st.caption(f"이미지 `{image_dir}`  \nGT `{gt_path}`  \n평가 항목: {', '.join(DATASETS[name]['tasks'])}")

    if not os.path.isdir(image_dir):
        st.error("이미지 폴더를 찾을 수 없습니다. 평가 데이터 폴더를 확인하세요.")
        return name, root, []
    if not os.path.exists(gt_path):
        st.error("GT를 찾을 수 없습니다. 평가 데이터 폴더를 확인하세요.")
        return name, root, []
    paths = _list_images(image_dir, os.path.getmtime(image_dir))
    limit = st.number_input("최대 장수 (0 = 전체, 인증 시험은 0)", min_value=0, value=0, step=100,
                            key="eval_limit", on_change=_reset_results)
    if limit:
        paths = paths[: int(limit)]
    st.info(f"이미지 {len(paths):,}장 선택됨")
    return name, root, [(E.stem_of(p), os.path.basename(p), p) for p in paths]


# --------------------------------------------------------------------------
# 2단계: 자동 라벨링
# --------------------------------------------------------------------------

def _step_autolabel(items, build_auto_labeling):
    st.subheader("2. 자동 라벨링 실행")

    if not items:
        st.caption("1단계에서 평가 데이터셋과 이미지를 먼저 확인하세요.")
        return

    if st.button(f"자동 라벨링 실행 ({len(items)}장)", key="eval_run", type="primary"):
        predictions, errors = {}, []
        progress = st.progress(0.0, text="처리 중…")
        for i, (stem, name, path) in enumerate(items, 1):
            try:
                auto = build_auto_labeling(name, _read_bytes(path))
                predictions[stem] = E.predictions_from_auto_labeling(auto)
            except Exception as exc:                      # 한 장 실패로 전체가 멈추지 않게 한다
                errors.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
            progress.progress(i / len(items), text=f"{i}/{len(items)}")
        progress.empty()
        st.session_state[K_PRED] = predictions
        st.session_state[K_ERRORS] = errors
        _clear((K_REPORT, K_PREVIEW, K_ZIP))            # 예측이 바뀌면 이전 평가·시각화 결과는 무효

    preds = st.session_state.get(K_PRED)
    if preds:
        errors = st.session_state.get(K_ERRORS) or []
        st.success(f"자동 라벨링 완료: {len(preds)}장" + (f" (실패 {len(errors)}장)" if errors else ""))
        if errors:
            with st.expander(f"실패 {len(errors)}건"):
                st.dataframe(errors, use_container_width=True)
        with st.expander("예측 결과 미리보기 (최대 5장)"):
            preview = {k: preds[k] for k in list(preds)[:5]}
            st.json(preview, expanded=False)


# --------------------------------------------------------------------------
# 3단계: GT 확인
# --------------------------------------------------------------------------

def _step_gt(dataset, root, stems):
    """데이터셋의 GT를 읽어 자동 라벨링한 이미지와 잇는다. 평가 항목에 쓰는 GT만 채운다."""
    st.subheader("3. GT 확인")
    gt = {"detection": {}, "weather": {}, "time": {}, "lanes": {}}
    if not stems:
        st.caption("2단계 자동 라벨링을 먼저 실행하세요.")
        return gt
    _, gt_path = _paths(root, dataset)
    if not os.path.exists(gt_path):
        st.error("GT를 찾을 수 없습니다.")
        return gt
    gt.update(DATASETS[dataset]["load_gt"](gt_path, stems))
    return gt


# --------------------------------------------------------------------------
# 4단계: 평가
# --------------------------------------------------------------------------

def _render_detection(report):
    """생성 라벨 기준: 생성된 박스 중 정답(같은 클래스, IoU 임계값 이상, GT 1:1)인 비율."""
    t = report["totals"]
    judged = t["TP"] + t["FP"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("정밀도", _pct(report["micro_precision"]), help=PRECISION_HELP)
    c2.metric("재현율", _pct(report["micro_recall"]))
    c3.metric("F1", _pct(report["micro_f1"]))
    c4.metric("판정한 생성 라벨", f"{judged:,}")
    st.caption(f"클래스 평균(GT가 있는 클래스): 정밀도 {_pct(report['macro_precision'])}, "
               f"재현율 {_pct(report['macro_recall'])}, F1 {_pct(report['macro_f1'])}")

    rows = []
    for cls, m in sorted(report["per_class"].items()):
        made = m["TP"] + m["FP"]
        rows.append({
            "class": cls,
            "생성 라벨": made,
            "정답": m["TP"],
            "정밀도": _ratio(m["precision"], made),
            "GT": m["n_gt"],
            "재현율": _ratio(m["recall"], m["n_gt"]),
            "F1": _ratio(m["f1"], made and m["n_gt"]),
            "오검출(FP)": m["FP"], "놓친 GT(FN)": m["FN"],
        })
    st.dataframe(rows, use_container_width=True)
    st.caption(f"정답 기준: 클래스 일치(KITTI 클래스로 매핑) + IoU {report['iou_threshold']:.2f} 이상, GT 1개당 생성 라벨 1개만 매칭. "
               f"정밀도 = 정답 {t['TP']:,} / 판정한 생성 라벨 {judged:,}. "
               f"무시 영역(DontCare, 평가 제외 객체) 안의 미매칭 생성 라벨 {t['n_pred'] - judged:,}개는 판정에서 뺐습니다. "
               f"이미지 {report['n_images']:,}장, GT {t['n_gt']:,}개.")


def _render_classification(title, report):
    """생성 라벨 기준: 이미지마다 1개씩 생성한 라벨 중 GT와 같은 비율."""
    if not report["n"]:
        st.warning(f"{title}: GT와 생성 라벨이 모두 있는 이미지가 없습니다.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("정밀도", _pct(report["accuracy"]), help=PRECISION_HELP)
    c2.metric("판정한 생성 라벨", f"{report['n']:,}")
    c3.metric("판정 제외", f"{report['n_excluded']:,}")

    rows = []
    for lab, m in report["per_class"].items():
        made = m["TP"] + m["FP"]
        rows.append({"class": lab, "생성 라벨": made, "정답": m["TP"],
                     "정밀도": _ratio(m["precision"], made),
                     "GT": m["support"], "재현율": _ratio(m["recall"], m["support"])})
    st.dataframe(rows, use_container_width=True)
    st.caption("이미지마다 라벨을 1개씩 생성하므로 정밀도 = GT와 같은 생성 라벨 / 판정한 생성 라벨입니다. "
               "클래스별 정밀도는 그 클래스로 생성한 라벨 중 맞은 비율입니다. "
               "판정 제외: GT가 없거나 GT 값이 모델 클래스에 없는 이미지(BDD100K의 foggy·undefined, dawn/dusk 등).")

    st.caption("혼동행렬 (행 = GT, 열 = 생성 라벨)")
    st.dataframe(
        [{"GT \\ 생성": lab, **{p: report["confusion"][i][j] for j, p in enumerate(report["labels"])}}
         for i, lab in enumerate(report["labels"])],
        use_container_width=True)


def _render_lanes(report):
    """생성 차선 기준: 생성된 차선 중 GT 차선과 맞는 비율."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("정밀도", _pct(report["precision"]), help=PRECISION_HELP)
    c2.metric("재현율", _pct(report["recall"]))
    c3.metric("F1", _pct(report["f1"]))
    c4.metric("생성 점 정밀도", _pct(report["point_precision"]))
    st.caption(f"생성 차선 {report['n_pred_lanes']:,}개 중 정답 {report['n_correct_lanes']:,}개, GT 차선 {report['n_gt_lanes']:,}개, "
               f"이미지 {report['n_images']:,}장. 정답 차선: 생성 차선의 점 중 {report['match_threshold']:.0%} 이상이 같은 y의 GT 차선과 "
               f"{report['pixel_threshold']}px / cos θ 미만으로 떨어진 경우(TuSimple 점 판정 기준), GT 차선 1개당 생성 차선 1개만 인정. "
               "평가는 GT의 h_samples 범위에서 하며, GT 차선이 없는 이미지의 생성 차선은 모두 틀린 차선으로 셉니다.")
    t = report["tusimple"]
    with st.expander("참고: TuSimple 공식 지표 (LaneEval)"):
        if t["n_images"]:
            d1, d2, d3 = st.columns(3)
            d1.metric("Accuracy", _pct(t["accuracy"]))
            d2.metric("FP", f"{t['FP']:.4f}")
            d3.metric("FN", f"{t['FN']:.4f}")
        st.caption(f"GT 차선 기준 지표입니다. GT 차선이 있는 이미지 {t['n_images']:,}장의 평균이며, "
                   "생성 차선이 GT보다 많아도 Accuracy는 줄지 않고 FP로만 나타납니다. "
                   f"생성 차선이 GT보다 3개 이상 많아 Accuracy 0·FP 0·FN 1로 처리된 이미지 {t['n_penalized_images']}장.")


def _evaluate_detection(preds, gt):
    return E.evaluate_detection(
        {s: {"pred": preds[s]["detection"], "gt": gt["detection"][s]} for s in preds if s in gt["detection"]},
        iou_thr=IOU_THRESHOLD)


def _evaluate_classification(key, labels):
    def evaluate(preds, gt):
        pairs = [(gt[key][s], preds[s][key]) for s in preds if s in gt[key] and preds[s][key] is not None]
        report = E.evaluate_classification(pairs, labels)
        report["n_excluded"] = sum(1 for s in preds if preds[s][key] is not None) - len(pairs)
        return report
    return evaluate


def _evaluate_lanes(preds, gt):
    return E.evaluate_lanes(
        {s: {"pred": preds[s]["lanes"], "gt": gt["lanes"][s]} for s in preds if s in gt["lanes"]})


# 평가 항목: 리포트 키, 결과 제목, 계산·표시 함수, 시각화 범례
_CLS_LEGEND = "이미지 위쪽에 생성 라벨과 GT를 표시합니다. [O] 정답, [X] 오답, [-] GT 없음(판정 제외)"
TASKS = {
    "Detection": {"key": "detection", "title": "객체 인지", "evaluate": _evaluate_detection,
                  "render": _render_detection,
                  "legend": f"🟦 TP(맞춘 검출)  🟥 FP(오검출)  🟨 FN(놓친 GT)  🟩 GT  ⬜ 회색: 무시 영역  — IoU {IOU_THRESHOLD:.2f} 기준"},
    "Weather": {"key": "weather", "title": "날씨 분류", "evaluate": _evaluate_classification("weather", E.WEATHER_CLASSES),
                "render": lambda r: _render_classification("Weather", r), "legend": _CLS_LEGEND},
    "Time": {"key": "time", "title": "시간 분류", "evaluate": _evaluate_classification("time", E.TIME_CLASSES),
             "render": lambda r: _render_classification("Time", r), "legend": _CLS_LEGEND},
    "Lane": {"key": "lanes", "title": "차선 인지", "evaluate": _evaluate_lanes, "render": _render_lanes,
             "legend": "청록: 생성 차선, 자홍: GT 차선"},
}


def _report_for_download(report):
    """평가 리포트에서 생성 라벨 평가에 쓰지 않는 AP·mAP를 뺀다."""
    out = dict(report)
    if "detection" in out:
        det = {k: v for k, v in out["detection"].items() if k != "mAP"}
        det["per_class"] = {c: {k: v for k, v in m.items() if k != "AP"} for c, m in det["per_class"].items()}
        out["detection"] = det
    return out


def _step_evaluate(dataset, preds, gt):
    st.subheader("4. 정확도 측정")
    if not preds:
        st.caption("2단계 자동 라벨링을 먼저 실행하세요.")
        return
    tasks = DATASETS[dataset]["tasks"]
    st.caption("평가 항목: " + ", ".join(f"{t} (IoU {IOU_THRESHOLD:.2f})" if t == "Detection" else t for t in tasks))

    if st.button("정확도 측정", key="eval_go", type="primary"):
        report = {"dataset": dataset}
        for t in tasks:
            report[TASKS[t]["key"]] = TASKS[t]["evaluate"](preds, gt)
        st.session_state[K_REPORT] = report
    report = st.session_state.get(K_REPORT)
    if not report:
        return

    st.divider()
    for t in tasks:
        st.markdown(f"### {TASKS[t]['title']}")
        TASKS[t]["render"](report[TASKS[t]["key"]])

    st.download_button("평가 리포트 JSON 다운로드",
                       data=json.dumps(_report_for_download(report), indent=2, ensure_ascii=False),
                       file_name=f"evaluation_report_{dataset}.json", mime="application/json",
                       key="eval_download")


# --------------------------------------------------------------------------
# 5단계: 결과 시각화
# --------------------------------------------------------------------------

def _windows_path(path):
    """WSL 경로(/mnt/d/...)를 탐색기에서 쓰는 형태(D:\\...)로 함께 보여준다."""
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", path)
    return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\") if m else path


def _caption(row):
    parts = [row["file"]]
    if row["TP"] != "":
        parts.append(f"TP {row['TP']} · FP {row['FP']} · FN {row['FN']}")
    for key, label in (("weather", "날씨"), ("time", "시간")):
        if row[f"{key}_pred"]:
            mark = {1: " ✅", 0: " ❌"}.get(row[f"{key}_ok"], "")
            parts.append(f"{label} {row[f'{key}_pred']} / GT {row[f'{key}_gt'] or '-'}{mark}")
    return "  |  ".join(parts)


def _render_preview(render, stems):
    """랜덤 이미지를 그려 [(JPEG 바이트, 설명) 또는 오류 문자열]로 돌려준다."""
    out = []
    for stem in random.sample(stems, min(PREVIEW_COUNT, len(stems))):
        try:
            _, data, row = render(stem)
            out.append((data, _caption(row)))
        except Exception as exc:
            out.append(f"{stem}: 시각화 실패 ({type(exc).__name__}: {exc})")
    return out


def _step_visualize(dataset, root, items, preds, gt):
    st.subheader("5. 결과 시각화")
    sources = {s: (name, path) for s, name, path in items if s in preds}
    stems = sorted(sources)
    if not stems:
        st.caption("2단계 자동 라벨링을 먼저 실행하세요.")
        return

    tasks = DATASETS[dataset]["tasks"]
    render = V.make_renderer(preds, gt, sources, tasks, iou_thr=IOU_THRESHOLD)
    for legend in dict.fromkeys(TASKS[t]["legend"] for t in tasks):
        st.caption(legend)

    # ---- 미리보기 ---------------------------------------------------------
    st.markdown(f"#### 랜덤 {min(PREVIEW_COUNT, len(stems))}장 미리보기")
    if st.button("다른 이미지 보기", key="eval_vis_shuffle") or K_PREVIEW not in st.session_state:
        st.session_state[K_PREVIEW] = _render_preview(render, stems)
    for item in st.session_state[K_PREVIEW]:
        if isinstance(item, str):
            st.warning(item)
        else:
            st.image(item[0], caption=item[1], use_column_width=True)

    # ---- zip 저장 ---------------------------------------------------------
    st.markdown("#### 시각화 이미지 zip 저장")
    # 평가 데이터 폴더 옆의 Eval_vis/<데이터셋>. 평가셋 폴더(체크섬 대상) 안에는 쓰지 않는다
    default_dir = os.path.join(os.path.dirname(root), "Eval_vis", dataset)
    out_dir = st.text_input("저장 폴더", value=default_dir, key=f"eval_vis_outdir_{dataset}")
    n_rand = min(RANDOM_ZIP_COUNT, len(stems))
    seed = int(st.number_input(f"랜덤 {n_rand}장 선택 시드 (같은 시드면 같은 이미지)", min_value=0,
                               value=42, step=1, key="eval_vis_seed"))
    c1, c2 = st.columns(2)
    do_rand = c1.button(f"랜덤 {n_rand}장 zip 저장", key="eval_zip_random", use_container_width=True)
    do_all = c2.button(f"전체 {len(stems)}장 zip 저장", key="eval_zip_all", use_container_width=True)

    if do_rand or do_all:
        out_dir = _clean_path(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if do_rand:
            kind, chosen = "random", sorted(random.Random(seed).sample(stems, n_rand))
            name = f"vis_random{n_rand}_seed{seed}_{stamp}.zip"
        else:
            kind, chosen = "all", stems
            name = f"vis_all{len(stems)}_{stamp}.zip"
        readme = V.zip_readme({
            "dataset": dataset,
            "evaluated": ", ".join(tasks),
            "images": len(chosen),
            "selection": f"random (seed {seed})" if do_rand else "all evaluated images",
            "iou_threshold": IOU_THRESHOLD if "Detection" in tasks else "-",
            "created": stamp,
        })
        bar = st.progress(0.0, text="시각화 이미지 생성 중…")
        info = V.build_zip(os.path.join(out_dir, name), chosen, render, readme=readme,
                           progress=lambda i, n: bar.progress(i / n, text=f"{i}/{n}"))
        bar.empty()
        info["size"] = os.path.getsize(info["path"])
        st.session_state.setdefault(K_ZIP, {})[kind] = info

    for kind, label in (("random", f"랜덤 {n_rand}장"), ("all", "전체")):
        info = (st.session_state.get(K_ZIP) or {}).get(kind)
        if not info or not os.path.isfile(info["path"]):
            continue
        msg = f"{label} zip 저장 완료: `{_windows_path(info['path'])}` ({info['size'] / 1e6:.1f} MB, {info['n_ok']}장)"
        if info["n_err"]:
            st.warning(msg + f" — 실패 {info['n_err']}장 (zip 안 summary.csv의 error 열 참고)")
        else:
            st.success(msg)
        if info["size"] <= DOWNLOAD_LIMIT:
            st.download_button(f"{label} zip 다운로드", data=_zip_bytes(info["path"], os.path.getmtime(info["path"])),
                               file_name=os.path.basename(info["path"]), mime="application/zip",
                               key=f"eval_zip_dl_{kind}")
        else:
            st.caption("용량이 커서 다운로드 버튼 대신 위 경로에 저장했습니다. 탐색기에서 열어주세요.")


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------

def render_evaluation_tab(build_auto_labeling):
    st.write("자동 라벨링으로 생성한 라벨 중 GT와 일치하는 비율(정밀도)을 인증 평가셋으로 측정합니다. "
             "평가 데이터셋 선택 → 자동 라벨링 → GT 확인 → 측정 → 시각화 순서로 진행하세요.")
    try:
        dataset, root, items = _step_dataset()
        st.divider()
        _step_autolabel(items, build_auto_labeling)
        st.divider()
        preds = st.session_state.get(K_PRED) or {}
        gt = _step_gt(dataset, root, sorted(preds))
        st.divider()
        _step_evaluate(dataset, preds, gt)
        st.divider()
        _step_visualize(dataset, root, items, preds, gt)
    except Exception:
        st.error("평가 탭에서 오류가 발생했습니다.")
        st.code(traceback.format_exc())
