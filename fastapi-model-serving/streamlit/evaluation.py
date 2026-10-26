"""자동 라벨링 결과의 정확도를 평가한다.

Streamlit UI와 분리된 순수 함수로 구성해 단독 테스트가 가능하다.

지원하는 GT (인증 평가셋별로 하나씩)
-----------
nuScenes  : 3D 박스를 투영한 2D bbox(COCO 형식, tools/make_nuscenes_2d_gt.py, iscrowd=1은 무시 영역).
KITTI     : ``label_2/*.txt`` 의 2D bbox. DontCare는 무시 영역.
BDD100K   : 날씨·시간 ``filename,weather,time`` CSV (tools/make_bdd100k_eval.py). 빈 칸은 평가 제외.
TuSimple  : 차선 TuSimple 형식(JSON Lines, tools/make_tusimple_eval.py).

내부 공통 표현
--------------
검출 GT   : {"cls": str, "bbox": (x1, y1, x2, y2), "ignore": bool}
검출 예측 : {"cls": str, "bbox": (x1, y1, x2, y2), "score": float | None}
차선      : [[(x, y), ...], ...]   (이미지 좌표계)
"""

import collections
import csv
import json
import os

import numpy as np

# --------------------------------------------------------------------------
# 클래스 정의와 매핑
# --------------------------------------------------------------------------

# 검출 모델(yolo11x_finetuned_*.pt, KITTI 8클래스)이 실제로 예측하는 클래스
KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian",
                 "Person_sitting", "Cyclist", "Tram", "Misc"]

# KITTI label_2 에만 존재하는 무시 영역. 여기에 겹치는 검출은 FP로 세지 않는다.
KITTI_IGNORE_CLASSES = {"DontCare"}

# nuScenes 10클래스 -> KITTI 8클래스. None은 "평가에서 제외"(GT·예측 모두 무시).
NUSCENES_TO_KITTI = {
    "car": "Car",
    "truck": "Truck",
    "bus": "Truck",
    "trailer": "Misc",          # KITTI는 화물 트레일러·캐러밴을 Misc로 라벨링한다
    "construction_vehicle": "Misc",
    "bicycle": "Cyclist",
    "motorcycle": "Misc",
    "pedestrian": "Pedestrian",
    "traffic_cone": None,
    "barrier": None,
}

WEATHER_CLASSES = ["Clear", "Overcast", "Rainy", "Snowy"]
TIME_CLASSES = ["Daytime", "Night"]


def stem_of(path_or_name):
    """경로/파일명에서 확장자를 뗀 basename을 돌려준다. GT와 예측을 잇는 키."""
    return os.path.splitext(os.path.basename(str(path_or_name)))[0]


# --------------------------------------------------------------------------
# GT 로더 - KITTI
# --------------------------------------------------------------------------

def parse_kitti_label(text):
    """KITTI label_2 텍스트 한 개를 파싱한다.

    형식: type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry [score]
    """
    objects = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        cls = parts[0]
        try:
            bbox = tuple(float(v) for v in parts[4:8])
        except ValueError:
            continue
        objects.append({
            "cls": cls,
            "bbox": bbox,
            "ignore": cls in KITTI_IGNORE_CLASSES,
        })
    return objects


def load_kitti_gt(label_dir, stems):
    """label_dir 안에서 stem.txt 들을 읽는다. 없는 파일은 결과에서 빠진다."""
    gt = {}
    for stem in stems:
        path = os.path.join(label_dir, stem + ".txt")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                gt[stem] = parse_kitti_label(f.read())
    return gt


# --------------------------------------------------------------------------
# GT 로더 - nuScenes
# --------------------------------------------------------------------------

class NuScenesGT:
    """nuScenes 2D GT COCO JSON(tools/make_nuscenes_2d_gt.py)을 읽어 검출 GT를 제공한다.

    coco_json 은 수십 MB라 한 번만 로드해 재사용한다(UI에서는 캐시).
    """

    def __init__(self, coco_json_path):
        with open(coco_json_path, encoding="utf-8") as f:
            data = json.load(f)

        # basename -> [{cls(nuScenes 원래 이름), bbox(x1,y1,x2,y2)}]
        self.boxes_by_stem = {}
        for ann in data.get("annotations", []):
            x, y, w, h = ann["bbox"]              # COCO는 [x, y, w, h]
            self.boxes_by_stem.setdefault(stem_of(ann["file_name"]), []).append({
                "cls": ann["category_name"],
                "bbox": (x, y, x + w, y + h),
                # COCO의 iscrowd=1은 무시 영역이다 (가시도 60% 미만, 벤치마크 외 클래스 등)
                "ignore": bool(ann.get("iscrowd", 0)),
            })

        # 박스가 0개인 이미지도 평가 대상에 넣으려고 이미지 목록을 따로 둔다
        self.image_stems = {stem_of(img["file_name"]) for img in data.get("images", [])}

    # -- 검출 ------------------------------------------------------------
    def covers(self, stem):
        """이 GT 셋에 포함된 이미지인지. 박스가 0개인 이미지도 포함된다.

        박스 유무(boxes_by_stem)로 판단하면 객체가 없는 이미지가 평가에서 빠져
        그 이미지의 검출이 FP로 집계되지 않는다.
        """
        return stem in self.image_stems

    def detections(self, stem, class_map=NUSCENES_TO_KITTI):
        """nuScenes 클래스를 KITTI 클래스로 바꿔 돌려준다. 매핑이 None이거나 iscrowd=1이면 무시 영역."""
        out = []
        for obj in self.boxes_by_stem.get(stem, []):
            mapped = class_map.get(obj["cls"], None)
            out.append({
                "cls": mapped if mapped else obj["cls"],
                "bbox": obj["bbox"],
                "ignore": mapped is None or obj["ignore"],
            })
        return out


# --------------------------------------------------------------------------
# GT 로더 - 분류 CSV / 차선 TuSimple
# --------------------------------------------------------------------------

def load_classification_csv(path):
    """tools/make_bdd100k_eval.py가 만든 `filename,weather,time` CSV를 읽는다. 빈 칸은 판정 제외라 뺀다."""
    weather, time_ = {}, {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = stem_of(row["filename"])
            if row["weather"]:
                weather[stem] = row["weather"]
            if row["time"]:
                time_[stem] = row["time"]
    return weather, time_


def load_tusimple_lanes(path):
    """TuSimple JSON Lines를 읽어 basename -> {"h_samples": [...], "lanes": [[x, ...], ...]} 로 돌려준다.

    공식 지표는 차선이 없는 지점(x = -2)까지 세므로 원본 배열을 그대로 둔다.
    점열이 필요하면 tusimple_lane_points()로 바꾼다.
    """
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[stem_of(rec.get("raw_file", ""))] = {
                "h_samples": [float(y) for y in rec.get("h_samples", [])],
                "lanes": [[float(x) for x in xs] for xs in rec.get("lanes", [])],
            }
    return records


def tusimple_lane_points(rec):
    """TuSimple 레코드 -> [[(x, y), ...], ...]. 차선이 없는 지점(x < 0)은 뺀다."""
    lanes = []
    for xs in rec["lanes"]:
        pts = [(x, y) for x, y in zip(xs, rec["h_samples"]) if x >= 0]
        if pts:
            lanes.append(pts)
    return lanes


# --------------------------------------------------------------------------
# 기하 계산
# --------------------------------------------------------------------------

def _intersection(box, boxes):
    """box(4,) 와 boxes(N,4) 의 교집합 넓이 (N,) 와 box 자신의 넓이."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter, max((box[2] - box[0]), 0) * max((box[3] - box[1]), 0)


def _iou_one_to_many(box, boxes):
    """box(4,) 와 boxes(N,4) 의 IoU (N,)."""
    if len(boxes) == 0:
        return np.zeros(0)
    inter, area_b = _intersection(box, boxes)
    area_s = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = area_b + area_s - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _ioa_one_to_many(box, boxes):
    """교집합을 box 자신의 넓이로 나눈 값. 무시 영역 판정에 쓴다."""
    if len(boxes) == 0:
        return np.zeros(0)
    inter, area_b = _intersection(box, boxes)
    return inter / max(area_b, 1e-9)


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _average_precision(rec, prec):
    """PR 곡선 아래 면적 (all-point interpolation, VOC2010/COCO 방식)."""
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# --------------------------------------------------------------------------
# 검출 평가
# --------------------------------------------------------------------------

def evaluate_detection(per_image, iou_thr=0.5):
    """per_image: {stem: {"pred": [...], "gt": [...]}} -> 클래스별 AP/P/R/F1 과 mAP.

    - 이미지마다 match_image_detections로 매칭한 결과를 클래스별로 합산한다.
      GT 매칭은 이미지 단위로 독립이라 전역으로 점수 정렬해 매칭한 결과와 같다.
    - 생성 라벨 정확도는 micro_precision(전체 박스 기준)이고, macro_*는 GT가 있는 클래스의 평균이다.
      mAP는 점수 문턱을 바꿔 가며 보는 모델 지표라, 문턱이 정해진 생성 라벨 평가에는 쓰지 않는다.
    """
    scored = collections.defaultdict(list)     # 클래스 -> [(score, 판정)] (이미지·예측 순서)
    n_gt = collections.Counter()
    for d in per_image.values():
        pred_status, gt_status = match_image_detections(d["pred"], d["gt"], iou_thr)
        for o, status in zip(d["pred"], pred_status):
            scored[o["cls"]].append((o["score"] if o.get("score") is not None else 1.0, status))
        for o, status in zip(d["gt"], gt_status):
            if status != "IGN":
                n_gt[o["cls"]] += 1

    per_class = {}
    for cls in sorted(set(scored) | set(n_gt)):
        rows = sorted(scored[cls], key=lambda t: -t[0])          # 안정 정렬: 같은 점수는 이미지 순서 유지
        ctp = np.cumsum([status == "TP" for _, status in rows])
        cfp = np.cumsum([status == "FP" for _, status in rows])  # 무시 영역(IGN)은 둘 다 0
        n_pos = n_gt[cls]
        n_tp = int(ctp[-1]) if len(rows) else 0
        n_fp = int(cfp[-1]) if len(rows) else 0
        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
        recall = n_tp / n_pos if n_pos else 0.0

        if n_pos and rows:
            ap = _average_precision(ctp / n_pos, ctp / np.maximum(ctp + cfp, 1e-9))
        elif n_pos:
            ap = 0.0
        else:
            ap = None       # GT가 없는 클래스는 AP 정의 불가 -> mAP에서 제외

        per_class[cls] = {
            "AP": ap, "precision": precision, "recall": recall, "f1": _f1(precision, recall),
            "TP": n_tp, "FP": n_fp, "FN": n_pos - n_tp, "n_gt": n_pos, "n_pred": len(rows),
        }

    with_ap = [v["AP"] for v in per_class.values() if v["AP"] is not None]
    totals = {k: sum(v[k] for v in per_class.values()) for k in ("TP", "FP", "FN", "n_gt", "n_pred")}
    micro_p = totals["TP"] / (totals["TP"] + totals["FP"]) if (totals["TP"] + totals["FP"]) else 0.0
    micro_r = totals["TP"] / totals["n_gt"] if totals["n_gt"] else 0.0
    with_gt = [v for v in per_class.values() if v["n_gt"]]
    macro = {k: float(np.mean([v[k] for v in with_gt])) if with_gt else 0.0 for k in ("precision", "recall", "f1")}
    return {
        "iou_threshold": iou_thr,
        "per_class": per_class,
        "mAP": float(np.mean(with_ap)) if with_ap else 0.0,
        "totals": totals,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": _f1(micro_p, micro_r),
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "n_images": len(per_image),
    }


def match_image_detections(pred, gt, iou_thr=0.5):
    """이미지 한 장 안에서 예측과 GT를 매칭한다. evaluate_detection과 시각화가 같이 쓰는 유일한 매칭 규칙.

    클래스마다 점수가 높은 예측부터, IoU가 임계값 이상이면서 아직 안 쓴 GT 중 IoU가 가장 큰 것과 짝짓는다.
    GT와 못 맞춘 예측이 무시 영역(DontCare, 미매핑 클래스)에 절반 이상 들어가 있으면 IGN으로 두고 집계하지 않는다.

    반환: (pred_status, gt_status)
      pred_status[i] in {"TP", "FP", "IGN"}  IGN = 무시 영역 안의 미매칭 검출
      gt_status[j]   in {"TP", "FN", "IGN"}  IGN = 무시 영역(DontCare, 미매핑 클래스)
    """
    pred_status = ["FP"] * len(pred)
    gt_status = ["IGN" if o.get("ignore") else "FN" for o in gt]
    ign_boxes = np.array([o["bbox"] for o in gt if o.get("ignore")], dtype=float).reshape(-1, 4)

    classes = {o["cls"] for o in pred} | {o["cls"] for o in gt if not o.get("ignore")}
    for cls in classes:
        gt_idx = [j for j, o in enumerate(gt) if o["cls"] == cls and not o.get("ignore")]
        boxes = np.array([gt[j]["bbox"] for j in gt_idx], dtype=float).reshape(-1, 4)
        used = np.zeros(len(gt_idx), dtype=bool)
        order_p = sorted((i for i, o in enumerate(pred) if o["cls"] == cls),
                         key=lambda i: -(pred[i]["score"] if pred[i].get("score") is not None else 1.0))
        for i in order_p:
            box = np.asarray(pred[i]["bbox"], dtype=float)
            if len(boxes):
                ious = _iou_one_to_many(box, boxes)
                best = next((j for j in np.argsort(-ious) if ious[j] >= iou_thr and not used[j]), None)
                if best is not None:
                    used[best] = True
                    pred_status[i] = "TP"
                    gt_status[gt_idx[best]] = "TP"
                    continue
            if len(ign_boxes) and _ioa_one_to_many(box, ign_boxes).max() >= 0.5:
                pred_status[i] = "IGN"
    return pred_status, gt_status


# --------------------------------------------------------------------------
# 분류 평가
# --------------------------------------------------------------------------

def evaluate_classification(pairs, labels):
    """pairs: [(gt, 생성 라벨), ...], labels: 고정 클래스 목록 -> accuracy, 클래스별 P/R/F1, 혼동행렬.

    생성 라벨이 labels에 없으면(서버 클래스명이 바뀐 경우 등) 혼동행렬에는 못 넣지만 틀린 라벨로 센다.
    """
    if not pairs:
        return {"n": 0, "accuracy": 0.0, "per_class": {}, "labels": [], "confusion": []}

    labels = list(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    conf = np.zeros((len(labels), len(labels)), dtype=int)   # 행=GT, 열=생성 라벨
    support = collections.Counter(g for g, _ in pairs)
    for g, p in pairs:
        if g in index and p in index:
            conf[index[g], index[p]] += 1

    per_class = {}
    for lab, i in index.items():
        tp = int(conf[i, i])
        fp = int(conf[:, i].sum() - tp)
        fn = support[lab] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support[lab] if support[lab] else 0.0
        per_class[lab] = {
            "precision": precision, "recall": recall, "f1": _f1(precision, recall),
            "support": support[lab], "TP": tp, "FP": fp, "FN": fn,
        }

    return {
        "n": len(pairs),
        "accuracy": sum(g == p for g, p in pairs) / len(pairs),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0,
        "labels": labels,
        "confusion": conf.tolist(),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------
# 차선 평가
# --------------------------------------------------------------------------
# 주 지표는 생성 차선 정확도(생성된 차선 중 정답인 비율)이고, 점 판정 기준은 TuSimple과 같다.
# 참고 지표인 TuSimple 공식 LaneEval은 벤치마크 평가 코드(LaneDetection/clrnet/utils/tusimple_metric.py)와
# 같은 계산이며, 기울기는 sklearn LinearRegression 대신 같은 최소제곱 해를 주는 np.polyfit으로 구한다.

LANE_PIXEL_THRESH = 20
LANE_PT_THRESH = 0.85


def _pred_lane_xs(points, h_samples):
    """예측 점열을 GT의 y 위치에서 x로 보간한다. 차선 y 범위 밖은 -2 (CLRNet to_array와 같은 규칙)."""
    if len(points) < 2:
        return [-2.0] * len(h_samples)
    pts = np.array(sorted(points, key=lambda p: p[1]), dtype=float)
    ys, idx = np.unique(pts[:, 1], return_index=True)
    xs = pts[idx, 0]
    return [float(np.interp(y, ys, xs)) if ys[0] <= y <= ys[-1] else -2.0 for y in h_samples]


def _lane_angle(xs, y_samples):
    valid = xs >= 0
    if valid.sum() > 1:
        return np.arctan(np.polyfit(y_samples[valid], xs[valid], 1)[0])
    return 0.0


def _lane_thresholds(gt, y_samples):
    """GT 차선별 점 허용 오차 20px / cos θ (θ: GT 차선 기울기, TuSimple 기준)."""
    return np.array([LANE_PIXEL_THRESH / np.cos(_lane_angle(g, y_samples)) for g in gt])


def _line_accuracy(pred, gt, thresh):
    pred = np.where(pred >= 0, pred, -100.0)
    gt = np.where(gt >= 0, gt, -100.0)
    return np.sum(np.abs(pred - gt) < thresh) / len(gt)


def tusimple_bench(pred, gt, y_samples):
    """이미지 1장의 (Accuracy, FP, FN). pred·gt는 y_samples 위치의 x 배열 목록 (없는 지점 -2).

    공식 코드의 실행 시간 조건(200ms 초과 시 0점)은 자동 라벨링 평가에 해당하지 않아 뺀다.
    """
    if len(gt) + 2 < len(pred):
        return 0.0, 0.0, 1.0
    y_samples = np.asarray(y_samples, dtype=float)
    pred = [np.asarray(p, dtype=float) for p in pred]
    gt = [np.asarray(g, dtype=float) for g in gt]
    line_accs, fn, matched = [], 0.0, 0.0
    for g, thresh in zip(gt, _lane_thresholds(gt, y_samples)):
        accs = [_line_accuracy(p, g, thresh) for p in pred]
        max_acc = np.max(accs) if accs else 0.0
        if max_acc < LANE_PT_THRESH:
            fn += 1
        else:
            matched += 1
        line_accs.append(max_acc)
    fp = len(pred) - matched
    if len(gt) > 4 and fn > 0:
        fn -= 1
    s = sum(line_accs)
    if len(gt) > 4:
        s -= min(line_accs)
    return (float(s / max(min(4.0, len(gt)), 1.0)),
            float(fp / len(pred)) if pred else 0.0,
            float(fn / max(min(len(gt), 4.0), 1.0)))


def match_generated_lanes(pred, gt, y_samples):
    """생성 차선과 GT 차선을 1:1로 매칭한다. pred·gt는 y_samples 위치의 x 배열 목록 (없는 지점 -2).

    - 생성 점: 생성 차선이 y_samples에서 값을 갖는 지점. 같은 y의 GT 차선과 x 차이가
      20px / cos θ (θ: GT 차선 기울기, TuSimple 기준) 미만이면 맞은 점이다.
    - 생성 차선의 점 정확도(맞은 점 / 생성 점)가 0.85 이상인 쌍만 후보로 두고, 점 정확도가
      높은 쌍부터 차선마다 한 번씩만 매칭한다. 매칭된 생성 차선이 정답 차선이다.
    - 생성 점이 하나도 없는 차선(GT가 차선이 없다고 한 범위에만 있는 차선)은 틀린 차선이다.

    반환: (매칭 쌍 [(생성 i, GT j, 맞은 점 수)], 생성 차선별 점 수, GT 차선별 점 수)
    """
    y_samples = np.asarray(y_samples, dtype=float)
    P = np.asarray(pred, dtype=float).reshape(len(pred), len(y_samples))
    G = np.asarray(gt, dtype=float).reshape(len(gt), len(y_samples))
    p_pts, g_pts = (P >= 0).sum(1), (G >= 0).sum(1)
    if not len(P) or not len(G):
        return [], p_pts, g_pts
    thr = _lane_thresholds(G, y_samples)
    both = (P[:, None, :] >= 0) & (G[None, :, :] >= 0)
    close = both & (np.abs(P[:, None, :] - G[None, :, :]) < thr[None, :, None])
    hits = close.sum(2)                                            # (생성, GT) 쌍별 맞은 점 수
    acc = hits / np.maximum(p_pts, 1)[:, None]
    pairs = sorted(((acc[i, j], hits[i, j], i, j) for i in range(len(P)) for j in range(len(G))
                    if p_pts[i] and acc[i, j] >= LANE_PT_THRESH), reverse=True)
    used_p, used_g, matches = set(), set(), []
    for _, h, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matches.append((i, j, int(h)))
    return matches, p_pts, g_pts


def evaluate_lanes(per_image):
    """per_image: {stem: {"pred": [[(x, y), ...]], "gt": load_tusimple_lanes()의 레코드}}

    주 지표 (생성 결과 기준, 전체 차선 합산)
      precision       생성 차선 정확도 = 정답 생성 차선 / 생성 차선
      recall          정답 GT 차선 중 정답 생성 차선과 매칭된 비율
      point_precision 맞은 생성 점 / 전체 생성 점 (정답 차선의 맞은 점만 셈)
    참고 지표 tusimple: 공식 LaneEval의 Accuracy·FP·FN (이미지별 값의 평균, bench_one_submit과 같음)
    평가는 GT의 h_samples 범위 안에서 한다. GT 차선이 없는 이미지의 생성 차선은 모두 틀린 차선이다.
    """
    acc = fp = fn = 0.0
    n_gt = n_pred = n_penalized = 0
    n_correct = hit_pts = pred_pts = gt_pts = 0
    n_official = 0
    for d in per_image.values():
        h, gt_lanes = d["gt"]["h_samples"], d["gt"]["lanes"]
        pred_lanes = [_pred_lane_xs(lane, h) for lane in d["pred"]]
        matches, p_pts, g_pts = match_generated_lanes(pred_lanes, gt_lanes, h)
        n_correct += len(matches)
        hit_pts += sum(m[2] for m in matches)
        pred_pts += int(p_pts.sum())
        gt_pts += int(g_pts.sum())
        n_gt += len(gt_lanes)
        n_pred += len(pred_lanes)
        if gt_lanes:                              # 공식 지표는 GT 차선이 없는 이미지를 정의하지 않는다
            a, p, n = tusimple_bench(pred_lanes, gt_lanes, h)
            acc += a
            fp += p
            fn += n
            n_official += 1
            n_penalized += len(gt_lanes) + 2 < len(pred_lanes)

    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_gt if n_gt else 0.0
    return {
        "n_images": len(per_image),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "point_precision": hit_pts / pred_pts if pred_pts else 0.0,
        "point_recall": hit_pts / gt_pts if gt_pts else 0.0,
        "n_pred_lanes": n_pred,
        "n_correct_lanes": n_correct,
        "n_gt_lanes": n_gt,
        "n_empty_gt_images": len(per_image) - n_official,
        "tusimple": {
            "n_images": n_official,
            "accuracy": acc / n_official if n_official else 0.0,
            "FP": fp / n_official if n_official else 0.0,
            "FN": fn / n_official if n_official else 0.0,
            "n_penalized_images": n_penalized,
        },
        "pixel_threshold": LANE_PIXEL_THRESH,
        "match_threshold": LANE_PT_THRESH,
    }


# --------------------------------------------------------------------------
# 예측(JSON) -> 내부 표현
# --------------------------------------------------------------------------

def predictions_from_auto_labeling(auto):
    """ui.build_auto_labeling() 결과를 평가용 구조로 바꾼다."""
    det = [
        {
            "cls": b["class"],
            "bbox": (b["bbox_x1"], b["bbox_y1"], b["bbox_x2"], b["bbox_y2"]),
            "score": b.get("confidence"),
        }
        for b in auto.get("Detection_information", {}).get("bbox_info", [])
    ]
    lanes = [
        [(float(x), float(y)) for x, y in lane.get("points", [])]
        for lane in auto.get("Lane_Detection_information", {}).get("lane_info", [])
    ]
    return {
        "detection": det,
        "weather": auto.get("Weather_information", {}).get("class"),
        "time": auto.get("Time_information", {}).get("class"),
        "lanes": lanes,
    }
