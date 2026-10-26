"""평가 결과 시각화: 모델 예측과 GT를 한 이미지에 겹쳐 그리고 zip으로 묶는다.

Streamlit과 분리돼 있어 단독 테스트가 가능하다. 이미지 안의 글자는 폰트 의존성을 줄이려고 영문만 쓴다.
"""

import collections
import csv
import io
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

import evaluation as E

COLORS = {
    "TP": (30, 144, 255),      # 맞춘 예측
    "FP": (255, 40, 40),       # 오검출
    "FN": (255, 190, 0),       # 놓친 GT
    "GT": (0, 200, 0),         # 예측과 매칭된 GT
    "IGN": (150, 150, 150),    # 무시 영역(DontCare·미매핑 클래스)과 그 안의 예측
    "PRED": (30, 144, 255),    # GT가 없는 이미지의 예측
    "LANE_PRED": (0, 255, 255),
    "LANE_GT": (255, 0, 255),
}
LEGEND = [("TP", "TP"), ("FP", "FP"), ("FN", "FN (missed)"), ("GT", "GT"),
          ("IGN", "ignored"), ("LANE_PRED", "lane"), ("LANE_GT", "lane GT")]

SUMMARY_FIELDS = ["file", "n_gt", "n_pred", "TP", "FP", "FN", "ignored_pred",
                  "weather_pred", "weather_gt", "weather_ok", "time_pred", "time_gt", "time_ok",
                  "n_lanes_pred", "n_lanes_gt", "error"]

_FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "DejaVuSans.ttf", "arial.ttf")
_local = threading.local()


def _font(size):
    # FreeType 글꼴 객체는 스레드 간 공유가 안전하지 않아 스레드마다 따로 만든다
    cache = getattr(_local, "fonts", None)
    if cache is None:
        cache = _local.fonts = {}
    if size not in cache:
        font = None
        for path in _FONT_PATHS:
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:
                font = ImageFont.load_default()
        cache[size] = font
    return cache[size]


def _text_size(draw, text, font):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def _label(draw, xy, text, color, font, img_w, img_h, above=True):
    """배경을 깐 라벨. 위쪽에 공간이 없으면 박스 안쪽에 그린다."""
    tw, th = _text_size(draw, text, font)
    pad = max(2, th // 6)
    x, y = xy
    y = y - th - 2 * pad if above and y - th - 2 * pad >= 0 else (y if above else y - th - 2 * pad)
    x = min(max(0, x), max(0, img_w - tw - 2 * pad))
    y = min(max(0, y), max(0, img_h - th - 2 * pad))
    draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=color)
    draw.text((x + pad, y + pad - 1), text, fill=(0, 0, 0) if sum(color) > 400 else (255, 255, 255), font=font)


def _clip_box(b, w, h):
    """이미지 안으로 자르고 좌표 순서를 정렬한다. PIL은 x2 < x1 이면 예외를 던진다."""
    x1, x2 = sorted((b[0], b[2]))
    y1, y2 = sorted((b[1], b[3]))
    return [max(0, min(w - 1, x1)), max(0, min(h - 1, y1)), max(0, min(w - 1, x2)), max(0, min(h - 1, y2))]


def classification_rows(pred, gt, stem, keys):
    """{키: (이름, 생성 라벨, GT, 정답 여부 or None)}. keys에 든 분류 항목만 만든다."""
    rows = {}
    for name, key in (("Weather", "weather"), ("Time", "time")):
        if key in keys:
            p, g = pred.get(key), gt[key].get(stem)
            rows[key] = (name, p, g, None if (p is None or g is None) else (p == g))
    return rows


def render(image_bytes, pred, gt_det=None, iou_thr=0.5, cls_rows=(), lanes_pred=(), lanes_gt=(), title=""):
    """예측(pred)과 GT 박스(gt_det, None이면 GT 없음)를 겹쳐 그린 PIL 이미지와 집계를 돌려준다."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size
    scale = max(W, H) / 1600.0
    lw = max(2, round(3 * scale))
    font = _font(max(12, round(17 * scale)))
    hfont = _font(max(13, round(20 * scale)))
    draw = ImageDraw.Draw(img)

    if gt_det is not None:
        p_st, g_st = E.match_image_detections(pred, gt_det, iou_thr)
    else:
        p_st, g_st = ["PRED"] * len(pred), []

    # 차선 -> GT 박스 -> 예측 박스 순으로 그려 예측이 맨 위에 오게 한다
    for lanes, key in ((lanes_gt, "LANE_GT"), (lanes_pred, "LANE_PRED")):
        for lane in lanes:
            pts = [(float(x), float(y)) for x, y in lane]
            if len(pts) >= 2:
                draw.line(pts, fill=COLORS[key], width=lw)

    for o, st in zip(gt_det or (), g_st):
        box = _clip_box(o["bbox"], W, H)
        color = COLORS["GT" if st == "TP" else st]
        draw.rectangle(box, outline=color, width=1 if st == "IGN" else (lw if st == "FN" else max(1, lw - 1)))
        if st == "FN":
            _label(draw, (box[0], box[3]), f"FN {o['cls']}", color, font, W, H, above=False)

    for o, st in zip(pred, p_st):
        box = _clip_box(o["bbox"], W, H)
        color = COLORS[st]
        draw.rectangle(box, outline=color, width=max(1, lw - 1) if st == "IGN" else lw)
        score = f" {o['score']:.2f}" if o.get("score") is not None else ""
        _label(draw, (box[0], box[1]), f"{'FP ' if st == 'FP' else ''}{o['cls']}{score}", color, font, W, H)

    stats = {
        "n_gt": sum(1 for s in g_st if s != "IGN"),
        "n_pred": len(pred),
        "TP": p_st.count("TP"),
        "FP": p_st.count("FP"),
        "FN": g_st.count("FN"),
        "ignored_pred": p_st.count("IGN"),
        "has_gt": gt_det is not None,
    }

    # ---- 상단 헤더: 제목/집계, 분류 결과, 범례 --------------------------------
    _, lh = _text_size(draw, "Ag", hfont)
    pad = max(6, lh // 2)
    lines = 3 if cls_rows else 2
    header_h = pad + lines * (lh + pad)
    canvas = Image.new("RGB", (W, H + header_h), (25, 25, 25))
    canvas.paste(img, (0, header_h))
    hd = ImageDraw.Draw(canvas)
    y = pad

    det_txt = (f"IoU {iou_thr:.2f}   TP {stats['TP']}   FP {stats['FP']}   FN {stats['FN']}"
               if stats["has_gt"] else f"no detection GT   pred {stats['n_pred']}")
    hd.text((pad, y), f"{title}    |    {det_txt}", fill=(255, 255, 255), font=hfont)
    y += lh + pad

    if cls_rows:
        x = pad
        for name, p, g, ok in cls_rows:
            seg = f"{name}: pred {p or '-'} / GT {g or '-'} "
            hd.text((x, y), seg, fill=(255, 255, 255), font=hfont)
            x += _text_size(hd, seg, hfont)[0]
            mark, col = ("[O]", (0, 220, 0)) if ok else (("[X]", (255, 70, 70)) if ok is False else ("[-]", (170, 170, 170)))
            hd.text((x, y), mark, fill=col, font=hfont)
            x += _text_size(hd, mark + "      ", hfont)[0]
        y += lh + pad

    x = pad
    for key, text in LEGEND:
        hd.rectangle([x, y + lh * 0.15, x + lh * 0.8, y + lh * 0.95], fill=COLORS[key])
        x += int(lh * 1.1)
        hd.text((x, y), text, fill=(220, 220, 220), font=hfont)
        x += _text_size(hd, text + "    ", hfont)[0]
    return canvas, stats


def make_renderer(preds, gt, sources, tasks, iou_thr=0.5, quality=90):
    """stem -> (zip 안 파일명, JPEG 바이트, 요약 행) 을 돌려주는 함수를 만든다.

    sources: {stem: (원래 파일명, 이미지 경로)}
    gt     : 평가 탭의 GT 묶음 {"detection": {...}, "weather": {...}, "time": {...}, "lanes": {stem: TuSimple 레코드}}
    tasks  : 평가한 항목("Detection", "Weather", "Time", "Lane"). 이 항목의 생성 라벨만 그린다.
    요약 행에서 해당 없는 칸은 ""로 둔다(summary.csv의 빈 칸).
    """
    det_gt, lane_gt = gt["detection"], gt["lanes"]
    cls_keys = tuple(k for k, task in (("weather", "Weather"), ("time", "Time")) if task in tasks)

    def _render(stem):
        name, path = sources[stem]
        p = preds[stem]
        cls = classification_rows(p, gt, stem, cls_keys)
        lanes_pred = (p.get("lanes") or ()) if "Lane" in tasks else ()
        lanes_gt = E.tusimple_lane_points(lane_gt[stem]) if stem in lane_gt else ()
        with open(path, "rb") as f:
            data = f.read()
        image, stats = render(
            data, p["detection"] if "Detection" in tasks else [],
            gt_det=det_gt.get(stem),
            iou_thr=iou_thr,
            cls_rows=list(cls.values()),
            lanes_pred=lanes_pred,
            lanes_gt=lanes_gt,
            title=name)
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=quality)
        row = {"file": name, "n_pred": stats["n_pred"], "error": "",
               "n_lanes_pred": len(lanes_pred) if "Lane" in tasks else "",
               "n_lanes_gt": len(lanes_gt) if stem in lane_gt else ""}
        row.update({k: stats[k] if stats["has_gt"] else "" for k in ("n_gt", "TP", "FP", "FN", "ignored_pred")})
        for key in ("weather", "time"):
            _, pred_label, gt_label, ok = cls.get(key, (None, None, None, None))
            row.update({f"{key}_pred": pred_label or "", f"{key}_gt": gt_label or "",
                        f"{key}_ok": "" if ok is None else int(ok)})
        return f"images/{os.path.splitext(name)[0]}.jpg", buf.getvalue(), row

    return _render


def zip_readme(settings):
    lines = [
        "Auto-labeling evaluation - visualization",
        "",
        "Box colors",
        "  blue   TP: prediction matched to a GT box (IoU >= threshold, same class)",
        "  red    FP: prediction with no matching GT",
        "  yellow FN: GT box the model missed",
        "  green  GT box that was matched",
        "  gray   ignored region (KITTI DontCare / unmapped nuScenes class) and predictions inside it",
        "  cyan   predicted lane, magenta GT lane",
        "Header: [O] correct, [X] wrong, [-] no GT",
        "",
        "Settings",
    ]
    lines += [f"  {k}: {v}" for k, v in settings.items()]
    lines += ["", "summary.csv: per-image counts and classification results (UTF-8 with BOM)"]
    return "\n".join(lines) + "\n"


def build_zip(path, stems, render_fn, readme="", workers=8, progress=None):
    """stems 순서대로 시각화 이미지를 만들어 zip에 쓴다. 한 장 실패해도 계속 진행한다.

    렌더링은 스레드 여러 개로 돌리고 zip 쓰기와 진행률 갱신은 호출 스레드에서만 한다.
    대기 작업 수를 제한해 수천 장이어도 메모리에 결과가 쌓이지 않게 한다.
    """
    def safe(stem):
        try:
            return render_fn(stem)
        except Exception as exc:                       # noqa: BLE001 - 개별 이미지 실패는 기록만 한다
            return None, None, {"file": stem, "error": f"{type(exc).__name__}: {exc}"}

    tmp = path + ".part"
    rows, n_ok = [], 0
    it = iter(stems)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf, ThreadPoolExecutor(workers) as ex:
        # range를 앞에 둬야 한도에 도달했을 때 it에서 원소를 하나 더 꺼내 버리지 않는다
        pending = collections.deque(ex.submit(safe, s) for _, s in zip(range(workers * 4), it))
        done = 0
        while pending:
            arc, data, row = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(ex.submit(safe, nxt))
            if data is not None:
                zf.writestr(arc, data)
                n_ok += 1
            rows.append(row)
            done += 1
            if progress:
                progress(done, len(stems))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=SUMMARY_FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        zf.writestr("summary.csv", buf.getvalue().encode("utf-8-sig"))   # 엑셀에서 바로 열리도록 BOM
        if readme:
            zf.writestr("README.txt", readme)
    os.replace(tmp, path)
    return {"path": path, "n_ok": n_ok, "n_err": len(rows) - n_ok}
