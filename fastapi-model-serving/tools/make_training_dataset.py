"""검출 모델 재학습용 데이터셋(YOLO 형식)을 만든다.

구성 (KITTI A안, 원본 해상도 그대로)
  KITTI    기존 YOLO 분할의 train 5,984장을 train 90% / val 10%로 다시 나눈다(seed 42).
           인증 평가셋(기존 val 1,496장)은 쓰지 않는다.
  nuScenes 공식 train 700 scene의 CAM_FRONT keyframe. 50 scene을 val로 뗀다
           (도시·야간·비 층화, seed 42). 공식 val 150 scene(인증 평가셋)은 쓰지 않는다.
  BDD100K  공식 train 70,000장 중 20,000장(비 오는 이미지 전부 + 나머지 무작위, seed 42)을 train,
           공식 val 10,000장 중 2,000장(무작위, seed 42)을 val로 쓴다.

라벨 규칙은 README.txt에 함께 기록한다. 클래스 매핑을 바꿀 때는 --labels-only로 라벨만 다시 만든다.

실행 환경: nuscenes-devkit==1.2.0, Pillow, numpy
사용법:    python make_training_dataset.py --out /mnt/d/02_Dataset/04_Bigdata/Train [--labels-only]
"""

import argparse
import collections
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

NAS = "/home/young/AILabDataset3/01_Open_Dataset"
KITTI = f"{NAS}/03_KITTI"
NUSC = f"{NAS}/46_nuScenes"
BDD = f"{NAS}/26_BDD-100K"
EVAL = "/mnt/d/02_Dataset/04_Bigdata/Eval"     # 인증 평가셋: 섞이지 않았는지 검사만 한다

SEED = 42
NAMES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
CID = {n: i for i, n in enumerate(NAMES)}

KITTI_VAL_RATIO = 0.10
NUSC_VAL_SCENES = 50
BDD_N_TRAIN, BDD_N_VAL = 20000, 2000

# nuScenes 공식 검출 클래스 -> KITTI 클래스 (평가 탭 evaluation.NUSCENES_TO_KITTI와 같은 방향)
# KITTI는 화물 트레일러·캐러밴을 Misc로 라벨링하므로 trailer도 Misc로 보낸다
NUSC_TRAIN = {"car": "Car", "truck": "Truck", "bus": "Truck", "trailer": "Misc",
              "construction_vehicle": "Misc", "pedestrian": "Pedestrian",
              "motorcycle": "Misc", "bicycle": "Cyclist"}
# 가시도 40% 이상만 정답으로 쓴다. 인증 평가 GT는 60% 이상만 평가하고 그 미만은 무시 영역이라
# 40~60%를 정답으로 학습해도 평가에서 손해가 없다. 40% 미만까지 넣으면(1280 실험) 많이 가려진 차에
# 박스를 쳐서 오검출이 늘었다
NUSC_VIS = {"2", "3", "4"}
# BDD100K 카테고리 -> KITTI 클래스. rider·bike·motor는 bdd_lines에서 따로 처리한다
BDD_DIRECT = {"car": "Car", "truck": "Truck", "bus": "Truck", "person": "Pedestrian", "train": "Tram"}

T0 = time.time()


def retry(fn, *args, tries=8, wait=15):
    """NAS(sshfs)가 재접속하는 동안 생기는 일시적 입출력 오류를 기다렸다가 다시 시도한다."""
    for k in range(tries):
        try:
            return fn(*args)
        except OSError as exc:
            if k == tries - 1:
                raise
            print(f"  입출력 오류, {wait}초 후 재시도 ({k + 1}/{tries}): {exc}", flush=True)
            time.sleep(wait)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def yolo_line(cls, box, w, h):
    x1, y1 = max(0.0, box[0]), max(0.0, box[1])
    x2, y2 = min(float(w), box[2]), min(float(h), box[3])
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return f"{CID[cls]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


# --------------------------------------------------------------------------
# 분할 선택
# --------------------------------------------------------------------------

def select_kitti():
    split_dir = f"{KITTI}/split_dataset_yolo/images"
    stems = sorted(os.path.splitext(f)[0] for f in retry(os.listdir, f"{split_dir}/train"))
    held_out = {os.path.splitext(f)[0] for f in retry(os.listdir, f"{split_dir}/val")}
    assert not set(stems) & held_out, "KITTI train과 인증 평가셋이 겹침"
    order = stems[:]
    random.Random(SEED).shuffle(order)
    val = set(order[:round(len(stems) * KITTI_VAL_RATIO)])
    return {"train": sorted(set(stems) - val), "val": sorted(val)}


def select_nuscenes(meta):
    from nuscenes.utils.splits import train as official_train
    scenes = json.load(open(f"{meta}/scene.json"))
    logs = {l["token"]: l for l in json.load(open(f"{meta}/log.json"))}
    train_scenes = sorted((s for s in scenes if s["name"] in set(official_train)), key=lambda s: s["name"])

    def stratum(s):
        d = s["description"].lower()
        return (logs[s["log_token"]]["location"].split("-")[0], "night" in d, "rain" in d)

    groups = collections.defaultdict(list)
    for s in train_scenes:
        groups[stratum(s)].append(s["name"])
    # 최대 잔여 방식으로 50 scene을 층별 비율대로 배분
    quota = {k: NUSC_VAL_SCENES * len(v) / len(train_scenes) for k, v in groups.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (quota[k] - alloc[k], k), reverse=True)[:NUSC_VAL_SCENES - sum(alloc.values())]:
        alloc[k] += 1
    rnd = random.Random(SEED)
    val = set()
    for k in sorted(groups):
        names = groups[k][:]
        rnd.shuffle(names)
        val.update(names[:alloc[k]])
    names = [s["name"] for s in train_scenes]
    return {"train": sorted(set(names) - val), "val": sorted(val)}, {k: (len(v), alloc[k]) for k, v in groups.items()}


def select_bdd(zf):
    def keys(split):
        return sorted(os.path.basename(n)[:-5] for n in zf.namelist()
                      if n.startswith(f"100k/{split}/") and n.endswith(".json"))
    train_keys = keys("train")
    rainy = [k for k in train_keys
             if json.loads(zf.read(f"100k/train/{k}.json")).get("attributes", {}).get("weather") == "rainy"]
    rest = sorted(set(train_keys) - set(rainy))
    random.Random(SEED).shuffle(rest)
    val_keys = keys("val")
    random.Random(SEED).shuffle(val_keys)
    return {"train": sorted(rainy + rest[:BDD_N_TRAIN - len(rainy)]), "val": sorted(val_keys[:BDD_N_VAL])}, len(rainy)


# --------------------------------------------------------------------------
# 라벨
# --------------------------------------------------------------------------

def kitti_lines(stem, w, h):
    lines = []
    for line in retry(read_bytes, f"{KITTI}/training/label_2/{stem}.txt").decode().splitlines():
        p = line.split()
        if len(p) < 8 or p[0] not in CID:          # DontCare 제외
            continue
        lines.append(yolo_line(p[0], tuple(map(float, p[4:8])), w, h))
    return [l for l in lines if l]


def bdd_lines(d, w, h):
    objs = [o for o in d["frames"][0]["objects"] if "box2d" in o]
    box = lambda o: (o["box2d"]["x1"], o["box2d"]["y1"], o["box2d"]["x2"], o["box2d"]["y2"])

    def overlap(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return ix * iy / small if small > 0 else 0.0

    riders = [o for o in objs if o["category"] == "rider"]
    vehicles = [o for o in objs if o["category"] in ("bike", "motor")]
    used, lines = set(), []
    # KITTI Cyclist는 사람+자전거를 한 박스로 본다. BDD100K는 rider와 bike/motor가 따로라 겹치는 쌍을 합친다
    for r in riders:
        rb = box(r)
        cands = [(overlap(rb, box(v)), i) for i, v in enumerate(vehicles) if i not in used]
        score, i = max(cands, default=(0.0, None))
        if i is not None and score >= 0.2:
            used.add(i)
            vb = box(vehicles[i])
            merged = (min(rb[0], vb[0]), min(rb[1], vb[1]), max(rb[2], vb[2]), max(rb[3], vb[3]))
            lines.append(yolo_line("Cyclist" if vehicles[i]["category"] == "bike" else "Misc", merged, w, h))
        else:
            lines.append(yolo_line("Cyclist", rb, w, h))
    for i, v in enumerate(vehicles):
        if i not in used and v["category"] == "motor":   # 사람 없는 자전거는 제외, 오토바이는 Misc
            lines.append(yolo_line("Misc", box(v), w, h))
    for o in objs:
        if o["category"] in BDD_DIRECT:
            lines.append(yolo_line(BDD_DIRECT[o["category"]], box(o), w, h))
    return [l for l in lines if l]


def records(path, want=None):
    """들여쓰기된 nuScenes 테이블을 레코드 단위로 스트리밍한다."""
    buf, depth = [], 0
    with open(path, "rb") as f:
        for line in f:
            if depth == 0 and line.strip() in (b"[", b"]", b""):
                continue
            depth += line.count(b"{") - line.count(b"}")
            buf.append(line)
            if depth == 0:
                raw = b"".join(buf).rstrip().rstrip(b",")
                buf = []
                if want is None or want(raw):
                    yield json.loads(raw)


def nuscenes_items(meta, scene_split):
    """공식 devkit get_2d_boxes를 수정 없이 호출해 CAM_FRONT 2D 라벨을 만든다."""
    import numpy as np
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import points_in_box
    from nuscenes.eval.detection.utils import category_to_detection_name
    import nuscenes.scripts.export_2d_annotations_as_json as X

    name_split = {n: sp for sp, names in scene_split.items() for n in names}
    scene_of = {s["token"]: name_split.get(s["name"]) for s in json.load(open(f"{meta}/scene.json"))}
    samples = {s["token"]: dict(s, anns=[]) for s in json.load(open(f"{meta}/sample.json")) if scene_of[s["scene_token"]]}
    cam = {}
    for r in records(f"{meta}/sample_data.json", lambda b: b'"samples/CAM_FRONT/' in b):
        if r["sample_token"] in samples and r["is_key_frame"]:
            r["sensor_modality"] = "camera"
            cam[r["token"]] = r
    need = {r["ego_pose_token"] for r in cam.values()}
    poses = {r["token"]: r for r in records(f"{meta}/ego_pose.json",
                                              lambda b: b[b.find(b'"token": "') + 10:][:32].decode() in need)}
    cs = {r["token"]: r for r in json.load(open(f"{meta}/calibrated_sensor.json"))}
    cat = {c["token"]: c["name"] for c in json.load(open(f"{meta}/category.json"))}
    inst = {i["token"]: cat[i["category_token"]] for i in json.load(open(f"{meta}/instance.json"))}
    attr = {a["token"]: a["name"] for a in json.load(open(f"{meta}/attribute.json"))}
    log(f"nuScenes: CAM_FRONT {len(cam):,}장, ego_pose {len(poses):,}")

    ann = {}                                        # 메모리를 아끼려고 필요한 필드만 튜플로 둔다
    for r in records(f"{meta}/sample_annotation.json"):
        if r["sample_token"] in samples:
            ann[r["token"]] = (r["sample_token"], r["instance_token"], r["visibility_token"],
                               tuple(r["attribute_tokens"]), tuple(r["translation"]), tuple(r["size"]),
                               tuple(r["rotation"]), r["num_lidar_pts"], r["num_radar_pts"])
            samples[r["sample_token"]]["anns"].append(r["token"])
    log(f"nuScenes: sample_annotation {len(ann):,}")

    def ann_rec(token):
        s, i, v, a, t, z, q, nl, nr = ann[token]
        return {"token": token, "sample_token": s, "instance_token": i, "visibility_token": v,
                "attribute_tokens": list(a), "translation": list(t), "size": list(z), "rotation": list(q),
                "num_lidar_pts": nl, "num_radar_pts": nr, "category_name": inst[i], "prev": "", "next": ""}

    class LiteNuScenes:
        def get(self, table, token):
            if table == "sample_annotation":
                return ann_rec(token)
            return {"sample_data": cam, "sample": samples, "calibrated_sensor": cs, "ego_pose": poses}[table][token]

        def get_box(self, token):
            r = ann_rec(token)
            return Box(r["translation"], r["size"], Quaternion(r["rotation"]), name=r["category_name"], token=token)

    X.nusc = LiteNuScenes()
    items, stats = [], collections.Counter()
    for sd_token in sorted(cam, key=lambda t: cam[t]["filename"]):
        sd = cam[sd_token]
        smp = samples[sd["sample_token"]]
        racks = [Box(ann[t][4], ann[t][5], Quaternion(ann[t][6])) for t in smp["anns"]
                 if inst[ann[t][1]] == "static_object.bicycle_rack"]
        lines = []
        for rec in X.get_2d_boxes(sd_token, ["", "1", "2", "3", "4"]):
            det = category_to_detection_name(rec["category_name"])
            cls = NUSC_TRAIN.get(det)
            if cls is None:
                continue
            names = [attr[t] for t in rec["attribute_tokens"]]
            if rec["visibility_token"] not in NUSC_VIS:
                stats["제외: 가시도 40% 미만"] += 1; continue
            if rec["num_lidar_pts"] + rec["num_radar_pts"] == 0:   # 공식 벤치마크·인증 평가 GT도 무시한다
                stats["제외: 센서 점 0개"] += 1; continue
            if det == "bicycle" and "cycle.with_rider" not in names:
                stats["제외: 사람 없는 자전거"] += 1; continue
            if det in ("bicycle", "motorcycle") and racks and any(
                    points_in_box(rb, np.array(ann[rec["sample_annotation_token"]][4])[:, None]).any() for rb in racks):
                stats["제외: 자전거 거치대"] += 1; continue
            line = yolo_line(cls, rec["bbox_corners"], sd["width"], sd["height"])
            if line:
                lines.append(line)
                stats[f"정답: {cls}"] += 1
        split = scene_of[smp["scene_token"]]
        items.append((split, "nuscenes", f"{NUSC}/{sd['filename']}", os.path.basename(sd["filename"]),
                      lines, (sd["width"], sd["height"])))
    # devkit 모듈 전역에 남은 참조를 끊어야 어노테이션 97만 개가 메모리에서 해제된다
    X.nusc = None
    ann.clear(); samples.clear(); cam.clear(); poses.clear()
    return items, stats


# --------------------------------------------------------------------------
# 복사·저장
# --------------------------------------------------------------------------

def copy_verify(src, dst):
    """원본을 그대로 복사하고 복사본을 다시 읽어 MD5를 대조한다. 이미 같은 파일이 있으면 건너뛴다."""
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        data = open(dst, "rb").read()
        return hashlib.md5(data).hexdigest(), Image.open(io.BytesIO(data)).size, True
    data = retry(read_bytes, src)
    md5 = hashlib.md5(data).hexdigest()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    with open(dst, "rb") as f:
        ok = hashlib.md5(f.read()).hexdigest() == md5
    if not ok:
        raise IOError(f"복사 검증 실패: {dst}")
    return md5, Image.open(io.BytesIO(data)).size, False


def yaml_files():
    """학습 설정 파일 이름 -> 내용. 경로는 yaml이 있는 폴더 기준이라 데이터셋을 옮겨도 그대로 쓴다."""
    names = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    return {
        "data.yaml": "# 데이터셋 루트는 이 파일이 있는 폴더다. 학습할 때 이 파일을 절대 경로로 지정할 것\n"
                     "train: images/train\nval: images/val\n" + names,
        "data_kitti_nuscenes.yaml": "# BDD100K를 뺀 KITTI+nuScenes 설정. 데이터셋 루트는 이 파일이 있는 폴더다\n"
                                    "train: [images/train/kitti, images/train/nuscenes]\n"
                                    "val: [images/val/kitti, images/val/nuscenes]\n" + names,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-only", action="store_true", help="이미지는 이미 복사돼 있다고 보고 라벨만 다시 만든다")
    args = ap.parse_args()
    out = args.out
    meta = f"{NUSC}/v1.0-trainval"

    # ---- 분할 ----
    kitti = select_kitti()
    nusc_split, nusc_strata = select_nuscenes(meta)
    # 라벨 zip은 7만 번 넘게 임의 접근하므로 NAS에서 직접 읽지 않고 로컬에 복사해 둔다
    local_zip = os.path.join(tempfile.gettempdir(), "bdd100k_labels.zip")
    src_zip = f"{BDD}/bdd100k_labels.zip"
    if not (os.path.exists(local_zip) and os.path.getsize(local_zip) == os.path.getsize(src_zip)):
        retry(shutil.copyfile, src_zip, local_zip)
    zf = zipfile.ZipFile(local_zip)
    bdd, n_rainy = select_bdd(zf)
    log(f"분할: KITTI {len(kitti['train'])}/{len(kitti['val'])}, nuScenes scene {len(nusc_split['train'])}/"
        f"{len(nusc_split['val'])}, BDD100K {len(bdd['train'])}/{len(bdd['val'])} (비 {n_rainy}장 전부 포함)")

    # ---- 인증 평가셋과 겹치지 않는지 ----
    eval_kitti = {l.strip() for l in open(f"{EVAL}/KITTI/val_list.txt") if l.strip()}
    eval_nusc = {os.path.splitext(l.strip())[0] for l in open(f"{EVAL}/nuScenes/val_list.txt") if l.strip()}
    assert not (set(kitti["train"]) | set(kitti["val"])) & eval_kitti, "KITTI 인증 평가셋 혼입"
    from nuscenes.utils.splits import val as official_val
    assert not (set(nusc_split["train"]) | set(nusc_split["val"])) & set(official_val), "nuScenes 공식 val 혼입"

    # ---- 항목 목록 (split, source, src, name, lines|None, size|None) ----
    items = [(sp, "kitti", f"{KITTI}/training/image_2/{s}.png", f"{s}.png", None, None)
             for sp in ("train", "val") for s in kitti[sp]]
    nusc_items, nusc_stats = nuscenes_items(meta, nusc_split)
    assert not {os.path.splitext(i[3])[0] for i in nusc_items} & eval_nusc, "nuScenes 인증 평가셋 혼입"
    items += nusc_items
    items += [(sp, "bdd100k", f"{BDD}/images/100k/{sp}/{k}.jpg", f"{k}.jpg", None, None)
              for sp in ("train", "val") for k in bdd[sp]]
    log(f"전체 {len(items):,}장")

    # ---- 이미지 복사 (원본 그대로) ----
    md5s, sizes = {}, {}
    def job(it):
        sp, srcname, src, name, _, _ = it
        rel = f"images/{sp}/{srcname}/{name}"
        dst = os.path.join(out, rel)
        if args.labels_only:
            data = retry(read_bytes, dst)          # /mnt/d(9p)는 메모리가 부족하면 읽기가 일시적으로 실패한다
            return rel, hashlib.md5(data).hexdigest(), Image.open(io.BytesIO(data)).size, True
        return (rel, *copy_verify(src, dst))
    skipped = 0
    with ThreadPoolExecutor(8) as ex:
        for n, (rel, md5, size, skip) in enumerate(ex.map(job, items), 1):
            md5s[rel], sizes[rel] = md5, size
            skipped += skip
            if n % 5000 == 0:
                log(f"이미지 {n:,}/{len(items):,}")
    log(f"이미지 {'확인' if args.labels_only else '복사·검증'} 완료 (기존 파일 재사용 {skipped:,}장)")

    # ---- 라벨 ----
    counts = collections.Counter()
    box_counts = collections.defaultdict(collections.Counter)
    for sp, srcname, src, name, lines, _ in items:
        rel = f"images/{sp}/{srcname}/{name}"
        w, h = sizes[rel]
        stem = os.path.splitext(name)[0]
        if srcname == "kitti":
            lines = kitti_lines(stem, w, h)
        elif srcname == "bdd100k":                  # 2만 개를 한꺼번에 올리지 않고 하나씩 읽는다
            lines = bdd_lines(json.loads(zf.read(f"100k/{sp}/{stem}.json")), w, h)
        lab_rel = f"labels/{sp}/{srcname}/{stem}.txt"
        os.makedirs(os.path.dirname(os.path.join(out, lab_rel)), exist_ok=True)
        text = "".join(l + "\n" for l in lines)
        with open(os.path.join(out, lab_rel), "w", newline="\n") as f:
            f.write(text)
        md5s[lab_rel] = hashlib.md5(text.encode()).hexdigest()
        counts[(sp, srcname)] += 1
        for l in lines:
            box_counts[(sp, srcname)][NAMES[int(l.split()[0])]] += 1
    log("라벨 작성 완료")

    # ---- 분할 목록·data.yaml·README ----
    os.makedirs(f"{out}/splits", exist_ok=True)
    lists = {"kitti_train.txt": kitti["train"], "kitti_val.txt": kitti["val"],
             "nuscenes_train_scenes.txt": nusc_split["train"], "nuscenes_val_scenes.txt": nusc_split["val"],
             "bdd100k_train.txt": bdd["train"], "bdd100k_val.txt": bdd["val"]}
    for fn, names in lists.items():
        with open(f"{out}/splits/{fn}", "w", newline="\n") as f:
            f.write("\n".join(names) + "\n")
    yamls = yaml_files()
    for fn, text in yamls.items():
        with open(f"{out}/{fn}", "w", newline="\n") as f:
            f.write(text)

    def table(sp):
        rows = []
        for src in ("kitti", "nuscenes", "bdd100k"):
            c = box_counts[(sp, src)]
            rows.append(f"  {src:9s} 이미지 {counts[(sp, src)]:6,}장  박스 {sum(c.values()):7,}개  "
                        + ", ".join(f"{n} {c[n]:,}" for n in NAMES if c[n]))
        return "\n".join(rows)

    readme = f"""검출 모델 재학습 데이터셋 (YOLO 형식, 원본 해상도)
==================================================
생성: {time.strftime('%Y-%m-%d %H:%M')}  /  생성 스크립트: auto-labeling-api 저장소 fastapi-model-serving/tools/make_training_dataset.py
클래스(8종, 자동 라벨링 툴 출력과 동일): {', '.join(f'{i}={n}' for i, n in enumerate(NAMES))}

[train]
{table('train')}

[val] 학습 중 체크포인트 선택용
{table('val')}

[분할 방법] 고정 시드 {SEED}, 목록은 splits/ 폴더
  KITTI    : 03_KITTI/split_dataset_yolo/images/train 5,984장을 train 90% / val 10%로 다시 분할
             (이미지·라벨은 원본 training/image_2, label_2에서 가져옴)
  nuScenes : 공식 train 700 scene의 CAM_FRONT keyframe. 50 scene을 val로 분리
             (도시·야간·비 층별 비율대로 배분)
  BDD100K  : 공식 train 70,000장 중 비 오는 이미지 {n_rainy:,}장 전부 + 나머지 무작위 = {BDD_N_TRAIN:,}장 → train
             공식 val 10,000장 중 무작위 {BDD_N_VAL:,}장 → val

[인증 평가셋과의 분리] 생성 스크립트가 아래가 섞이지 않았음을 검사함
  KITTI 인증 평가셋 1,496장(Eval/KITTI), nuScenes 공식 val 150 scene(Eval/nuScenes)

[라벨 규칙]
  KITTI    : 원본 label_2 그대로. DontCare 제외
  nuScenes : 공식 devkit 1.2.0 get_2d_boxes로 3D 박스를 투영(평가 GT와 같은 방식)
             car→Car, truck·bus→Truck, pedestrian→Pedestrian, bicycle(사람 탄 것만)→Cyclist,
             trailer·construction_vehicle·motorcycle→Misc (KITTI는 트레일러·캐러밴을 Misc로 라벨링)
             정답으로 쓰는 조건: 가시도 40% 이상, 센서 점 1개 이상, 자전거 거치대 밖
             (인증 평가 GT는 공식 필터를 통과하고 가시도 60% 이상인 객체만 평가, 나머지는 무시 영역)
             조건에 안 맞는 객체와 traffic_cone·barrier 등은 라벨에서 뺌(배경)
             제외 내역: {', '.join(f'{k} {v:,}' for k, v in sorted(nusc_stats.items()) if k.startswith('제외'))}
  BDD100K  : car→Car, truck·bus→Truck, person→Pedestrian, train→Tram
             rider+bike를 한 박스로 합쳐 Cyclist, rider+motor는 Misc, 짝 없는 rider는 Cyclist,
             짝 없는 motor는 Misc, 짝 없는 bike(사람 없는 자전거)와 신호등·표지판은 제외
  ※ nuScenes·BDD100K의 car에는 밴·SUV가 포함되어 KITTI의 Van과 경계가 다름
  ※ YOLO 학습은 무시 영역을 지원하지 않아 제외 객체는 단순히 라벨에서 뺌(이미지 가림 처리 없음)

[사용법]
  yolo train data=<이 폴더의 절대 경로>/data.yaml ...
  BDD100K 없이 KITTI+nuScenes만 쓸 때는 data=<이 폴더의 절대 경로>/data_kitti_nuscenes.yaml
  옮긴 뒤 검증: 이 폴더에서  md5sum -c --quiet checksums.md5
"""
    with open(f"{out}/README.txt", "w", newline="\n") as f:
        f.write(readme)
    for rel in list(yamls) + ["README.txt"] + [f"splits/{fn}" for fn in lists]:
        md5s[rel] = hashlib.md5(open(os.path.join(out, rel), "rb").read()).hexdigest()
    with open(f"{out}/checksums.md5", "w", newline="\n") as f:
        for rel in sorted(md5s):
            f.write(f"{md5s[rel]}  {rel}\n")
    log(f"완료: 파일 {len(md5s):,}개")
    print(readme)
    print("nuScenes 층화:", {str(k): v for k, v in sorted(nusc_strata.items())})


if __name__ == "__main__":
    sys.exit(main())
