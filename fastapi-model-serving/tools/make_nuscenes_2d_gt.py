# 실행 환경: pip install nuscenes-devkit==1.2.0  (자동 라벨링 툴 실행에는 필요 없음)
# 사용법: python make_nuscenes_2d_gt.py <nuScenes 루트(v1.0-trainval 포함)> <출력 JSON>
# 결과물: Eval/nuScenes/annotations/nuscenes_val_CAM_FRONT_devkit2d_official_vis60.coco.json
"""nuScenes val CAM_FRONT 2D GT를 공식 devkit 함수로 생성한다.

- 투영: nuscenes.scripts.export_2d_annotations_as_json.get_2d_boxes 를 수정 없이 호출
- 클래스: nuscenes.eval.detection.utils.category_to_detection_name (공식 검출 벤치마크 10종)
- 공식 검출 평가 필터(nuscenes.eval.common.loaders.add_center_dist / filter_eval_boxes, detection_cvpr_2019의
  class_range)를 통과하고 가시도 3·4(60% 이상)인 객체만 평가 대상, 나머지와 벤치마크 외 클래스는 iscrowd=1(무시 영역)
메모리 제약 때문에 NuScenes() 전체 로드 대신 필요한 레코드만 스트리밍으로 읽어 같은 인터페이스로 넘긴다.
"""
import json, sys, time, importlib.metadata as md
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
import nuscenes.scripts.export_2d_annotations_as_json as X

ROOT, OUT = sys.argv[1], sys.argv[2]
META = f"{ROOT}/v1.0-trainval"
EVAL_VIS = {"3", "4"}
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

def records(path, want=None):
    """들여쓰기된 nuScenes 테이블을 레코드 단위로 읽는다. want(bytes)가 False면 json 파싱을 건너뛴다."""
    buf, depth = [], 0
    with open(path, "rb") as f:
        for line in f:
            s = line.strip()
            if depth == 0 and s in (b"[", b"]", b""):
                continue
            depth += line.count(b"{") - line.count(b"}")
            buf.append(line)
            if depth == 0:
                raw = b"".join(buf).rstrip().rstrip(b",")
                buf = []
                if want is None or want(raw):
                    yield json.loads(raw)

# 공식 val 분할 (설치된 devkit 패키지의 splits 모듈)
from nuscenes.utils.splits import val as OFFICIAL_VAL
val_scenes = set(OFFICIAL_VAL)
scene_name = {s["token"]: s["name"] for s in json.load(open(f"{META}/scene.json"))}
samples = {s["token"]: s for s in json.load(open(f"{META}/sample.json")) if scene_name[s["scene_token"]] in val_scenes}
for s in samples.values():
    s["anns"] = []
log(f"val sample {len(samples)}")

cam, lidar = {}, {}
for r in records(f"{META}/sample_data.json", lambda b: b'"samples/CAM_FRONT/' in b or b'"samples/LIDAR_TOP/' in b):
    if r["sample_token"] in samples and r["is_key_frame"]:
        if "/CAM_FRONT/" in r["filename"]:
            r["sensor_modality"] = "camera"       # devkit 역색인이 붙여주는 필드
            cam[r["token"]] = r
        else:
            lidar[r["token"]] = r
            samples[r["sample_token"]]["data"] = {"LIDAR_TOP": r["token"]}   # add_center_dist가 쓰는 역색인
log(f"CAM_FRONT keyframe {len(cam)}, LIDAR_TOP keyframe {len(lidar)}")
need_pose = {r["ego_pose_token"] for r in cam.values()} | {r["ego_pose_token"] for r in lidar.values()}
poses = {r["token"]: r for r in records(f"{META}/ego_pose.json",
                                          lambda b: b[b.find(b'"token": "') + 10:][:32].decode() in need_pose)}
cs = {r["token"]: r for r in json.load(open(f"{META}/calibrated_sensor.json"))}
cat_name = {c["token"]: c["name"] for c in json.load(open(f"{META}/category.json"))}
inst_cat = {i["token"]: cat_name[i["category_token"]] for i in json.load(open(f"{META}/instance.json"))}
attr_name = {a["token"]: a["name"] for a in json.load(open(f"{META}/attribute.json"))}
log(f"ego_pose {len(poses)}")

anns = {}
for r in records(f"{META}/sample_annotation.json"):
    if r["sample_token"] in samples:
        r["category_name"] = inst_cat[r["instance_token"]]   # devkit 역색인이 붙여주는 필드
        anns[r["token"]] = r
        samples[r["sample_token"]]["anns"].append(r["token"])  # devkit과 같은 테이블 순서
log(f"val sample_annotation {len(anns)}")

class LiteNuScenes:
    """get_2d_boxes와 공식 평가 필터가 쓰는 nusc.get / nusc.get_box 만 제공한다."""
    tables = {"sample_data": {**cam, **lidar}, "sample": samples, "calibrated_sensor": cs, "ego_pose": poses,
              "sample_annotation": anns}
    def get(self, table, token):
        return self.tables[table][token]
    def get_box(self, token):
        r = anns[token]
        return Box(r["translation"], r["size"], Quaternion(r["rotation"]), name=r["category_name"], token=r["token"])

X.nusc = LiteNuScenes()

# 공식 검출 평가와 같은 GT 필터. 박스는 nuscenes.eval.detection.evaluate.load_gt와 같이 만들되,
# 필터에 쓰이지 않는 velocity는 계산하지 않는다
CLASS_RANGE = config_factory("detection_cvpr_2019").class_range
gt_boxes, box_token = EvalBoxes(), {}
for st, smp in samples.items():
    boxes = []
    for t in smp["anns"]:
        r = anns[t]
        det = category_to_detection_name(r["category_name"])
        if det is None:
            continue
        b = DetectionBox(sample_token=st, translation=r["translation"], size=r["size"], rotation=r["rotation"],
                         num_pts=r["num_lidar_pts"] + r["num_radar_pts"], detection_name=det)
        box_token[id(b)] = t
        boxes.append(b)
    gt_boxes.add_boxes(st, boxes)
add_center_dist(X.nusc, gt_boxes)
ego_dist = {box_token[id(b)]: b.ego_dist for st in gt_boxes.sample_tokens for b in gt_boxes[st]}
filter_eval_boxes(X.nusc, gt_boxes, CLASS_RANGE, verbose=True)
official = {box_token[id(b)] for st in gt_boxes.sample_tokens for b in gt_boxes[st]}
log(f"공식 필터 통과 {len(official)} / {len(ego_dist)} (val 전체 sample, 360도)")
images, out_anns = [], []
vis_count = {}
for sd_token in sorted(cam, key=lambda t: cam[t]["filename"]):
    sd = cam[sd_token]
    images.append({"file_name": sd["filename"], "id": sd_token, "token": sd["sample_token"],
                   "width": sd["width"], "height": sd["height"],
                   "cam_intrinsic": cs[sd["calibrated_sensor_token"]]["camera_intrinsic"]})
    for rec in X.get_2d_boxes(sd_token, ["", "1", "2", "3", "4"]):    # 전부 투영한 뒤 가시도로 평가/무시를 나눈다
        x1, y1, x2, y2 = rec["bbox_corners"]
        det = category_to_detection_name(rec["category_name"])
        vis = rec["visibility_token"]
        attrs = [attr_name[t] for t in rec["attribute_tokens"]]
        ok = rec["sample_annotation_token"] in official
        crowd = int(det is None or vis not in EVAL_VIS or not ok)
        vis_count[(vis, ok, crowd)] = vis_count.get((vis, ok, crowd), 0) + 1
        out_anns.append({
            "id": len(out_anns), "image_id": sd_token, "file_name": sd["filename"],
            "category_name": det or rec["category_name"], "category_full": rec["category_name"],
            "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": crowd,
            "visibility": vis, "attribute_name": attrs[0] if attrs else None, "attribute_names": attrs,
            "num_lidar_pts": rec["num_lidar_pts"], "num_radar_pts": rec["num_radar_pts"],
            "ego_dist": ego_dist.get(rec["sample_annotation_token"]), "official_filter": ok,
            "sample_annotation_token": rec["sample_annotation_token"], "instance_token": rec["instance_token"],
        })
DET = ["car", "truck", "trailer", "bus", "construction_vehicle", "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"]
json.dump({
    "info": {
        "description": "nuScenes v1.0-trainval 공식 val 분할 CAM_FRONT keyframe 2D GT",
        "method": "nuscenes-devkit %s export_2d_annotations_as_json.get_2d_boxes (3D 박스 투영, 수정 없이 호출)" % md.version("nuscenes-devkit"),
        "class_mapping": "nuscenes.eval.detection.utils.category_to_detection_name (공식 검출 벤치마크 10종)",
        "official_filter": "nuscenes.eval.common.loaders.add_center_dist + filter_eval_boxes, detection_cvpr_2019 "
                           "class_range %s (자차 xy 거리, LiDAR+레이더 점 0개 제외, 자전거 거치대 안 bicycle·motorcycle 제외)"
                           % CLASS_RANGE,
        "evaluated": "벤치마크 10종이면서 공식 필터를 통과하고 visibility_token 3, 4 (가시도 60% 이상)인 객체 (iscrowd=0)",
        "ignored": "벤치마크 외 클래스, 공식 필터 탈락 또는 가시도 60% 미만 (iscrowd=1, 무시 영역)",
    },
    "images": images, "annotations": out_anns,
    "categories": [{"id": i, "name": n} for i, n in enumerate(DET)],
}, open(OUT, "w"), ensure_ascii=False)
log(f"완료: 이미지 {len(images)}, 박스 {len(out_anns)}, 평가 {sum(1 for a in out_anns if not a['iscrowd'])}")
print("(가시도, 공식 필터 통과, 무시여부): 개수", dict(sorted(vis_count.items())))
