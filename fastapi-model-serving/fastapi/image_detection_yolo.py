import io
import torch
from PIL import Image, ImageDraw
import json
import base64
from ultralytics import YOLO
from glob import glob
import cv2
import numpy as np
import os

# KITTI+nuScenes(전방)로 1280 해상도에서 학습한 모델. 학습과 같은 해상도로 추론해야 성능이 나온다
DET_WEIGHTS = "yolo11x_finetuned_kitti_nusc_1280_vis40_ep20.pt"
DET_IMGSZ = 1280

def get_image_detector_yolo(device = 'cpu'):
    # model = YOLOv10.from_pretrained('jameslahm/yolov10m')
    model = YOLO(DET_WEIGHTS)
    model = model.to(device)
    return model

def numpy_to_json(array):
    # NumPy 배열을 바이트로 변환
    array_bytes = array.tobytes()
    # 바이트를 Base64로 인코딩
    array_b64 = base64.b64encode(array_bytes).decode('utf-8')
    # 배열의 형상과 dtype을 함께 저장
    return json.dumps({
        'shape': array.shape,
        'dtype': str(array.dtype),
        'data': array_b64
    })
    
def get_image_detections_yolo(model, binary_image, threshold=0.5, draw=True):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    
    # Disable augmentation for consistent inference results
    # threshold 미만 검출은 라벨로 내보내지 않는다(인증 평가의 생성 라벨 기준 conf 0.5)
    result = model.predict(input_image, augment=False, imgsz=DET_IMGSZ, conf=threshold)
    json_results = json.loads(result[0].to_json())
    ret_list = []
    for json_result in json_results:
        # print("json_result: ", json_result)
        name = json_result['name']
        x1 = json_result["box"]['x1']
        y1 = json_result["box"]['y1']
        x2 = json_result["box"]['x2']
        y2 = json_result["box"]['y2']
        # confidence는 mAP 계산 시 검출 결과를 점수순으로 정렬하는 데 필요하고,
        # 사람이 검수 대상을 고를 때의 기준이기도 하다. 6번째 원소로 덧붙인다.
        conf = json_result.get('confidence')
        ret_list.append([name, x1, y1, x2, y2, conf])

    # 좌표만 필요한 호출(include_image=false)은 결과 이미지를 그리지 않는다
    if not draw:
        return None, ret_list

    bboxes = result[0].boxes.xyxy.detach().cpu().numpy().astype(int)
    output_image = np.array(input_image)
    for x1, y1, x2, y2 in bboxes:
        output_image = cv2.rectangle(output_image, (x1, y1), (x2, y2), 127, 5)
    output_image = Image.fromarray(output_image)
    return output_image, ret_list