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

def get_image_detector_yolo(device = 'cpu'):
    # model = YOLOv10.from_pretrained('jameslahm/yolov10m')
    model = YOLO("yolo11x.pt")
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
    
def get_image_detections_yolo(model, binary_image, threshold=0.5):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    
    result = model.predict(input_image)
    json_results = json.loads(result[0].to_json())
    ret_list = []
    for json_result in json_results:
        # print("json_result: ", json_result)
        name = json_result['name']
        x1 = json_result["box"]['x1']
        y1 = json_result["box"]['y1']
        x2 = json_result["box"]['x2']
        y2 = json_result["box"]['y2']
        ret_list.append([name, x1, y1, x2, y2])
    
    bboxes = result[0].boxes.xyxy.detach().cpu().numpy().astype(int)
    input_image_array = np.array(input_image)
    output_image = input_image_array.copy()
    for bbox in bboxes:
        random_color_tuple = tuple(np.random.choice(range(256), size=3))
        x1, y1, x2, y2 = bbox
        # print(output_image.shape)
        # print(output_image.dtype)
        # print(x1, y1, x2, y2)
        # print(tuple(random_color_tuple))
        output_image = cv2.rectangle(output_image, (x1, y1), (x2, y2), 127, 5)
    output_image = Image.fromarray(output_image)
    return output_image, ret_list