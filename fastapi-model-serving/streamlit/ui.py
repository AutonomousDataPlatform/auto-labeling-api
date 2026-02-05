import io
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st

import json
import base64
import numpy as np
from fastapi import UploadFile
import os
import glob
import zipfile

# interact with FastAPI endpoint
# segmentation_backend = "http://fastapi:8000/segmentation"
# detection_backend = "http://fastapi:8000/detection"
# weather_classification_backend = "http://fastapi:8000/weather_classification"
def dict_to_numpy(json_str):
    # JSON 문자열을 딕셔너리로 파싱
    decoded = json.loads(json_str.content)
    # Base64로 인코딩된 데이터를 바이트로 디코딩
    # response to dict
    print("decoded: ", decoded)
    array_bytes = base64.b64decode(decoded['data'])
    # NumPy 배열로 변환
    return np.frombuffer(array_bytes, dtype=decoded['dtype']).reshape(decoded['shape'])

segmentation_backend = "http://localhost:8000/segmentation"
detection_yolo_backend = "http://localhost:8000/detection_yolo"
detection_gpt_backend = "http://localhost:8000/detection_gpt"
weather_classification_backend = "http://localhost:8000/weather_classification"
time_classification_backend = "http://localhost:8000/time_classification"
image_backend = "http://localhost:8000/image"
lane_detection_backend = "http://localhost:8001/lane_detection"

def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )
    return r

def process_image(uploade_file, server_url: str):
    file_bytes = uploade_file.getvalue()
    file_name = uploade_file.name
    m = MultipartEncoder(fields={"file": (file_name, file_bytes, "image/jpeg")})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )
    return r

def process_image_to_json(input_image, image_backend, weather_classification_backend, time_classification_backend, detection_yolov10_backend, lane_detection_backend, detection_gpt_backend):
    image_process = process_image(input_image, image_backend)
    image_result = image_process.content
    weather_process = process(input_image, weather_classification_backend)
    weather_result = weather_process.content
    time_process = process(input_image, time_classification_backend)
    time_result = time_process.content
    detection_process = process(input_image, detection_yolov10_backend)
    detection_result = detection_process.content
    lane_detection_process = process(input_image, lane_detection_backend)
    lane_detection_result = lane_detection_process.content
    detection_gpt_process = process(input_image, detection_gpt_backend)
    detection_gpt_result = detection_gpt_process.content

    if isinstance(weather_result, bytes):
        weather_result = weather_result.decode("utf-8")
    else:
        str(weather_result)
    if isinstance(time_result, bytes):
        time_result = time_result.decode("utf-8")
    else:
        str(time_result)
    if isinstance(detection_result, bytes):
        detection_result = detection_result.decode("utf-8")
    else:
        str(detection_result)
    if isinstance(lane_detection_result, bytes):
        lane_detection_result = lane_detection_result.decode("utf-8")
    else:
        str(lane_detection_result)
    if isinstance(detection_gpt_result, bytes):
        detection_gpt_result = detection_gpt_result.decode("utf-8")
    else:
        str(detection_gpt_result)

    image_data = json.loads(image_result)
    image_info = image_data["image_info"]
    weather_data = json.loads(weather_result)
    weather_class = weather_data["weather_class"]
    time_data = json.loads(time_result)
    time_class = time_data["time_class"]
    detection_data = json.loads(detection_result)
    detection_list = detection_data["detection_result"]
    lane_detection_data = json.loads(lane_detection_result)
    lane_detection_list = lane_detection_data["detection_result"]
    detection_gpt_data = json.loads(detection_gpt_result)
    detection_gpt_list = detection_gpt_data["detection_result"]
    
    structured_result = {
        "Original_calib": {},
        "Original_label": {},
        "Auto_labeling": {
            "Image_information": {
                "file_name": image_info["name"],
                "width": image_info["width"],
                "height": image_info["height"],
                "format": image_info["format"],
                "mode": image_info["mode"]
            },
            "Time_information": {
                "class": time_class
            },
            "Weather_information": {
                "class": weather_class
            },
            "Detection_information": {
                "num_of_bbox": len(detection_list),
                "bbox_info": []
            },
            "Detection_gpt_information": {
                "num_of_bbox": len(detection_gpt_list),
                "bbox_info": []
            },
            "Lane_Detection_information": {
                "num_of_lanes": len(lane_detection_list),
                "lane_info": []
            }
        },
    }
    
    for box in detection_list:
        class_label = box[0]
        x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
        
        structured_result["Auto_labeling"]["Detection_information"]["bbox_info"].append({
        "class": class_label,
        "type": "Bounding_box",
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2
    })
    for box in detection_gpt_list:
        class_label = box[0]
        x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
        
        structured_result["Auto_labeling"]["Detection_gpt_information"]["bbox_info"].append({
        "class": class_label,
        "type": "Bounding_box",
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2
    })
    for lane in lane_detection_list:
        structured_result["Auto_labeling"]["Lane_Detection_information"]["lane_info"].append({
        "type": "Line",
        "points": lane
    })
    return structured_result

def read_text_if_exists(path):
    """경로가 존재하면 텍스트 반환, 없으면 빈 문자열"""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [line.rstrip("\n") for line in f]

# construct UI layout
st.title("[BigData] Auto-Labeling Web Frontend")

st.write(
    """Obtain segmentation, detection, classification predictions from image inputs via models implemented in PyTorch.
         This Streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation."""
)  # description and instructions

folder_path = st.text_input("Folder path")
if st.button("File List") and folder_path:
    folder_path = os.path.normpath(folder_path.strip(' "\''))
    if not os.path.isdir(folder_path):
        st.error("Invalid folder path!")
        st.stop()

    # 하위 디렉터리까지 jpg·png 탐색
    pattern = os.path.join(folder_path, "**", "*.[pj][pn]g")   # *.jpg, *.png
    img_paths = glob.glob(pattern, recursive=True)

    if not img_paths:
        st.warning("No images")
    else:
        st.success(f"Found {len(img_paths)} images")
        # for p in sorted(img_paths):
        #     st.markdown(f"• `{os.path.relpath(p, folder_path)}`")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for p in img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            label_txt = os.path.join(folder_path, "label_2", stem + ".txt")
            calib_txt = os.path.join(folder_path, "calib", stem + ".txt")
            
            with open(p, 'rb') as f:
                img_bytes = f.read()
            file_like = io.BytesIO(img_bytes)
            file_like.name = os.path.basename(p)  # Set the name of the file-like
            structured_result = process_image_to_json(file_like, image_backend, weather_classification_backend, time_classification_backend, detection_yolo_backend, lane_detection_backend)
            
            structured_result["Original_calib"] = read_text_if_exists(calib_txt)
            structured_result["Original_label"] = read_text_if_exists(label_txt)
            
            json_str = json.dumps(structured_result, indent=4)
            json_filename = os.path.splitext(os.path.basename(p))[0] + ".json"
            zip_file.writestr(json_filename, json_str)
    zip_buffer.seek(0)
    
    st.download_button(
        label="Download JSON Results",
        data=zip_buffer,
        file_name="results.zip",
        mime="application/zip"
    )

input_image = st.file_uploader("Insert image")  # image upload widget

if input_image:
    image_process = process_image(input_image, image_backend)
    image_result = image_process.content
    weather_process = process(input_image, weather_classification_backend)
    weather_result = weather_process.content
    time_process = process(input_image, time_classification_backend)
    time_result = time_process.content
    detection_process = process(input_image, detection_yolo_backend)
    detection_result = detection_process.content
    detection_gpt_process = process(input_image, detection_gpt_backend)
    detection_gpt_result = detection_gpt_process.content
    lane_detection_process = process(input_image, lane_detection_backend)
    lane_detection_result = lane_detection_process.content

    if isinstance(weather_result, bytes):
        weather_result = weather_result.decode("utf-8")
    else:
        str(weather_result)
    if isinstance(time_result, bytes):
        time_result = time_result.decode("utf-8")
    else:
        str(time_result)
    if isinstance(detection_result, bytes):
        detection_result = detection_result.decode("utf-8")
    else:
        str(detection_result)
    if isinstance(lane_detection_result, bytes):
        lane_detection_result = lane_detection_result.decode("utf-8")
    else:
        str(lane_detection_result)
    if isinstance(detection_gpt_result, bytes):
        detection_gpt_result = detection_gpt_result.decode("utf-8")
    else:
        str(detection_gpt_result)

    image_data = json.loads(image_result)
    image_info = image_data["image_info"]
    weather_data = json.loads(weather_result)
    weather_class = weather_data["weather_class"]
    time_data = json.loads(time_result)
    time_class = time_data["time_class"]
    detection_data = json.loads(detection_result)
    detection_list = detection_data["detection_result"]
    detection_gpt_data = json.loads(detection_gpt_result)
    detection_gpt_list = detection_gpt_data["detection_gpt_result"]
    lane_detection_data = json.loads(lane_detection_result)
    lane_detection_list = lane_detection_data["detection_result"]

    structured_result = {
        "Image_information": {
            "file_name": image_info["name"],
            "width": image_info["width"],
            "height": image_info["height"],
            "format": image_info["format"],
            "mode": image_info["mode"]
        },
        "Time_information": {
            "class": time_class
        },
        "Weather_information": {
            "class": weather_class
        },
        "Detection_information": {
            "num_of_bbox": len(detection_list),
            "bbox_info": []
        },
        "Detection_gpt_information": {
            "num_of_bbox": len(detection_gpt_list),
            "bbox_info": []
        },
        "Lane_Detection_information": {
            "num_of_lanes": len(lane_detection_list),
            "lane_info": []
        }
    }
    
    for box in detection_list:
        class_label = box[0]
        x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
        
        structured_result["Detection_information"]["bbox_info"].append({
        "class": class_label,
        "type": "Bounding_box",
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2
    })
    for box in detection_gpt_list:
        class_label = box[0]
        x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
        
        structured_result["Detection_gpt_information"]["bbox_info"].append({
        "class": class_label,
        "type": "Bounding_box",
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2
    })
    for lane in lane_detection_list:
        structured_result["Lane_Detection_information"]["lane_info"].append({
        "type": "Line",
        "points": lane
    })
        
    # JSON 문자열로 변환합니다.
    json_data = json.dumps(structured_result, indent=4)
    st.download_button(
        label="Download JSON Results",
        data=json_data,
        file_name="results.json",
        mime="application/json"
    )
    
# if st.button("download JSON results"):
#     if input_image:
#         weather_process = process(input_image, weather_classification_backend)
#         weather_result = weather_process.content
#         detection_process = process(input_image, detection_yolov10_backend)
#         detection_result = detection_process.content
#         lane_detection_process = process(input_image, lane_detection_backend)
#         lane_detection_image = Image.open(io.BytesIO(lane_detection_process.content)).convert("RGB")
#         # lane_detection_process = process(input_image, lane_detection_backend)
#         # print("Response text:", lane_detection_process.text)
#         # lane_detection_data = lane_detection_process.json()
#         # lane_image = base64.b64decode(lane_detection_data["image"])
#         # lane_detection_image = Image.open(io.BytesIO(lane_image)).convert("RGB")
#         # lane = lane_detection_data["detection_result"]
        
#         results = {
#             "weather": weather_result.decode("utf-8") if isinstance(weather_result, bytes) else str(weather_result),
#             "detection": detection_result.decode("utf-8") if isinstance(detection_result, bytes) else str(detection_result),
#             # lane_detection 결과가 이미지이므로, base64 인코딩으로 저장할 수 있습니다.
#             # "lane_detection": base64.b64encode(lane_detection_process.content).decode("utf-8")
#         }
#         # JSON 문자열로 변환합니다.
#         json_data = json.dumps(results, indent=4)
        
#         # JavaScript를 이용해 자동 다운로드 실행 (파일 저장 위치 선택 창이 뜹니다)
#         download_js = f"""
#         <script>
#         var a = document.createElement('a');
#         a.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(`{json_data}`);
#         a.download = 'results.json';
#         document.body.appendChild(a);
#         a.click();
#         document.body.removeChild(a);
#         </script>
#         """
#         st.markdown(download_js, unsafe_allow_html=True)
#         st.success("JSON 결과가 다운로드됩니다.")
#     else:
#         st.error("Insert an image!")

if st.button("get total result"):
    col1, col2, col3, col4, col5 = st.columns(5)
    if input_image:
        weather_process = process(input_image, weather_classification_backend)
        weather_result = weather_process.content
        time_process = process(input_image, time_classification_backend)
        time_result = time_process.content
        
        detection_process = process(input_image, detection_yolo_backend)
        detection_result = detection_process.content
        if isinstance(detection_result, bytes):
            detection_result = detection_result.decode("utf-8")
        else:
            detection_result = str(detection_result)
            
        payload = json.loads(detection_result)
        detection_result = payload["detection_result"]
        
        detection_gpt_process = process(input_image, detection_gpt_backend)
        detection_gpt_result = detection_gpt_process.content
        if isinstance(detection_gpt_result, bytes):
            detection_gpt_result = detection_gpt_result.decode("utf-8")
        else:
            detection_gpt_result = str(detection_gpt_result)
            
        payload = json.loads(detection_gpt_result)
        detection_gpt_result = payload["detection_gpt_result"]
        
        lane_detection_process = process(input_image, lane_detection_backend)
        lane_detection_result = lane_detection_process.content
        if isinstance(lane_detection_result, bytes):
            lane_detection_result = lane_detection_result.decode("utf-8")
        else:
            lane_detection_result = str(lane_detection_result)

        lane_payload = json.loads(lane_detection_result)
        lane_detection_result = lane_payload["detection_result"]
        
        col1.header("Time")
        col1.write(time_result)
        col2.header("Weather")
        col2.write(weather_result)
        col3.header("Detection")
        col3.write(detection_result)
        col4.header("Detection_gpt")
        col4.write(detection_gpt_result)
        col5.header("Lane detection")
        col5.write(lane_detection_result)
        
if st.button("get lane detection result"):
    col1, col2, col3 = st.columns(3)

    if input_image:
        # JSONResponse(content={"detection_result": detection_result})
        lane_detection_process = process(input_image, lane_detection_backend)
        lane_detection_result = lane_detection_process.content
        if isinstance(lane_detection_result, bytes):
            lane_detection_result = lane_detection_result.decode("utf-8")
        else:
            lane_detection_result = str(lane_detection_result)

        payload          = json.loads(lane_detection_result)
        lane_detection_result = payload["detection_result"]
        img_b64          = payload["image"]
        img_bytes        = base64.b64decode(img_b64)
        
        original_image = Image.open(input_image).convert("RGB")
        detected_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        col1.header("Original")
        col1.image(original_image)
        col2.header("Detected")
        col2.image(detected_image)
        col3.header("Detection Result")
        col3.write(lane_detection_result)
    else:
        st.write("Insert an image!")
    
# if st.button("Get segmentation map"):
#     col1, col2 = st.columns(2)

#     if input_image:
#         segments = process(input_image, segmentation_backend)
#         original_image = Image.open(input_image).convert("RGB")
#         segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
#         col1.header("Original")
#         col1.image(original_image, use_column_width=True)
#         col2.header("Segmented")
#         col2.image(segmented_image, use_column_width=True)
#     else:
#         st.write("Insert an image!")

if st.button("Get time classification"):
    col1, col2 = st.columns(2)

    if input_image:
        time_process = process(input_image, time_classification_backend)
        time_result = time_process.content
        original_image = Image.open(input_image).convert("RGB")
        # classified_image = Image.open(io.BytesIO(classifications.content)).convert("RGB")
        col1.header("Original")
        col1.image(original_image)
        col2.header("Classified")
        col2.write(time_result)
    else:
        st.write("Insert an image!")

if st.button("Get weather classification"):
    col1, col2 = st.columns(2)

    if input_image:
        weather_process = process(input_image, weather_classification_backend)
        weather_result = weather_process.content
        original_image = Image.open(input_image).convert("RGB")
        # classified_image = Image.open(io.BytesIO(classifications.content)).convert("RGB")
        col1.header("Original")
        col1.image(original_image)
        col2.header("Classified")
        col2.write(weather_result)
    else:
        st.write("Insert an image!")
        
if st.button("Get detection yolo map"):
    col1, col2, col3 = st.columns(3)

    if input_image:
        # JSONResponse(content={"detection_result": detection_result})
        detection_process = process(input_image, detection_yolo_backend)
        detection_result = detection_process.content
        if isinstance(detection_result, bytes):
            detection_result = detection_result.decode("utf-8")
        else:
            detection_result = str(detection_result)
            
        payload          = json.loads(detection_result)
        detection_result = payload["detection_result"]
        img_b64          = payload["image"]
        img_bytes        = base64.b64decode(img_b64)
        
        original_image = Image.open(input_image).convert("RGB")
        detected_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        col1.header("Original")
        col1.image(original_image)
        col2.header("Detected")
        col2.image(detected_image)
        col3.header("Detection Result")
        col3.write(detection_result)
    else:
        st.write("Insert an image!")        
    
if st.button("Get detection gpt"):
    col1, col2, col3 = st.columns(3)

    if input_image:
        # JSONResponse(content={"detection_result": detection_result})
        detection_gpt_process = process(input_image, detection_gpt_backend)
        detection_gpt_result = detection_gpt_process.content
        if isinstance(detection_gpt_result, bytes):
            detection_gpt_result = detection_gpt_result.decode("utf-8")
        else:
            detection_gpt_result = str(detection_gpt_result)
            
        payload          = json.loads(detection_gpt_result)
        detection_gpt_result = payload["detection_gpt_result"]
        img_b64          = payload["image"]
        img_bytes        = base64.b64decode(img_b64)
        
        original_image = Image.open(input_image).convert("RGB")
        detected_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        col1.header("Original")
        col1.image(original_image)
        col2.header("Detected")
        col2.image(detected_image)
        col3.header("Detection Result")
        col3.write(detection_gpt_result)
    else:
        st.write("Insert an image!")