import io
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st

import json
import base64
import numpy as np
from fastapi import UploadFile

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
detection_yolov10_backend = "http://localhost:8000/detection_yolov10"
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

# construct UI layout
st.title("[BigData] Auto-Labeling Web Frontend")

st.write(
    """Obtain segmentation, detection, classification predictions from image inputs via models implemented in PyTorch.
         This Streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation."""
)  # description and instructions

input_image = st.file_uploader("Insert image")  # image upload widget

if input_image:
    image_process = process_image(input_image, image_backend)
    image_result = image_process.content
    weather_process = process(input_image, weather_classification_backend)
    weather_result = weather_process.content
    time_process = process(input_image, time_classification_backend)
    time_result = time_process.content
    detection_process = process(input_image, detection_yolov10_backend)
    detection_result = detection_process.content
    lane_detection_process = process(input_image, lane_detection_backend)
    image_result = image_result.decode("utf-8")

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
    
    image_data = json.loads(image_result)
    image_info = image_data["image_info"]
    weather_data = json.loads(weather_result)
    weather_class = weather_data["weather_class"]
    time_data = json.loads(time_result)
    time_class = time_data["time_class"]
    detection_data = json.loads(detection_result)
    detection_list = detection_data["detection_result"]
    
    lane_detection_data = lane_detection_process.json()
    image_b64 = lane_detection_data.get("image")
    lane_detection_result = lane_detection_data.get("detection_result")
    image_bytes = base64.b64decode(image_b64)
    lane_detection_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
        "Lane_Detection_information": {
            "num_of_lanes": len(lane_detection_result),
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
    for lane in lane_detection_result:
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
    col1, col2, col3, col4 = st.columns(4)
    if input_image:
        weather_process = process(input_image, weather_classification_backend)
        weather_result = weather_process.content
        time_process = process(input_image, time_classification_backend)
        time_result = time_process.content
        detection_process = process(input_image, detection_yolov10_backend)
        detection_result = detection_process.content
        lane_detection_process = process(input_image, lane_detection_backend)
        lane_detection_image = Image.open(io.BytesIO(lane_detection_process.content)).convert("RGB")
        
        col1.header("Lane detection result")
        col1.image(lane_detection_image, use_column_width=True)
        col2.header("Time")
        col2.write(time_result)
        col3.header("Weather")
        col3.write(weather_result)
        col4.header("Detection")
        col4.write(detection_result)

if st.button("get lane detection result"):
    col1, col2 = st.columns(2)
    if input_image:
        lane_detection_process = process(input_image, lane_detection_backend)
        lane_detection_image = Image.open(io.BytesIO(lane_detection_process.content)).convert("RGB")
        
        col1.header("Image")
        col1.image(input_image, use_column_width=True)
        col2.header("Lane Detection")
        col2.image(lane_detection_image, use_column_width=True)
    
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
        col1.image(original_image, use_column_width=True)
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
        col1.image(original_image, use_column_width=True)
        col2.header("Classified")
        col2.write(weather_result)
    else:
        st.write("Insert an image!")
        
if st.button("Get detection yolo map"):
    col1, col2, col3 = st.columns(3)

    if input_image:
        # JSONResponse(content={"detection_result": detection_result})
        detection_process = process(input_image, detection_yolov10_backend)
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
        col1.image(original_image, use_column_width=True)
        col2.header("Detected")
        col2.image(detected_image, use_column_width=True)
        col3.header("Detection Result")
        col3.write(detection_result)
    else:
        st.write("Insert an image!")        
