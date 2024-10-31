import io
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st

import json
import base64
import numpy as np

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
lane_detection_backend = "http://localhost:8001/lane_detection"

def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
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

if st.button("get total result"):
    col1, col2, col3 = st.columns(3)
    if input_image:
        weather_process = process(input_image, weather_classification_backend)
        weather_result = weather_process.content
        detection_process = process(input_image, detection_yolov10_backend)
        detection_result = detection_process.content
        lane_detection_process = process(input_image, lane_detection_backend)
        lane_detection_image = Image.open(io.BytesIO(lane_detection_process.content)).convert("RGB")
        
        col1.header("Lane detection result")
        col1.image(lane_detection_image, use_column_width=True)
        col2.header("Weather")
        col2.write(weather_result)
        col3.header("Detection")
        col3.write(detection_result)

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


if st.button("Get weather classification"):
    col1, col2 = st.columns(2)

    if input_image:
        weather_result = process(input_image, weather_classification_backend)
        original_image = Image.open(input_image).convert("RGB")
        # classified_image = Image.open(io.BytesIO(classifications.content)).convert("RGB")
        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Classified")
        col2.write(weather_result.content)
    else:
        st.write("Insert an image!")
        
if st.button("Get detection yolo map"):
    col1, col2 = st.columns(2)

    if input_image:
        # JSONResponse(content={"detection_result": detection_result})
        detections = process(input_image, detection_yolov10_backend)
        original_image = Image.open(input_image).convert("RGB")
        # detected_image = Image.open(io.BytesIO(detections.content)).convert("RGB")
        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Detected")
        # col2.image(detected_image, use_column_width=True)
        col2.write(detections.content)
    else:
        st.write("Insert an image!")        
