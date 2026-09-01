import io
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st

import json
import base64
import numpy as np
import os
import glob
import zipfile
import threading
import concurrent.futures as futures


def dict_to_numpy(json_str):
    # JSON 문자열을 딕셔너리로 파싱
    decoded = json.loads(json_str.content)
    # Base64로 인코딩된 데이터를 바이트로 디코딩
    # response to dict
    print("decoded: ", decoded)
    array_bytes = base64.b64decode(decoded['data'])
    # NumPy 배열로 변환
    return np.frombuffer(array_bytes, dtype=decoded['dtype']).reshape(decoded['shape'])

# segmentation_backend = "http://localhost:8000/segmentation"
detection_yolo_backend = "http://localhost:8000/detection_yolo"
weather_classification_backend = "http://localhost:8000/weather_classification"
time_classification_backend = "http://localhost:8000/time_classification"
# image_backend = "http://localhost:8000/image"  # 미사용: 이미지 메타데이터는 로컬에서 직접 읽는다
lane_detection_backend = "http://localhost:8001/lane_detection"

REQUEST_TIMEOUT = 120   # 초. 기존 8000은 사실상 무한 대기라 하나 멈추면 배치가 통째로 정지한다

# 배치 경로는 좌표만 쓰므로 시각화 PNG를 만들지 않도록 include_image=false 로 호출한다
ANALYSIS_BACKENDS = {
    "weather":   weather_classification_backend,
    "time":      time_classification_backend,
    "detection": detection_yolo_backend + "?include_image=false",
    "lane":      lane_detection_backend + "?include_image=false",
}

_local = threading.local()

def _session():
    # requests.Session은 스레드 간 공유가 보장되지 않으므로 스레드마다 하나씩 둔다
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session

def post_image(server_url, filename, img_bytes):
    """이미지 바이트를 백엔드에 POST하고 JSON을 반환한다.

    호출마다 새 BytesIO를 만들어 스레드끼리 파일 포인터를 공유하지 않게 한다.
    """
    m = MultipartEncoder(fields={"file": (filename, io.BytesIO(img_bytes), "image/jpeg")})
    r = _session().post(
        server_url,
        data=m,
        headers={"Content-Type": m.content_type},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()   # 백엔드 500을 여기서 잡는다 (기존엔 json.loads에서 엉뚱하게 터졌다)
    return r.json()

def analyze_image(filename, img_bytes):
    """4개 백엔드를 동시에 호출해 {키: 응답JSON} 으로 반환한다."""
    with futures.ThreadPoolExecutor(max_workers=len(ANALYSIS_BACKENDS)) as ex:
        jobs = {key: ex.submit(post_image, url, filename, img_bytes)
                for key, url in ANALYSIS_BACKENDS.items()}
        return {key: job.result() for key, job in jobs.items()}


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

def build_auto_labeling(filename, img_bytes):
    """이미지 1장에 대한 Auto_labeling 블록을 만든다.

    - 이미지 메타데이터는 로컬에서 직접 읽는다 (/image 왕복 제거)
    - 나머지 4개 백엔드는 동시에 호출한다 (소요시간이 합계가 아니라 최댓값이 된다)
    """
    with Image.open(io.BytesIO(img_bytes)) as im:
        width, height, image_format, image_mode = im.width, im.height, im.format, im.mode

    res = analyze_image(filename, img_bytes)
    detection_list = res["detection"]["detection_result"]
    lane_detection_list = res["lane"]["detection_result"]

    return {
        "Image_information": {
            "file_name": filename,
            "width": width,
            "height": height,
            "format": image_format,
            "mode": image_mode,
        },
        "Time_information": {"class": res["time"]["time_class"]},
        "Weather_information": {"class": res["weather"]["weather_class"]},
        "Detection_information": {
            "num_of_bbox": len(detection_list),
            "bbox_info": [
                {
                    "class": box[0],
                    "type": "Bounding_box",
                    "bbox_x1": box[1],
                    "bbox_y1": box[2],
                    "bbox_x2": box[3],
                    "bbox_y2": box[4],
                }
                for box in detection_list
            ],
        },
        "Lane_Detection_information": {
            "num_of_lanes": len(lane_detection_list),
            "lane_info": [{"type": "Line", "points": lane} for lane in lane_detection_list],
        },
    }

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

            structured_result = {
                "Original_calib": read_text_if_exists(calib_txt),
                "Original_label": read_text_if_exists(label_txt),
                "Auto_labeling": build_auto_labeling(os.path.basename(p), img_bytes),
            }
            
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
    structured_result = build_auto_labeling(input_image.name, input_image.getvalue())

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
    col1, col2, col3, col4, col5, col6 = st.columns(6)
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
