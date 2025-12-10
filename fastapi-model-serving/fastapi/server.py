import io

from segmentation import get_segmentator, get_segments
from detection import get_detector, get_detections
from image_detection_yolo import get_image_detector_yolo, get_image_detections_yolo
from weather_classification import get_weather_classifier, get_weather_classifications
from time_classification import get_time_classifier, get_time_classifications
from starlette.responses import Response, JSONResponse
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import base64

device = "cuda:0" if torch.cuda.is_available() else "cpu"
seg_model = get_segmentator(device)
det_model = get_detector(device)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cls_model_time = get_time_classifier(device)
cls_model_weather = get_weather_classifier(device)
det_model_yolo = get_image_detector_yolo(device)

app = FastAPI(
    title="[BigData] Auto-Labeling API Server",
    
    description="""Obtain segmentation, detection, classification predictions from image inputs via models implemented in PyTorch.
                           Visit this URL at port 8501 for the Streamlit interface.""",
    version="0.1.0",
)

@app.post("/segmentation")
def get_segmentation_map(file: bytes = File(...)):
    """Get segmentation maps from image file"""
    segmented_image = get_segments(seg_model, file)
    bytes_io = io.BytesIO()
    segmented_image.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")

@app.post("/detection_yolo")
def get_detection_map(file: bytes = File(...)):
    """Get detection maps from image file"""
    detection_image, detection_result = get_image_detections_yolo(det_model_yolo, file)
    # print("detection_result: ", detection_result)
    bytes_io = io.BytesIO()
    detection_image.save(bytes_io, format="PNG")
    png_bytes = bytes_io.getvalue()
    img_b64 = base64.b64encode(png_bytes).decode("utf-8")
    return JSONResponse(content={"detection_result": detection_result, "image": img_b64})

@app.post("/weather_classification")
def get_classification_map(file: bytes = File(...)):
    """Get weather classification from image file"""
    weather_image, weather_class = get_weather_classifications(cls_model_weather, file, device)
    # bytes_io = io.BytesIO()
    # weather_image.save(bytes_io, format="PNG")

    return JSONResponse(content={"weather_class": weather_class})
    # return Response(bytes_io.getvalue(), media_type="text/plane")

@app.post("/time_classification")
def get_classification_time(file: bytes = File(...)):
    """Get time classification from image file"""
    time_image, time_class = get_time_classifications(cls_model_time, file, device)
    # bytes_io = io.BytesIO()
    # weather_image.save(bytes_io, format="PNG")

    return JSONResponse(content={"time_class": time_class})
    # return Response(bytes_io.getvalue(), media_type="text/plane")
    
@app.post("/image")
async def upload_image(file: UploadFile = File(...)):
    image_name = file.filename
    file_bytes = await file.read()
    image = Image.open(io.BytesIO(file_bytes))
    width, height = image.size
    image_format = image.format
    image_mode = image.mode

    # 필요한 경우 이미지 정보를 dict로 구성하여 반환
    image_info = {
        "name": image_name,
        "width": width,
        "height": height,
        "format": image_format,
        "mode": image_mode
    }
    
    return JSONResponse(content={"image_info": image_info})
