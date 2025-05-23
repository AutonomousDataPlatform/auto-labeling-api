import io

from image_lane_detection import get_lane_detector, get_lane_detections
from starlette.responses import Response, JSONResponse
import torch
from fastapi import FastAPI, File
import base64
import numpy as np

device = "cuda:2" if torch.cuda.is_available() else "cpu"
# cls_model = get_classifier(device)
lane_detection_model = get_lane_detector(device)


app = FastAPI(
    title="[BigData] Auto-Labeling API Server",
    
    description="""Obtain segmentation, detection, classification predictions from image inputs via models implemented in PyTorch.
                           Visit this URL at port 8501 for the Streamlit interface.""",
    version="0.1.0",
)

# @app.post("/lane_detection")
# def get_lane_detection(file: bytes = File(...)):
#     """Get lane detection from image file"""
#     lane_image, lane_detection = get_lane_detections(lane_detection_model, file, device)
#     bytes_io = io.BytesIO()
#     lane_image.save(bytes_io, format="PNG")

#     return Response(bytes_io.getvalue(), media_type="image/png")
def convert_to_serializable(obj):
    """
    obj가 numpy.ndarray이면 리스트로 변환하고,
    리스트나 dict 내에 있는 ndarray도 재귀적으로 처리합니다.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj

@app.post("/lane_detection")
def get_lane_detection(file: bytes = File(...)):
    """Get lane detection from image file"""
    lane_image, lane_detection = get_lane_detections(lane_detection_model, file, device)
    bytes_io = io.BytesIO()
    lane_image.save(bytes_io, format="PNG")
    image_bytes = bytes_io.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    lane_detection = convert_to_serializable(lane_detection)

    return JSONResponse(content={"image": image_b64, "detection_result": lane_detection})

# @app.post("/test")
# def get_classification_map(file: bytes = File(...)):
#     """Get weather classification from image file"""
#     weather_image, weather_class = get_weather_classifications(cls_model, file, device)
#     # bytes_io = io.BytesIO()
#     # weather_image.save(bytes_io, format="PNG")

#     return JSONResponse(content={"weather_class": weather_class})
#     # return Response(bytes_io.getvalue(), media_type="text/plane")
