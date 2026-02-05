import io
from PIL import Image, ImageDraw
import json
import base64
import cv2
import numpy as np
import os
from openai import OpenAI

def get_image_detector_gpt(api_key="AA"):
    """
    Initialize OpenAI client for GPT-4 Vision API
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    return client

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
    
def get_image_detections_gpt(client, binary_image, threshold=0.5):
    """
    Detect objects in an image using GPT-4 Vision API
    Returns: output_image with bounding boxes, list of detections [name, x1, y1, x2, y2]
    """
    # Convert binary image to PIL Image
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    
    # Encode image to base64
    buffered = io.BytesIO()
    input_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Create prompt for object detection
    prompt = """Analyze this image and detect all objects. For each object, provide:
1. Object name/class
2. Bounding box coordinates in the format [x1, y1, x2, y2] where:
   - x1, y1 is the top-left corner
   - x2, y2 is the bottom-right corner
   - Coordinates should be in pixels relative to the image dimensions

Return the results as a JSON array with this format:
[
  {{
    "name": "object_class",
    "box": {{"x1": 100, "y1": 50, "x2": 200, "y2": 150}}
  }}
]

Image dimensions: {}x{} pixels

Only return the JSON array, no additional text.""".format(width, height)
    
    try:
        # Call GPT-4 Vision API
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-vision-preview"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response (in case there's additional text)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        json_results = json.loads(response_text)
        
        # Process results
        ret_list = []
        bboxes = []
        
        for json_result in json_results:            
            name = json_result['name']
            x1 = json_result["box"]['x1']
            y1 = json_result["box"]['y1']
            x2 = json_result["box"]['x2']
            y2 = json_result["box"]['y2']
            ret_list.append([name, x1, y1, x2, y2])
            
            ret_list.append([name, x1, y1, x2, y2])
            bboxes.append([x1, y1, x2, y2])
        
        # Draw bounding boxes on image
        input_image_array = np.array(input_image)
        output_image = input_image_array.copy()
        
        for bbox in bboxes:
            random_color = tuple(np.random.choice(range(256), size=3).tolist())
            x1, y1, x2, y2 = bbox
            output_image = cv2.rectangle(output_image, (x1, y1), (x2, y2), random_color, 3)
        
        output_image = Image.fromarray(output_image)
        return output_image, ret_list
        
    except Exception as e:
        print(f"Error in GPT-4 Vision detection: {str(e)}")
        # Return original image and empty list on error
        return input_image, []