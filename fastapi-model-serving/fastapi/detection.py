import io
import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_detector(device = 'cpu'):
    model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    model = model.to(device)
    return model

def get_detections(model, binary_image, threshold=0.5):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    input_tensor = transform(input_image)
    input_batch = [input_tensor]

    with torch.no_grad():
        outputs = model(input_batch)

    # Process the outputs
    output = outputs[0]
    boxes = output['boxes']
    labels = output['labels']
    scores = output['scores']

    draw = ImageDraw.Draw(input_image)
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], box[1]), f"{label.item()} {score:.2f}", fill="red")

    return input_image