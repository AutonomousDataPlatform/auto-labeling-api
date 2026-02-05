import io
import torch
import clip
from PIL import Image

    
def get_weather_classifier_clip(device = 'cpu'):
    # Load the EfficientNet model
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    return model, preprocess

def get_weather_classifications_clip(model, preprocess, binary_image, device = 'cpu'):
    
    # classes=['daytime', 'dawn/dusk', 'Night']
    classes=['Clear', 'Overcast', 'Foggy', 'Rainy']

    # Load the image and convert it to RGB
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    
    image = preprocess(input_image).unsqueeze(0).to(device)
    text = clip.tokenize(classes).to(device)

    # Perform inference with the model
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Compute similarity between image and text features
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Get the predicted class and its probability
    probabilities = torch.nn.functional.softmax(torch.tensor(probs[0]), dim=0)
    max_prob, weather_idx = torch.max(probabilities, dim=0)
    weather_class = classes[weather_idx]

    return input_image, weather_class
