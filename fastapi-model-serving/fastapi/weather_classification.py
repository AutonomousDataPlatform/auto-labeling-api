import io
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

# Define a custom head for the model to predict time
class CustomHead(torch.nn.Module):
    def __init__(self, num_ftrs):
        super(CustomHead, self).__init__()
        self.fc_time = torch.nn.Linear(num_ftrs, 4)

    def forward(self, x):
        time = self.fc_time(x)
        return time
    
def get_classifier(device = 'cpu'):
    # Load the EfficientNet model
    model = EfficientNet.from_name('efficientnet-b5')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_ftrs = model._fc.in_features

    model._fc = CustomHead(num_ftrs)
    model._fc = model._fc.to(device)

    # model.load_state_dict(torch.load('/fastapi/weights/best_weather_model_e10_lr0001_s01.pth', map_location=device))
    model.load_state_dict(torch.load('./weights/best_weather_model_e10_lr0001_s01.pth', map_location=device))

    model.eval()
    return model

def get_weather_classifications(model, binary_image, device = 'cpu', threshold=0.5):
    
    # classes=['daytime', 'dawn/dusk', 'Night']
    classes=['Clear', 'Overcast', 'Foggy', 'Rainy']

    # Load the image and convert it to RGB
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    
    # Define the transform to convert the image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Apply the transform to the image
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    input_batch = input_batch.to(device)

    # Perform inference with the model
    with torch.no_grad():
        outputs = model(input_batch)

    # Get the predicted class and its probability
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    max_prob, weather_idx = torch.max(probabilities, dim=0)
    weather_class = classes[weather_idx]

    # Draw the prediction on the image
    draw = ImageDraw.Draw(input_image)
    text = f"{weather_class} ({max_prob:.2f})"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
    # text_size = draw.textsize(text, font=font)
    # text_x = 10
    # text_y = 10
    # draw.rectangle([text_x, text_y, text_x + text_size[0], text_y + text_size[1]], fill="black")
    # draw.text((text_x, text_y), text, fill="white", font=font)

    return input_image, weather_class
