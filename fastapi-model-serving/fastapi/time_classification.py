import io
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

# Define a custom head for the model to predict time
class CustomHead(torch.nn.Module):
    def __init__(self, num_ftrs, num_cls):
        super(CustomHead, self).__init__()
        self.fc_output = torch.nn.Linear(num_ftrs, num_cls)

    def forward(self, x):
        result = self.fc_output(x)
        return result
    
def get_time_classifier(device = 'cpu'):
    # Load the EfficientNet model
    model = EfficientNet.from_name('efficientnet-b5')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_ftrs = model._fc.in_features

    model._fc = CustomHead(num_ftrs, 2)
    model._fc = model._fc.to(device)

    # Load the state dict and remove 'module.' prefix if present
    state_dict = torch.load('./weights/best_model_time_e10_lr0001_1.pth', map_location=device)
    
    # Remove 'module.' prefix from state dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value  # Remove 'module.' prefix (7 characters)
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)

    model.eval()
    return model

def get_time_classifications(model, binary_image, device = 'cpu', threshold=0.5):
    
    classes=['Daytime', 'Night']
    # classes=['Clear', 'Overcast', 'Foggy', 'Rainy']

    # Load the image and convert it to RGB
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    
    # Define the transform to convert the image to a tensor
    # EfficientNet-B5 requires 456x456 input and ImageNet normalization
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    max_prob, time_idx = torch.max(probabilities, dim=0)
    time_class = classes[time_idx]

    # Draw the prediction on the image
    draw = ImageDraw.Draw(input_image)
    text = f"{time_class} ({max_prob:.2f})"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
    # text_size = draw.textsize(text, font=font)
    # text_x = 10
    # text_y = 10
    # draw.rectangle([text_x, text_y, text_x + text_size[0], text_y + text_size[1]], fill="black")
    # draw.text((text_x, text_y), text, fill="white", font=font)

    return input_image, time_class
