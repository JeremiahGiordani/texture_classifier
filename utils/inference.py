import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.model import SimpleCNN

def load_model(model_path, device, num_classes=4):
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model.pth"
    model = load_model(model_path, device)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Inference on a single image
    test_image_path = "data/texture_windows/img_001-003.tiff"
    image = Image.open(test_image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    print(f"Predicted class for {os.path.basename(test_image_path)}: {pred}")
