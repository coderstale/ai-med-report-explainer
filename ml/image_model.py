import torch
from torchvision import models, transforms
from PIL import Image

LABELS = ["Normal", "Anaemia", "Leukemia", "Infection"]
MODEL_PATH = "models/cnn_diagnosis_model.pt"

def load_cnn_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_image_diagnosis(img: Image.Image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        return LABELS[predicted.item()]
