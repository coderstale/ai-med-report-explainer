import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

LABELS = [
    "Normal",
    "Anaemia",
    "Thrombocytopenia",
    "Leukocytosis",
    "Leukopenia",
    "Macrocytosis",
    "Microcytosis",
    "Eosinophilia",
    "Other"
]

def predict_condition(features):
    """
    Accepts a list of 4 lab values: [haemoglobin, t.l.c, platelet count, mcv]
    Applies a simple rule-based logic for demonstration.
    """
    hb, tlc, platelets, mcv = features

    if hb < 10:
        if mcv < 80:
            return "Microcytic Anaemia"
        elif mcv > 100:
            return "Macrocytic Anaemia"
        else:
            return "Normocytic Anaemia"
    elif platelets < 150:
        return "Thrombocytopenia"
    elif tlc > 11:
        return "Leukocytosis"
    elif tlc < 4:
        return "Leukopenia"
    else:
        return "Normal"

def predict_condition_from_image(image: Image.Image):
    """
    Takes a PIL image, resizes and normalises it,
    loads the saved CNN model and returns the predicted condition.
    """
    model_path = "model.pth"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = transform(image).unsqueeze(0)  

    from torchvision.models import convnext_tiny
    model = convnext_tiny(num_classes=len(LABELS))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    return LABELS[pred_idx]
