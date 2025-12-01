# app/main.py

import io
import os
from typing import Dict

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from src.model import build_model
from src.dataset import CIFAR10_MEAN, CIFAR10_STD

# CIFAR-10 classes
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Starting API. Using device:", device)

model_name = "simple_cnn"
model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES)).to(device)

weights_path = f"models/best_model_{model_name}.pth"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Could not find {weights_path}. Train model first.")

state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Transform pipeline for inference
inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def predict_pil_image(img: Image.Image) -> Dict:
    img = img.convert("RGB")
    tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    probs_np = probs.cpu().numpy()
    top_idx = int(np.argmax(probs_np))

    return {
        "predicted_class": CLASS_NAMES[top_idx],
        "predicted_index": top_idx,
        "confidence": float(probs_np[top_idx]),
        "all_probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs_np)}
    }


# FastAPI app
app = FastAPI(title="CIFAR10 Image Classifier API")

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serve frontend."""
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/predict-image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        result = predict_pil_image(img)
        result["filename"] = file.filename
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
