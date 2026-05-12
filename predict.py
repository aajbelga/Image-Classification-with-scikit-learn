"""
predict.py — Run inference on a single image using the saved model.

Usage:
    python src/predict.py --image path/to/image.jpg
"""

import argparse
import pickle
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage import color

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_model(model_path="results/model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    arr = np.array(img) / 255.0
    gray = color.rgb2gray(arr)
    feat = hog(
        gray,
        orientations=8,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return feat.reshape(1, -1)

def predict(image_path, model_path="results/model.pkl"):
    model = load_model(model_path)
    features = preprocess(image_path)
    pred = model.predict(features)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
    label = CLASSES[pred]
    print(f"\n🔍 Prediction: {label.upper()}")
    if proba is not None:
        print(f"   Confidence: {proba[pred]*100:.1f}%")
    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Image Classifier — Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="results/model.pkl", help="Path to saved model")
    args = parser.parse_args()
    predict(args.image, args.model)
