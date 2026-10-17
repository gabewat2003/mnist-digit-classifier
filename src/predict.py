"""
predict.py: Preprocesses custom user-provided handwritten digit images and predicts the digit.
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageOps
import joblib


def preprocess_image(image_path):
    """
    Preprocesses a custom handwriting image to match MNIST specifications:
    1. Grayscale conversion
    2. Inversion (dark ink becomes bright on dark background)
    3. Tight crop, padding to square, and resizing to 28x28
    4. Scaling to [0, 1] and flattening to a 784 vector
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load image and convert to grayscale
    img = Image.open(image_path).convert("L")

    # Invert colors if necessary (assuming dark ink on light paper)
    # If your image is already white-on-black, comment out ImageOps.invert
    img = ImageOps.invert(img)

    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array, scale to [0, 1], and flatten
    img_array = np.array(img, dtype=np.float32) / 255.0
    x = img_array.reshape(1, -1)

    return x


def main():
    parser = argparse.ArgumentParser(
        description="Predict a custom handwritten digit image."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to custom digit image."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/svm_pca50.joblib",
        help="Path to trained model joblib file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        # Fallback to any available model in models/
        model_files = [
            os.path.join("models", f)
            for f in os.listdir("models")
            if f.endswith(".joblib")
        ]
        if model_files:
            args.model = model_files[0]
        else:
            raise FileNotFoundError(
                "No trained models found. Please train a model first using src.train."
            )

    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)

    print(f"Preprocessing image {args.image}...")
    x = preprocess_image(args.image)

    prediction = model.predict(x)[0]
    print(f"\nPredicted Digit: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[0]
        print(f"Confidence: {probabilities[prediction] * 100:.2f}%")


if __name__ == "__main__":
    main()
