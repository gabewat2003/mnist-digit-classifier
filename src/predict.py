"""
predict.py: Preprocess a photo or scan of a handwritten digit the way MNIST was
built, then predict it with a trained model.

Usage:
    python -m src.predict --image my_digit.png [--model models/svm_pca50.joblib]
"""

import argparse
import os

import joblib
import numpy as np
from PIL import Image, ImageOps

DEFAULT_MODEL = "models/svm_pca50.joblib"


def preprocess_image(image_path, invert=True, threshold=0.2):
    """Convert an image to a 1x784 vector matching MNIST.

    MNIST digits were size-normalised to fit a 20x20 box, then centred by
    centre of mass in a 28x28 frame. This reproduces that:
      1. grayscale, 2. invert (dark ink on light paper -> bright on dark),
      3. crop to the ink bounding box, 4. pad to square, resize to 20x20,
      5. paste into 28x28 and shift so the centre of mass is in the middle,
      6. scale to [0, 1] and flatten.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("L")
    if invert:
        img = ImageOps.invert(img)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Zero out faint background (paper texture, shadows) before cropping.
    arr[arr < threshold] = 0.0
    rows, cols = np.nonzero(arr)
    if len(rows) == 0:
        raise ValueError("No ink found in the image. Try --no-invert or a cleaner photo.")
    arr = arr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    h, w = arr.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = arr

    digit = Image.fromarray((square * 255).astype(np.uint8)).resize(
        (20, 20), Image.Resampling.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = np.asarray(digit, dtype=np.float32) / 255.0

    # Shift so the centre of mass sits at the image centre.
    total = canvas.sum()
    ys, xs = np.indices(canvas.shape)
    cy, cx = (ys * canvas).sum() / total, (xs * canvas).sum() / total
    canvas = np.roll(canvas, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))

    return canvas.reshape(1, -1)


def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit image.")
    parser.add_argument("--image", required=True, help="Path to the digit image.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Trained .joblib model.")
    parser.add_argument("--no-invert", action="store_true",
                        help="Use if the image is already light ink on a dark background.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"{args.model} not found. Train it first: python -m src.train --model svm --pca 50")

    model = joblib.load(args.model)
    x = preprocess_image(args.image, invert=not args.no_invert)
    prediction = model.predict(x)[0]
    print(f"Predicted digit: {prediction}")

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x)[0]
            print(f"Confidence: {proba[prediction] * 100:.1f}%")
        except AttributeError:
            pass  # e.g. SVC trained without probability=True


if __name__ == "__main__":
    main()
