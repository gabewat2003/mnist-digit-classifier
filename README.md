# mnist-digit-classifier

Classifying handwritten digits (0-9) from the MNIST dataset using scikit-learn, comparing several classical models and testing the best one on your own handwriting.

## Dataset

MNIST: 70,000 grayscale images, 28x28 pixels, ten classes. The standard split is 60,000 train / 10,000 test. Each image flattens to a 784-feature vector with pixel values 0-255.

```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(int)
X = X / 255.0                                   # scale to 0-1
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

The data is downloaded and cached on first run; it is not committed to the repo.

## Approach

Same data, same split, several models:

| Model | Notes |
|---|---|
| Logistic regression | Linear baseline |
| k-NN | Strong on MNIST despite simplicity; slow at prediction |
| SVM (RBF kernel) | Typically the best of the classical models; slow to train on 60k |
| Random forest | Fast, robust, ignores scaling |

**Dimensionality reduction:** PCA to ~50 components keeps most of the variance and speeds up k-NN and SVM a lot.

**Tuning:** `GridSearchCV` over `C` and `gamma` (SVM) or `k` (k-NN) on a subset (e.g. 10k samples) to keep runtimes sane, then refit the best on the full training set.

## Try it on your own handwriting

MNIST digits are white on black, centered, 28x28. A photo of your writing needs the same preprocessing:

1. Convert to grayscale
2. Invert (dark ink becomes bright)
3. Crop tightly, pad to square, resize to 28x28
4. Scale to 0-1, flatten to 784

```python
from PIL import Image, ImageOps
img = ImageOps.invert(Image.open("my_digit.png").convert("L")).resize((28, 28))
x = (np.array(img) / 255.0).reshape(1, -1)
print(model.predict(x))
```

Mismatched preprocessing is the usual cause of "the model is wrong on my digit."

## Project layout

```
mnist-digit-classifier/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py            # load + split
│   ├── train.py           # fit models, save with joblib
│   ├── evaluate.py        # metrics, confusion matrix, misclassified grid
│   └── predict.py         # preprocess a custom image and predict
├── notebooks/01_exploration.ipynb
├── results/               # figures and metrics.json
└── models/                # not committed
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.train --model svm --pca 50
python -m src.predict --image my_digit.png
```

## Tech

Python, scikit-learn, NumPy, matplotlib, Pillow, joblib.

