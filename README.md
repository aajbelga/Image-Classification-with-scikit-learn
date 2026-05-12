# 🖼️ CIFAR-10 Image Classifier — HOG + SVM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/Domain-Computer%20Vision-purple)

A beginner-friendly **Computer Vision** project that classifies images from the CIFAR-10 dataset into 10 categories using **Histogram of Oriented Gradients (HOG)** feature extraction and a **Support Vector Machine (SVM)** classifier — all with scikit-learn.

---

## 📌 What This Project Does

Given any image, the model predicts which of these 10 categories it belongs to:

| ✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer |
|:-----------:|:-------------:|:-------:|:------:|:-------:|
| 🐶 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚛 Truck |

---

## 🧠 How It Works

```
Raw Image (32×32 RGB)
        │
        ▼
   Grayscale Conversion
        │
        ▼
  HOG Feature Extraction        ← Captures edges & texture
  (8 orientations, 4×4 cells)
        │
        ▼
   StandardScaler               ← Normalize features
        │
        ▼
   PCA (200 components)         ← Dimensionality reduction
        │
        ▼
   SVM (RBF kernel)             ← Classification
        │
        ▼
   Predicted Class 🎯
```

### Why HOG?
HOG (Histogram of Oriented Gradients) captures the **shape and edge structure** of objects by counting how often gradient orientations occur in localized portions of the image. It's lightweight, interpretable, and works well with SVMs.

---

## 📁 Project Structure

```
image-classifier/
├── src/
│   ├── train.py           # Full training pipeline
│   ├── predict.py         # Inference on a single image
│   └── download_data.py   # Downloads CIFAR-10 automatically
├── data/                  # Dataset (auto-downloaded)
├── results/               # Saved model + plots (auto-generated)
│   ├── model.pkl
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── sample_images.png
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/image-classifier.git
cd image-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
python src/download_data.py
```

### 4. Train the model
```bash
python src/train.py
```

Training takes **~2–5 minutes** on a standard CPU. You'll see step-by-step logs.

### 5. Predict on your own image
```bash
python src/predict.py --image path/to/your/image.jpg
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~52–55% |
| Train Samples | 10,000 |
| Feature Size (HOG) | 1,152 |
| After PCA | 200 components |

> **Note:** CIFAR-10 is a challenging benchmark even for deep learning (CNNs reach ~93%). For a classical ML approach with HOG + SVM, ~50–55% is a solid baseline.

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Per-Class F1-Score
![F1 Scores](results/per_class_accuracy.png)

---

## 🔧 Experiment Ideas

Want to improve accuracy? Try these:
- ⚡ Increase `TRAIN_SAMPLES` in `train.py` to use all 50,000 images
- 🔬 Tune HOG parameters: `orientations`, `pixels_per_cell`, `cells_per_block`
- 🧪 Try different classifiers: `RandomForestClassifier`, `LogisticRegression`
- 📐 Try different `PCA` component counts
- 🎨 Add color histograms alongside HOG features

---

## 📚 Concepts Covered

- Image preprocessing & feature extraction
- Histogram of Oriented Gradients (HOG)
- Support Vector Machines (SVM) with RBF kernel
- Dimensionality reduction with PCA
- scikit-learn Pipelines
- Model evaluation: confusion matrix, F1-score, classification report

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **scikit-learn** — ML pipeline, SVM, PCA
- **scikit-image** — HOG feature extraction
- **NumPy** — array operations
- **Matplotlib** — visualizations
- **Pillow** — image I/O

---

## 📄 License

MIT License — feel free to use, fork, and build on this project.

---

## 🙌 Acknowledgements

- Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- HOG paper: Dalal & Triggs, CVPR 2005
