"""
CIFAR-10 Image Classifier using HOG Features + SVM
====================================================
A beginner-friendly Computer Vision pipeline that:
  1. Loads the CIFAR-10 dataset
  2. Extracts Histogram of Oriented Gradients (HOG) features
  3. Trains an SVM classifier with scikit-learn
  4. Evaluates and saves results

Author: <Your Name>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import color

import pickle
import os
import time

# ── Dataset ──────────────────────────────────────────────────────────────────

def unpickle(file):
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    return d

def load_cifar10(data_dir="data/cifar-10-batches-py"):
    """Load CIFAR-10 from extracted directory."""
    X_train, y_train = [], []

    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        X_train.append(batch[b"data"])
        y_train.extend(batch[b"labels"])

    X_train = np.vstack(X_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)

    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    X_test = test_batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_batch[b"labels"])

    return X_train, y_train, X_test, y_test

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_hog_features(images, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """Convert images to HOG feature vectors."""
    features = []
    for img in images:
        gray = color.rgb2gray(img)
        hog_feat = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
        )
        features.append(hog_feat)
    return np.array(features)

# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_sample_images(X, y, save_path="results/sample_images.png"):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("CIFAR-10 — Sample Images", fontsize=14, fontweight="bold")
    for ax, cls_idx in zip(axes.flat, range(10)):
        idx = np.where(y == cls_idx)[0][0]
        ax.imshow(X[idx])
        ax.set_title(CLASSES[cls_idx], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix — HOG + SVM on CIFAR-10", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")

def plot_per_class_accuracy(report_dict, save_path="results/per_class_accuracy.png"):
    classes = CLASSES
    f1_scores = [report_dict[c]["f1-score"] for c in classes]
    colors = plt.cm.RdYlGn(np.array(f1_scores))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, f1_scores, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1-Score — HOG + SVM", fontweight="bold")
    ax.axhline(np.mean(f1_scores), color="steelblue", linestyle="--", label=f"Mean F1: {np.mean(f1_scores):.2f}")
    ax.legend()
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")

# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    print("\n🔵 Step 1 — Loading CIFAR-10 dataset ...")
    X_train, y_train, X_test, y_test = load_cifar10()
    print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")

    plot_sample_images(X_train, y_train)

    # Use a subset for speed (increase for better accuracy)
    TRAIN_SAMPLES = 10000
    TEST_SAMPLES  = 2000
    X_train, y_train = X_train[:TRAIN_SAMPLES], y_train[:TRAIN_SAMPLES]
    X_test,  y_test  = X_test[:TEST_SAMPLES],   y_test[:TEST_SAMPLES]
    print(f"   Using {TRAIN_SAMPLES} train / {TEST_SAMPLES} test samples")

    print("\n🔵 Step 2 — Extracting HOG features ...")
    t0 = time.time()
    X_train_hog = extract_hog_features(X_train)
    X_test_hog  = extract_hog_features(X_test)
    print(f"   HOG feature size: {X_train_hog.shape[1]}  |  Time: {time.time()-t0:.1f}s")

    print("\n🔵 Step 3 — Building Pipeline (Scaler → PCA → SVM) ...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=200, random_state=42)),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ])

    print("\n🔵 Step 4 — Training ...")
    t0 = time.time()
    pipeline.fit(X_train_hog, y_train)
    print(f"   Training complete in {time.time()-t0:.1f}s")

    print("\n🔵 Step 5 — Evaluation ...")
    y_pred = pipeline.predict(X_test_hog)
    acc = (y_pred == y_test).mean()
    print(f"\n   ✅ Test Accuracy: {acc*100:.2f}%\n")
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    print("\n🔵 Step 6 — Saving plots ...")
    plot_confusion_matrix(y_test, y_pred)
    plot_per_class_accuracy(report)

    print("\n🔵 Step 7 — Saving model ...")
    with open("results/model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("   Saved → results/model.pkl")

    print("\n✅ All done! Check the results/ folder.\n")

if __name__ == "__main__":
    main()
