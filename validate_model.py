"""
validate_model.py

Usage:
  python validate_model.py --data_dir path/to/val --model alzheimer_model.h5

Expects `data_dir` to be structured as:
  val/
    Non-Demented/
    Very Mild Demented/
    Mild Demented/
    Moderate Demented/

This script computes accuracy, confusion matrix, per-class metrics, mean predicted probabilities,
and a quick check for prior-like outputs.
"""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

CLASS_NAMES = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="Path to validation directory (with subfolders per class)")
parser.add_argument("--model", default="alzheimer_model.h5", help="Model file to load")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

print("Loading model:", args.model)
model = tf.keras.models.load_model(args.model)
print("Model loaded.")

# Helper: preprocess same as app
def preprocess_pil(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

# Gather validation image paths and labels
val_dir = Path(args.data_dir)
paths = []
labels = []

# First try canonical structure: class subfolders
for idx, cls in enumerate(CLASS_NAMES):
    cls_dir = val_dir / cls
    if not cls_dir.exists():
        print(f"Warning: expected folder {cls_dir} for class {cls} not found; skipping")
        continue
    for img_path in cls_dir.glob("*.jpg"):
        paths.append(img_path)
        labels.append(idx)
    for img_path in cls_dir.glob("*.png"):
        paths.append(img_path)
        labels.append(idx)

# Fallback: some datasets encode labels in filename prefixes (e.g., ND__, VMD__, MD__, MoD__)
if len(paths) == 0:
    prefix_map = {
        'ND__': 0,   # Non-Demented
        'VMD__': 1,  # Very Mild Demented
        'MD__': 2,   # Mild Demented
        'MoD__': 3,  # Moderate Demented
        'MOD__': 3,  # sometimes uppercase
    }
    for img_path in val_dir.glob('*'):
        if not img_path.is_file():
            continue
        name = img_path.name
        matched = False
        for pfx, idx in prefix_map.items():
            if name.startswith(pfx):
                paths.append(img_path)
                labels.append(idx)
                matched = True
                break
        if not matched:
            # try underscore-separated tokens
            tokens = name.split('_')
            if tokens:
                t0 = tokens[0].upper()
                if t0.startswith('ND'):
                    paths.append(img_path); labels.append(0)
                elif t0.startswith('VMD'):
                    paths.append(img_path); labels.append(1)
                elif t0.startswith('MD') and not t0.startswith('MOD'):
                    paths.append(img_path); labels.append(2)
                elif t0.startswith('MOD') or t0.startswith('MO'):
                    paths.append(img_path); labels.append(3)

if len(paths) == 0:
    raise SystemExit("No validation images found. Make sure your `data_dir` contains class subfolders with images or files named with class prefixes like 'ND__', 'VMD__', 'MD__', 'MoD__'.")

print(f"Found {len(paths)} validation images")

# Run predictions in batches
preds = []
for p in paths:
    img = Image.open(p)
    arr = preprocess_pil(img)
    pred = model.predict(arr)
    preds.append(pred[0])

preds = np.array(preds)
labels = np.array(labels)

# Metrics
pred_labels = np.argmax(preds, axis=1)
acc = accuracy_score(labels, pred_labels)
print(f"Validation accuracy: {acc:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(labels, pred_labels))

print("\nClassification report:")
print(classification_report(labels, pred_labels, target_names=CLASS_NAMES, digits=4))

# Mean predicted probabilities
mean_probs = preds.mean(axis=0)
print("\nMean predicted probabilities across val set:")
for cls, m in zip(CLASS_NAMES, mean_probs):
    print(f"  {cls}: {m:.4f}")

# Check if predictions are near-uniform or near-prior
entropy_per_sample = -np.sum(preds * np.log(preds + 1e-12), axis=1)
print(f"\nMean prediction entropy: {entropy_per_sample.mean():.4f} (higher -> more uniform) )")

# Quick check: fraction where prediction equals training prior-most class
most_common_idx = np.argmax(mean_probs)
frac_most_common = np.mean(pred_labels == most_common_idx)
print(f"Fraction of predictions equal to most frequent predicted class ({CLASS_NAMES[most_common_idx]}): {frac_most_common:.4f}")

# Save a small CSV with image, true, pred, probs
import csv
out_csv = 'validation_predictions.csv'
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'true', 'pred'] + CLASS_NAMES)
    for p, t, pl, pr in zip(paths, labels, pred_labels, preds):
        writer.writerow([str(p), CLASS_NAMES[t], CLASS_NAMES[pl]] + pr.tolist())
print(f"Saved predictions to {out_csv}")
