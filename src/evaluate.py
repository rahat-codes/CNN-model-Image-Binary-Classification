import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow import keras

from data_loader import test_dataset


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_leaf_classifier.keras"

RESULTS_DIR = PROJECT_ROOT / "results"
PLOT_DIR = RESULTS_DIR / "plots"

CLASS_NAMES = ["Healthy_leaf", "Unhealthy_leaf"]

THRESHOLD = 0.5


# Create output directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

print("\nLoading trained model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 3. COLLECT ACTUAL LABELS AND PREDICTIONS
# ==========================================

print("\nEvaluating test dataset...")

actual_labels = []
predicted_probabilities = []

for images, labels in test_dataset:

    # Get model predictions
    probabilities = model(images, training=False).numpy().flatten()

    # Store actual labels
    actual_labels.extend(labels.numpy().flatten())

    # Store predicted probabilities
    predicted_probabilities.extend(probabilities)


# Convert lists into NumPy arrays
y_true = np.array(actual_labels).astype(int)

y_prob = np.array(predicted_probabilities)

# Convert probabilities into binary classes
y_pred = (y_prob >= THRESHOLD).astype(int)


# ==========================================
# 4. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

print("\n========== TEST RESULTS ==========")

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy (%): {accuracy * 100:.2f}%")


# Classification report
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)
report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ==========================================
# 5. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# Save structured evaluation metrics for the dashboard
metrics_path = RESULTS_DIR / "evaluation_metrics.json"

evaluation_metrics = {
    "accuracy": float(accuracy),
    "total_samples": int(len(y_true)),
    "threshold": float(THRESHOLD),
    "class_names": CLASS_NAMES,
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist()
}

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(evaluation_metrics, file, indent=4)

print(f"Structured evaluation metrics saved: {metrics_path}")

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 6. SAVE EVALUATION REPORT
# ==========================================

report_path = RESULTS_DIR / "evaluation_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("BINARY CNN CLASSIFICATION - TEST RESULTS\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Test Accuracy: {accuracy:.4f}\n")
    file.write(f"Test Accuracy (%): {accuracy * 100:.2f}%\n\n")

    file.write("Classification Report:\n")
    file.write(report)

    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))


print(f"\nEvaluation report saved: {report_path}")


# ==========================================
# 7. VISUALIZE CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix - Test Dataset")
plt.colorbar()

plt.xticks(
    [0, 1],
    CLASS_NAMES,
    rotation=15
)

plt.yticks(
    [0, 1],
    CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Display values inside the matrix
for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=14
        )

plt.tight_layout()

confusion_matrix_path = PLOT_DIR / "confusion_matrix.png"

plt.savefig(confusion_matrix_path, dpi=300)

plt.show()

print(f"Confusion matrix saved: {confusion_matrix_path}")

print("\nEvaluation completed successfully!")
