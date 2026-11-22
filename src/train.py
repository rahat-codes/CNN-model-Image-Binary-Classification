import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import (
    train_dataset,
    validation_dataset
)

from model import model


# --------------------------------
# 1. Project paths and settings
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 30

BEST_MODEL_PATH = MODEL_DIR / "best_leaf_classifier.keras"
FINAL_MODEL_PATH = MODEL_DIR / "leaf_classifier.keras"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"


# --------------------------------
# 2. Define training callbacks
# --------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(BEST_MODEL_PATH),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001,
    verbose=1
)

callbacks = [
    early_stopping,
    model_checkpoint,
    reduce_lr
]


# --------------------------------
# 3. Display model information
# --------------------------------

print("\nStarting CNN training...")
print("Maximum epochs:", EPOCHS)
print("Training batches:", tf.data.experimental.cardinality(
    train_dataset
).numpy())
print("Validation batches:", tf.data.experimental.cardinality(
    validation_dataset
).numpy())


# --------------------------------
# 4. Train the CNN
# --------------------------------

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# --------------------------------
# 5. Save the best model
# --------------------------------

model.save(FINAL_MODEL_PATH)

print("\nBest checkpoint:", BEST_MODEL_PATH)
print("Final saved model:", FINAL_MODEL_PATH)


# --------------------------------
# 6. Save training history as CSV
# --------------------------------

history_data = history.history

with open(
    HISTORY_PATH,
    mode="w",
    newline=""
) as file:

    writer = csv.writer(file)

    columns = list(history_data.keys())

    writer.writerow(["epoch"] + columns)

    for index in range(len(history_data["loss"])):

        row = [index + 1]

        for column in columns:
            row.append(history_data[column][index])

        writer.writerow(row)

print("Training history saved:", HISTORY_PATH)


# --------------------------------
# 7. Plot accuracy and loss
# --------------------------------

epochs_range = range(1, len(history_data["loss"]) + 1)

plt.figure(figsize=(12, 5))

# Accuracy graph
plt.subplot(1, 2, 1)

plt.plot(
    epochs_range,
    history_data["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    history_data["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)


# Loss graph
plt.subplot(1, 2, 2)

plt.plot(
    epochs_range,
    history_data["loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    history_data["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)


plt.tight_layout()

GRAPH_PATH = PLOTS_DIR / "training_history.png"

plt.savefig(GRAPH_PATH, dpi=300)
plt.show()

print("\nTraining graphs saved:", GRAPH_PATH)
print("\nTraining process completed!")
