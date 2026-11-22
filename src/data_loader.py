
import tensorflow as tf
from pathlib import Path

# -------------------------------
# 1. Configuration
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "split"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


# -------------------------------
# 2. Load training dataset
# -------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)


# -------------------------------
# 3. Load validation dataset
# -------------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


# -------------------------------
# 4. Load testing dataset
# -------------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Preserve test image filenames for error analysis
test_file_paths = test_dataset.file_paths

print("\nTotal test image paths:", len(test_file_paths))
print("First test image:", test_file_paths[0])


# -------------------------------
# 5. Check class labels
# -------------------------------

class_names = train_dataset.class_names

print("\nClass names:", class_names)
print("Class mapping:", {
    name: index for index, name in enumerate(class_names)
})


# -------------------------------
# 6. Optimize data loading
# -------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# -------------------------------
# 7. Inspect one batch
# -------------------------------

for images, labels in train_dataset.take(1):

    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image data type:", images.dtype)
    print("Pixel value range:",
          tf.reduce_min(images).numpy(),
          "to",
          tf.reduce_max(images).numpy())

    print("\nFirst 10 labels:")
    print(labels[:10].numpy().flatten())
