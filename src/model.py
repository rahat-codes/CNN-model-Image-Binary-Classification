
import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------
# 1. Configuration
# --------------------------------

IMAGE_SIZE = (128, 128, 3)


# --------------------------------
# 2. Data augmentation
# --------------------------------

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")


# --------------------------------
# 3. Build CNN
# --------------------------------

model = models.Sequential([

    # Input
    layers.Input(shape=IMAGE_SIZE),

    # Normalize pixel values: 0-255 -> 0-1
    layers.Rescaling(1.0 / 255),

    # Augmentation (active during training)
    data_augmentation,

    # Convolution block 1
    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Convolution block 2
    layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Convolution block 3
    layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Classification head
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.4),

    # Binary output
    layers.Dense(1, activation="sigmoid")

], name="leaf_classifier")


# --------------------------------
# 4. Compile model
# --------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------
# 5. Display architecture
# --------------------------------

if __name__ == "__main__":
    model.summary()
