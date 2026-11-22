
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset" / "split"

IMAGE_SIZE = (128, 128)

CLASSES = ["Healthy_leaf", "Unhealthy_leaf"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ==========================================
# 2. LOAD IMAGE
# ==========================================

def load_image(image_path):

    try:
        with Image.open(image_path) as image:

            # Convert image to RGB
            image = image.convert("RGB")

            # Resize image
            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            return image.copy()

    except (UnidentifiedImageError, OSError, ValueError) as error:

        raise ValueError(
            f"Could not process image: {image_path}"
        ) from error


# ==========================================
# 3. CONVERT IMAGE TO NUMPY ARRAY
# ==========================================

def image_to_array(image):

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    return image_array


# ==========================================
# 4. NORMALIZE IMAGE
# ==========================================

def normalize_image(image_array):

    normalized_array = image_array / 255.0

    return normalized_array


# ==========================================
# 5. FIND SAMPLE IMAGES
# ==========================================

def get_sample_image(class_name):

    class_folder = DATASET_DIR / "train" / class_name

    if not class_folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {class_folder}"
        )

    image_files = sorted([
        file for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        raise FileNotFoundError(
            f"No images found in {class_folder}"
        )

    return image_files[0]


# ==========================================
# 6. PROCESS SAMPLE IMAGES
# ==========================================

processed_images = []

for class_name in CLASSES:

    image_path = get_sample_image(class_name)

    print(f"\nProcessing: {class_name}")
    print(f"Image: {image_path.name}")

    # Load and resize
    image = load_image(image_path)

    # Convert to numerical array
    image_array = image_to_array(image)

    # Normalize pixel values
    normalized_array = normalize_image(image_array)

    # Display information
    print(f"Image dimensions: {image.size}")
    print(f"Array shape: {image_array.shape}")
    print(f"Data type: {image_array.dtype}")

    print(
        f"Original pixel range: "
        f"{image_array.min():.2f} - {image_array.max():.2f}"
    )

    print(
        f"Normalized pixel range: "
        f"{normalized_array.min():.4f} - "
        f"{normalized_array.max():.4f}"
    )

    processed_images.append(
        (class_name, image, normalized_array)
    )


# ==========================================
# 7. VISUALIZE RESULTS
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for row, (class_name, image, normalized_array) in enumerate(
    processed_images
):

    # Display resized image
    axes[row, 0].imshow(image)

    axes[row, 0].set_title(
        f"{class_name} - Resized"
    )

    axes[row, 0].axis("off")

    # Display normalized image
    axes[row, 1].imshow(normalized_array)

    axes[row, 1].set_title(
        f"{class_name} - Normalized"
    )

    axes[row, 1].axis("off")


plt.tight_layout()

plt.show()


print("\nImage preprocessing completed successfully!")
