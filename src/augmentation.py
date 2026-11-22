
from pathlib import Path
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import random
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset" / "split"

RESULTS_DIR = PROJECT_ROOT / "results" / "augmentation"

IMAGE_SIZE = (128, 128)

CLASSES = ["Healthy_leaf", "Unhealthy_leaf"]

random.seed(42)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD SAMPLE IMAGE
# ==========================================

def load_sample_image(class_name):

    folder = DATASET_DIR / "train" / class_name

    images = [
        file for file in folder.iterdir()
        if file.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    if not images:
        raise FileNotFoundError(
            f"No images found in {folder}"
        )

    image_path = random.choice(images)

    with Image.open(image_path) as image:
        return image.convert("RGB").resize(IMAGE_SIZE)


# ==========================================
# 3. AUGMENTATION FUNCTIONS
# ==========================================

def rotate_image(image):
    angle = random.uniform(-25, 25)

    # Rotate the image
    rotated = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False
    )

    # Calculate a safe center crop
    width, height = rotated.size
    radians = math.radians(abs(angle))

    crop_width = int(
        width / (math.cos(radians) + math.sin(radians))
    )

    crop_height = int(
        height / (math.cos(radians) + math.sin(radians))
    )

    # Crop the center to remove empty corners
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    right = left + crop_width
    bottom = top + crop_height

    rotated = rotated.crop((left, top, right, bottom))

    # Restore original dimensions
    rotated = rotated.resize(
        (width, height),
        Image.Resampling.BICUBIC
    )

    return rotated


def flip_image(image):

    return image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )


def zoom_image(image):

    width, height = image.size

    zoom_factor = random.uniform(1.0, 1.2)

    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    zoomed = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    left = (new_width - width) // 2
    top = (new_height - height) // 2

    return zoomed.crop(
        (left, top, left + width, top + height)
    )


def change_brightness(image):

    factor = random.uniform(0.7, 1.3)

    enhancer = ImageEnhance.Brightness(image)

    return enhancer.enhance(factor)


# ==========================================
# 4. GENERATE AUGMENTED IMAGES
# ==========================================

def generate_augmentations(image):

    return [
        ("Original", image),
        ("Rotation", rotate_image(image)),
        ("Horizontal_Flip", flip_image(image)),
        ("Zoom", zoom_image(image)),
        ("Brightness", change_brightness(image))
    ]


# ==========================================
# 5. VISUALIZE RESULTS
# ==========================================

for class_name in CLASSES:

    original_image = load_sample_image(class_name)

    augmented_images = generate_augmentations(
        original_image
    )

    fig, axes = plt.subplots(
        1, 5,
        figsize=(16, 4)
    )

    fig.suptitle(
        f"Image Augmentation - {class_name}",
        fontsize=14
    )

    for index, (name, image) in enumerate(
        augmented_images
    ):

        axes[index].imshow(image)

        axes[index].set_title(name)

        axes[index].axis("off")

        if index > 0:

            save_path = (
                RESULTS_DIR /
                f"{class_name}_{name}.jpg"
            )

            image.save(save_path)

    plt.tight_layout()

    plt.show()

print("\nImage augmentation visualization completed!")
