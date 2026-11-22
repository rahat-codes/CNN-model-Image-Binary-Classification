
import random
import shutil
from pathlib import Path
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "dataset" / "original"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "split"

CLASSES = ["Healthy_leaf", "Unhealthy_leaf"]

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ==========================================
# 2. VALIDATE SOURCE DATASET
# ==========================================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source dataset not found: {SOURCE_DIR}"
    )

if not abs(sum(SPLIT_RATIOS.values()) - 1.0) < 1e-9:
    raise ValueError("Split ratios must add up to 1.")

# Prevent accidental mixing when running twice.
if OUTPUT_DIR.exists() and any(
    path.is_file() for path in OUTPUT_DIR.rglob("*")
):
    raise FileExistsError(
        f"Output dataset already contains files: {OUTPUT_DIR}\n"
        "Remove the existing dataset/split folder before "
        "running this script again."
    )

random_generator = random.Random(RANDOM_SEED)

# ==========================================
# 3. CREATE OUTPUT DIRECTORIES
# ==========================================

for split_name in SPLIT_RATIOS:

    for class_name in CLASSES:

        folder = OUTPUT_DIR / split_name / class_name

        folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# 4. SPLIT IMAGES
# ==========================================

total_images = 0

print("\nSTARTING DATASET SPLITTING")
print("=" * 55)

for class_name in CLASSES:

    class_path = SOURCE_DIR / class_name

    if not class_path.exists():
        raise FileNotFoundError(
            f"Missing class directory: {class_path}"
        )

    image_files = [
        file for file in class_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Keep only images that can be opened.
    valid_images = []

    for image_path in image_files:

        try:
            with Image.open(image_path) as image:
                image.verify()

            valid_images.append(image_path)

        except (OSError, ValueError):
            print(f"[SKIPPED] Invalid image: {image_path.name}")

    # Randomize image order reproducibly.
    random_generator.shuffle(valid_images)

    image_count = len(valid_images)

    train_count = round(image_count * 0.70)
    validation_count = round(image_count * 0.15)

    train_images = valid_images[:train_count]

    validation_images = valid_images[
        train_count:train_count + validation_count
    ]

    test_images = valid_images[
        train_count + validation_count:
    ]

    split_data = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }

    print(f"\nClass: {class_name}")
    print(f"Original images: {image_count}")

    # Copy images into their respective folders.
    for split_name, images in split_data.items():

        destination = OUTPUT_DIR / split_name / class_name

        for image_path in images:

            shutil.copy2(
                image_path,
                destination / image_path.name
            )

        print(f"{split_name}: {len(images)}")

    total_images += image_count

# ==========================================
# 5. FINAL REPORT
# ==========================================

print("\n" + "=" * 55)
print("DATASET SPLITTING COMPLETED")
print("=" * 55)

print(f"Total images processed: {total_images}")

for split_name in SPLIT_RATIOS:

    split_total = sum(
        1 for class_name in CLASSES
        for file in (OUTPUT_DIR / split_name / class_name).iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"{split_name.capitalize()}: {split_total}")

print("=" * 55)
