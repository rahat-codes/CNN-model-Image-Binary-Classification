
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset location
DATASET_PATH = PROJECT_ROOT / "dataset" / "original"

# Our two classes
CLASSES = ["Healthy_leaf", "Unhealthy_leaf"]

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

total_valid = 0
total_invalid = 0

print("\nPLANT LEAF DATASET CHECKER")
print("=" * 50)

for class_name in CLASSES:

    class_path = DATASET_PATH / class_name

    if not class_path.exists():
        print(f"\n[ERROR] Folder not found: {class_name}")
        continue

    image_files = [
        file for file in class_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    valid_count = 0
    invalid_count = 0

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                img.verify()

            valid_count += 1

        except (UnidentifiedImageError, OSError, ValueError):
            invalid_count += 1
            print(f"[INVALID] {image_path.name}")

    total_valid += valid_count
    total_invalid += invalid_count

    print(f"\nClass: {class_name}")
    print(f"Valid images: {valid_count}")
    print(f"Invalid images: {invalid_count}")

print("\n" + "=" * 50)
print("FINAL DATASET REPORT")
print("=" * 50)

print(f"Total valid images: {total_valid}")
print(f"Total invalid images: {total_invalid}")

print("\nDataset distribution:")

if total_valid > 0:
    for class_name in CLASSES:
        folder = DATASET_PATH / class_name

        if folder.exists():
            count = sum(
                1 for f in folder.iterdir()
                if f.is_file()
                and f.suffix.lower() in IMAGE_EXTENSIONS
            )

            percentage = count / total_valid * 100

            print(
                f"{class_name}: "
                f"{count} ({percentage:.2f}%)"
            )

print("=" * 50)
