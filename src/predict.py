
from pathlib import Path
import shutil
import tensorflow as tf

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_leaf_classifier.keras"
TEST_DIR = PROJECT_ROOT / "dataset" / "split" / "test"
RESULTS_DIR = PROJECT_ROOT / "results" / "predictions"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
CLASS_NAMES = ["Healthy_leaf", "Unhealthy_leaf"]
THRESHOLD = 0.5


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

def load_model():
    print("Loading trained model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    return model


# --------------------------------------------------
# 3. LOAD TEST IMAGES
# --------------------------------------------------

def load_test_images():
    print("Loading test images...")

    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels=None,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    file_paths = dataset.file_paths

    return dataset, file_paths


# --------------------------------------------------
# 4. CLEAR OLD RESULTS
# --------------------------------------------------

def prepare_result_folders():
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    for class_name in CLASS_NAMES:
        (RESULTS_DIR / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

    print("Result folders prepared.")


# --------------------------------------------------
# 5. CLASSIFY AND COPY IMAGES
# --------------------------------------------------

def classify_and_copy(model, dataset, file_paths):

    image_index = 0
    healthy_count = 0
    unhealthy_count = 0

    print("\nClassifying images...\n")

    for batch in dataset:

        probabilities = model.predict(
            batch,
            verbose=0
        ).flatten()

        for probability in probabilities:

            # Sigmoid probability:
            # >= 0.5 means Unhealthy
            if probability >= THRESHOLD:
                predicted_class = "Unhealthy_leaf"
                unhealthy_count += 1
            else:
                predicted_class = "Healthy_leaf"
                healthy_count += 1

            original_path = Path(file_paths[image_index])

            destination_folder = RESULTS_DIR / predicted_class
            destination_path = destination_folder / original_path.name

            # Copy the original image
            shutil.copy2(original_path, destination_path)

            image_index += 1

    return healthy_count, unhealthy_count


# --------------------------------------------------
# 6. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 50)
    print("APPLE LEAF IMAGE CLASSIFICATION")
    print("=" * 50)

    # Check required files
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

    prepare_result_folders()

    model = load_model()

    dataset, file_paths = load_test_images()

    healthy_count, unhealthy_count = classify_and_copy(
        model,
        dataset,
        file_paths
    )

    print("\n" + "=" * 50)
    print("CLASSIFICATION COMPLETED")
    print("=" * 50)

    print(f"Total images: {healthy_count + unhealthy_count}")
    print(f"Predicted Healthy: {healthy_count}")
    print(f"Predicted Unhealthy: {unhealthy_count}")

    print(f"\nResults saved in:\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()
