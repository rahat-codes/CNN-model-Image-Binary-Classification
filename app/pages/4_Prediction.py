
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_leaf_classifier.keras"

IMAGE_SIZE = (128, 128)
THRESHOLD = 0.5

CLASS_NAMES = {
    0: "Healthy Leaf",
    1: "Unhealthy Leaf",
}


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.title("🔍 Interactive Image Prediction")

st.caption(
    "Upload an apple leaf image and let your trained CNN classify it."
)

st.divider()


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    """Load and cache the trained Keras model."""
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Train your CNN model first.")
    st.stop()

try:
    model = load_model()

except Exception as error:
    st.error(f"Could not load the CNN model: {error}")
    st.stop()


# ==========================================
# 4. IMAGE UPLOAD
# ==========================================

st.subheader("Upload an Apple Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload a clear image of an apple leaf.",
)

if uploaded_file is None:
    st.info("Upload an image to begin classification.")

    st.markdown(
        """
        **Supported formats:** JPG, JPEG, PNG, BMP, WEBP

        **Model input size:** 128 × 128 pixels

        **Classification:** Healthy Leaf / Unhealthy Leaf
        """
    )

    st.stop()


# ==========================================
# 5. READ AND DISPLAY IMAGE
# ==========================================

try:
    image = Image.open(uploaded_file).convert("RGB")

except Exception as error:
    st.error(f"Could not read the uploaded image: {error}")
    st.stop()

st.subheader("Image Preview")

col1, col2 = st.columns(2)

with col1:
    st.image(
        image,
        caption="Original Uploaded Image",
        use_container_width=True,
    )

with col2:
    st.markdown("#### Image Information")

    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Original dimensions:** {image.width} × {image.height}")
    st.write("**Color channels:** RGB")
    st.write(f"**Model input:** {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}")

st.divider()


# ==========================================
# 6. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):
    """
    Prepare an image for CNN prediction.

    The model already contains a Rescaling(1./255) layer,
    so do not normalize the image a second time here.
    """

    resized_image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array


# ==========================================
# 7. RUN PREDICTION
# ==========================================

st.subheader("CNN Classification")

if st.button(
    "🔬 Classify Image",
    type="primary",
    use_container_width=True,
):

    try:
        with st.spinner("CNN is analyzing your image..."):

            processed_image = preprocess_image(image)

            prediction = model.predict(
                processed_image,
                verbose=0,
            )

            unhealthy_probability = float(
                np.asarray(prediction).reshape(-1)[0]
            )

            if not 0 <= unhealthy_probability <= 1:
                raise ValueError(
                    "Model output is outside the expected sigmoid range."
                )

            healthy_probability = 1 - unhealthy_probability

            if unhealthy_probability >= THRESHOLD:
                predicted_class = CLASS_NAMES[1]
                confidence = unhealthy_probability
            else:
                predicted_class = CLASS_NAMES[0]
                confidence = healthy_probability

        # Store prediction for display
        st.session_state["prediction_result"] = {
            "class": predicted_class,
            "confidence": confidence,
            "healthy_probability": healthy_probability,
            "unhealthy_probability": unhealthy_probability,
        }

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ==========================================
# 8. DISPLAY PREDICTION RESULT
# ==========================================

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    st.divider()

    st.subheader("Prediction Result")

    if result["class"] == "Healthy Leaf":
        st.success("🍃 Classification: Healthy Leaf")
    else:
        st.warning("⚠️ Classification: Unhealthy Leaf")

    st.metric(
        "Prediction Confidence",
        f"{result['confidence'] * 100:.2f}%",
    )

    st.divider()

    # Probability comparison
    st.subheader("Class Probabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Healthy Leaf**")
        st.metric(
            "Probability",
            f"{result['healthy_probability'] * 100:.2f}%",
        )
        st.progress(result["healthy_probability"])

    with col2:
        st.write("**Unhealthy Leaf**")
        st.metric(
            "Probability",
            f"{result['unhealthy_probability'] * 100:.2f}%",
        )
        st.progress(result["unhealthy_probability"])

    st.divider()

    st.caption(
        "Prediction threshold: 0.50. "
        "Confidence represents the model's output probability, "
        "not a guarantee that the classification is correct."
    )

    if st.button("Clear Prediction"):
        del st.session_state["prediction_result"]
        st.rerun()
