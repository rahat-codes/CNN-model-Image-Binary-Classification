
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset" / "split"

CLASSES = ["Healthy_leaf", "Unhealthy_leaf"]
SPLITS = ["train", "validation", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_images(folder):
    """Return image files from a folder."""
    if not folder.exists():
        return []

    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def get_dataset_statistics():
    """Count images in every dataset split and class."""
    records = []

    for split in SPLITS:
        record = {"Split": split}

        for class_name in CLASSES:
            folder = DATASET_DIR / split / class_name
            record[class_name] = len(get_images(folder))

        record["Total"] = sum(record[name] for name in CLASSES)
        records.append(record)

    return pd.DataFrame(records)


# ==========================================
# 3. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Dataset Explorer | LeafVision",
    page_icon="🗂️",
    layout="wide",
)

st.title("🗂️ Dataset Explorer")
st.caption("Explore and analyze your Apple Leaf image dataset.")

st.divider()


# ==========================================
# 4. CHECK DATASET DIRECTORY
# ==========================================

if not DATASET_DIR.exists():
    st.error(f"Dataset directory not found: {DATASET_DIR}")
    st.info("Run the dataset splitting script first.")
    st.stop()

df = get_dataset_statistics()


# ==========================================
# 5. OVERALL DATASET METRICS
# ==========================================

total_healthy = int(df["Healthy_leaf"].sum())
total_unhealthy = int(df["Unhealthy_leaf"].sum())
total_images = total_healthy + total_unhealthy

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Images", total_images)

with col2:
    st.metric("Healthy Images", total_healthy)

with col3:
    st.metric("Unhealthy Images", total_unhealthy)

with col4:
    st.metric("Dataset Splits", len(SPLITS))

st.divider()


# ==========================================
# 6. DATASET DISTRIBUTION
# ==========================================

st.subheader("Dataset Distribution")

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("#### Images by Split")

    chart_data = df.set_index("Split")[["Healthy_leaf", "Unhealthy_leaf"]]

    st.bar_chart(chart_data)

with right_col:
    st.markdown("#### Split Statistics")

    display_df = df.rename(
        columns={
            "Healthy_leaf": "Healthy",
            "Unhealthy_leaf": "Unhealthy",
        }
    )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
    )

st.divider()


# ==========================================
# 7. CLASS DISTRIBUTION
# ==========================================

st.subheader("Class Distribution")

if total_images > 0:
    healthy_percentage = total_healthy / total_images * 100
    unhealthy_percentage = total_unhealthy / total_images * 100

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"Healthy: {healthy_percentage:.2f}%")
        st.progress(healthy_percentage / 100)

    with col2:
        st.write(f"Unhealthy: {unhealthy_percentage:.2f}%")
        st.progress(unhealthy_percentage / 100)
else:
    st.warning("No images were found in the dataset.")

st.divider()


# ==========================================
# 8. IMAGE GALLERY
# ==========================================

st.subheader("Sample Image Gallery")

gallery_col1, gallery_col2, gallery_col3 = st.columns(3)

with gallery_col1:
    selected_split = st.selectbox(
        "Select Dataset Split",
        SPLITS,
    )

with gallery_col2:
    selected_class = st.selectbox(
        "Select Class",
        CLASSES,
        format_func=lambda value: value.replace("_", " "),
    )

with gallery_col3:
    number_of_images = st.selectbox(
        "Number of Samples",
        [2, 4, 6, 8],
        index=1,
    )

image_folder = DATASET_DIR / selected_split / selected_class
sample_images = get_images(image_folder)[:number_of_images]

if sample_images:
    st.caption(
        f"Showing {len(sample_images)} images from "
        f"{selected_split}/{selected_class}"
    )

    gallery_columns = st.columns(4)

    for index, image_path in enumerate(sample_images):
        try:
            with gallery_columns[index % 4]:
                with Image.open(image_path) as image:
                    st.image(
                        image.copy(),
                        caption=image_path.name,
                        use_container_width=True,
                    )
        except Exception as error:
            st.warning(f"Could not load {image_path.name}: {error}")

else:
    st.warning("No images found for this selection.")


# ==========================================
# 9. DATASET INFORMATION
# ==========================================

st.divider()

with st.expander("Dataset Directory Information"):
    st.code(str(DATASET_DIR), language="text")

    for split in SPLITS:
        st.write(f"**{split.capitalize()}**")

        for class_name in CLASSES:
            folder = DATASET_DIR / split / class_name
            count = len(get_images(folder))
            st.write(f"- {class_name}: {count} images")
