
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_PATH = PROJECT_ROOT / "results" / "evaluation_metrics.json"
REPORT_PATH = PROJECT_ROOT / "results" / "evaluation_report.txt"
CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT / "results" / "plots" / "confusion_matrix.png"
)


# ==========================================
# 2. PAGE HEADER
# ==========================================

st.title("📊 Model Evaluation")
st.caption("Analyze the performance of your trained CNN on test images.")

st.divider()


# ==========================================
# 3. LOAD EVALUATION DATA
# ==========================================

if not METRICS_PATH.exists():
    st.warning("Structured evaluation results were not found.")

    st.info(
        "Run the evaluation script first using option 5 "
        "in your terminal menu."
    )

    st.code("python main.py")
    st.stop()

try:
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

except (json.JSONDecodeError, OSError) as error:
    st.error(f"Could not load evaluation metrics: {error}")
    st.stop()


# ==========================================
# 4. EXTRACT METRICS
# ==========================================

accuracy = metrics["accuracy"]
total_samples = metrics["total_samples"]
threshold = metrics["threshold"]

class_names = metrics["class_names"]
report = metrics["classification_report"]
confusion_matrix = metrics["confusion_matrix"]


# ==========================================
# 5. OVERALL PERFORMANCE
# ==========================================

st.subheader("Overall Test Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%",
    )

with col2:
    st.metric(
        "Test Images",
        total_samples,
    )

with col3:
    st.metric(
        "Classification Threshold",
        f"{threshold:.2f}",
    )

correct_predictions = sum(
    confusion_matrix[index][index]
    for index in range(len(class_names))
)

incorrect_predictions = total_samples - correct_predictions

col1, col2 = st.columns(2)

with col1:
    st.metric("Correct Predictions", correct_predictions)

with col2:
    st.metric("Incorrect Predictions", incorrect_predictions)

st.progress(accuracy)

st.divider()


# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

st.subheader("Confusion Matrix")

st.caption(
    "Rows represent actual classes. "
    "Columns represent predicted classes."
)

if (
    len(confusion_matrix) != len(class_names)
    or any(len(row) != len(class_names) for row in confusion_matrix)
):
    st.error("Confusion matrix dimensions do not match class names.")
    st.stop()

fig, ax = plt.subplots(figsize=(7, 5))

image = ax.imshow(
    confusion_matrix,
    interpolation="nearest",
    cmap="Blues",
)

fig.colorbar(image, ax=ax)

ax.set(
    xticks=range(len(class_names)),
    yticks=range(len(class_names)),
    xticklabels=[name.replace("_", " ") for name in class_names],
    yticklabels=[name.replace("_", " ") for name in class_names],
    xlabel="Predicted Class",
    ylabel="Actual Class",
    title="CNN Test Confusion Matrix",
)

plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

for row in range(len(class_names)):
    for column in range(len(class_names)):
        ax.text(
            column,
            row,
            str(confusion_matrix[row][column]),
            ha="center",
            va="center",
            color="black",
            fontsize=13,
        )

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.divider()


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

st.subheader("Classification Report")

class_metrics = []

for class_name in class_names:
    class_data = report[class_name]

    class_metrics.append({
        "Class": class_name.replace("_", " "),
        "Precision": class_data["precision"],
        "Recall": class_data["recall"],
        "F1-Score": class_data["f1-score"],
        "Support": int(class_data["support"]),
    })

metrics_df = pd.DataFrame(class_metrics)

st.dataframe(
    metrics_df.style.format({
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}",
    }),
    hide_index=True,
    use_container_width=True,
)

st.divider()


# ==========================================
# 8. MACRO AND WEIGHTED AVERAGES
# ==========================================

st.subheader("Overall Class Metrics")

macro_avg = report["macro avg"]
weighted_avg = report["weighted avg"]

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Macro Average")

    st.metric(
        "Macro Precision",
        f"{macro_avg['precision']:.4f}",
    )

    st.metric(
        "Macro Recall",
        f"{macro_avg['recall']:.4f}",
    )

    st.metric(
        "Macro F1-Score",
        f"{macro_avg['f1-score']:.4f}",
    )

with col2:
    st.markdown("#### Weighted Average")

    st.metric(
        "Weighted Precision",
        f"{weighted_avg['precision']:.4f}",
    )

    st.metric(
        "Weighted Recall",
        f"{weighted_avg['recall']:.4f}",
    )

    st.metric(
        "Weighted F1-Score",
        f"{weighted_avg['f1-score']:.4f}",
    )

st.divider()


# ==========================================
# 9. ORIGINAL EVALUATION ARTIFACTS
# ==========================================

st.subheader("Saved Evaluation Artifacts")

if CONFUSION_MATRIX_PATH.exists():
    with st.expander("View Original Confusion Matrix Image"):
        st.image(
            str(CONFUSION_MATRIX_PATH),
            use_container_width=True,
        )

if REPORT_PATH.exists():
    with st.expander("View Original Text Evaluation Report"):
        report_text = REPORT_PATH.read_text(encoding="utf-8")
        st.code(report_text, language="text")


# ==========================================
# 10. DOWNLOAD RESULTS
# ==========================================

st.subheader("Download Evaluation Results")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "Download Metrics JSON",
        data=json.dumps(metrics, indent=4),
        file_name="evaluation_metrics.json",
        mime="application/json",
    )

with col2:
    if REPORT_PATH.exists():
        st.download_button(
            "Download Evaluation Report",
            data=REPORT_PATH.read_bytes(),
            file_name="evaluation_report.txt",
            mime="text/plain",
        )

st.success("Evaluation dashboard loaded successfully.")
