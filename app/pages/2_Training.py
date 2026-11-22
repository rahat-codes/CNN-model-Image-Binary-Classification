
from pathlib import Path

import pandas as pd
import streamlit as st


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = PROJECT_ROOT / "results" / "training_history.csv"
GRAPH_PATH = PROJECT_ROOT / "results" / "plots" / "training_history.png"

BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_leaf_classifier.keras"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "leaf_classifier.keras"


# ==========================================
# 2. PAGE HEADER
# ==========================================

st.title("🧠 CNN Training Dashboard")
st.caption("Monitor and analyze your model's training performance.")

st.divider()


# ==========================================
# 3. CHECK TRAINING HISTORY
# ==========================================

if not HISTORY_PATH.exists():
    st.warning("Training history was not found.")
    st.info(
        "Train your CNN model first using option 4 in the terminal menu."
    )
    st.code("python main.py")
    st.stop()

try:
    history_df = pd.read_csv(HISTORY_PATH)
except Exception as error:
    st.error(f"Could not read training history: {error}")
    st.stop()


required_columns = [
    "epoch",
    "loss",
    "accuracy",
    "val_loss",
    "val_accuracy",
]

missing_columns = [
    column for column in required_columns
    if column not in history_df.columns
]

if missing_columns:
    st.error(
        f"Training history is missing required columns: {missing_columns}"
    )
    st.stop()

if history_df.empty:
    st.warning("The training history file contains no records.")
    st.stop()


# ==========================================
# 4. CALCULATE TRAINING STATISTICS
# ==========================================

best_epoch_index = history_df["val_loss"].idxmin()
best_epoch = int(history_df.loc[best_epoch_index, "epoch"])

best_val_loss = float(history_df.loc[best_epoch_index, "val_loss"])
best_val_accuracy = float(
    history_df.loc[best_epoch_index, "val_accuracy"]
)

best_train_accuracy = float(
    history_df.loc[best_epoch_index, "accuracy"]
)

total_epochs = len(history_df)

final_train_accuracy = float(history_df.iloc[-1]["accuracy"])
final_val_accuracy = float(history_df.iloc[-1]["val_accuracy"])

final_train_loss = float(history_df.iloc[-1]["loss"])
final_val_loss = float(history_df.iloc[-1]["val_loss"])


# ==========================================
# 5. MODEL STATUS
# ==========================================

st.subheader("Model Checkpoint Status")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Best Model",
        "Available" if BEST_MODEL_PATH.exists() else "Not Found",
    )

with col2:
    st.metric(
        "Final Model",
        "Available" if FINAL_MODEL_PATH.exists() else "Not Found",
    )

st.divider()


# ==========================================
# 6. TRAINING SUMMARY
# ==========================================

st.subheader("Training Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Completed Epochs", total_epochs)

with col2:
    st.metric("Best Epoch", best_epoch)

with col3:
    st.metric(
        "Best Validation Accuracy",
        f"{best_val_accuracy * 100:.2f}%",
    )

with col4:
    st.metric(
        "Best Validation Loss",
        f"{best_val_loss:.4f}",
    )

st.divider()


# ==========================================
# 7. ACCURACY VISUALIZATION
# ==========================================

st.subheader("Accuracy Performance")

accuracy_df = history_df.set_index("epoch")[
    ["accuracy", "val_accuracy"]
].rename(
    columns={
        "accuracy": "Training Accuracy",
        "val_accuracy": "Validation Accuracy",
    }
)

st.line_chart(
    accuracy_df,
    y=["Training Accuracy", "Validation Accuracy"],
    x_label="Epoch",
    y_label="Accuracy",
)

st.caption("Compare how training and validation accuracy change over time.")

st.divider()


# ==========================================
# 8. LOSS VISUALIZATION
# ==========================================

st.subheader("Loss Performance")

loss_df = history_df.set_index("epoch")[
    ["loss", "val_loss"]
].rename(
    columns={
        "loss": "Training Loss",
        "val_loss": "Validation Loss",
    }
)

st.line_chart(
    loss_df,
    y=["Training Loss", "Validation Loss"],
    x_label="Epoch",
    y_label="Loss",
)

st.caption("Lower loss generally indicates better prediction performance.")

st.divider()


# ==========================================
# 9. BEST EPOCH ANALYSIS
# ==========================================

st.subheader("Best Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Accuracy at Best Epoch",
        f"{best_train_accuracy * 100:.2f}%",
    )

    st.metric(
        "Validation Accuracy at Best Epoch",
        f"{best_val_accuracy * 100:.2f}%",
    )

with col2:
    st.metric(
        "Training Loss at Best Epoch",
        f"{float(history_df.loc[best_epoch_index, 'loss']):.4f}",
    )

    st.metric(
        "Validation Loss at Best Epoch",
        f"{best_val_loss:.4f}",
    )

st.info(
    f"Epoch {best_epoch} had the lowest validation loss "
    f"({best_val_loss:.4f}). This is the checkpoint-selection "
    f"criterion used by your ModelCheckpoint callback."
)

st.divider()


# ==========================================
# 10. FINAL EPOCH PERFORMANCE
# ==========================================

st.subheader("Final Epoch Performance")

col1, col2 = st.columns(2)

with col1:
    st.write("**Training**")
    st.metric("Final Training Accuracy", f"{final_train_accuracy * 100:.2f}%")
    st.metric("Final Training Loss", f"{final_train_loss:.4f}")

with col2:
    st.write("**Validation**")
    st.metric("Final Validation Accuracy", f"{final_val_accuracy * 100:.2f}%")
    st.metric("Final Validation Loss", f"{final_val_loss:.4f}")

st.divider()


# ==========================================
# 11. ORIGINAL TRAINING GRAPH
# ==========================================

st.subheader("Saved Training Visualization")

if GRAPH_PATH.exists():
    st.image(
        str(GRAPH_PATH),
        caption="Training and Validation Accuracy / Loss",
        use_container_width=True,
    )
else:
    st.warning("The original training graph was not found.")


# ==========================================
# 12. RAW TRAINING HISTORY
# ==========================================

with st.expander("View Epoch-by-Epoch Training Records"):
    st.dataframe(
        history_df,
        hide_index=True,
        use_container_width=True,
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Training History CSV",
        data=csv_data,
        file_name="training_history.csv",
        mime="text/csv",
    )


# ==========================================
# 13. FILE INFORMATION
# ==========================================

with st.expander("Training Artifact Locations"):
    st.write("**Training history:**")
    st.code(str(HISTORY_PATH))

    st.write("**Best model:**")
    st.code(str(BEST_MODEL_PATH))

    st.write("**Final model:**")
    st.code(str(FINAL_MODEL_PATH))

    st.write("**Training graph:**")
    st.code(str(GRAPH_PATH))
