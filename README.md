# 🍃 LeafVision — Apple Leaf Binary Classification Using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** to classify apple leaf images into two categories: **Healthy** and **Unhealthy**.

LeafVision provides a complete machine learning workflow, including dataset inspection, preprocessing, dataset splitting, model training, evaluation, and image prediction. It also includes a command-line interface (CLI) and an interactive Streamlit dashboard for visualizing the project.

---

## 📌 Project Overview

| Attribute | Details |
|---|---|
| Project Name | LeafVision |
| Project Type | Binary Image Classification |
| Machine Learning Task | Supervised Learning |
| Model Architecture | Convolutional Neural Network (CNN) |
| Framework | TensorFlow / Keras |
| Programming Language | Python |
| Image Input Size | 128 × 128 × 3 |
| Output Layer | Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Maximum Training Epochs | 30 |
| Frontend | Streamlit |

### Classification Categories

- **Healthy Leaf (Class 0):** An apple leaf classified as healthy.
- **Unhealthy Leaf (Class 1):** An apple leaf classified as unhealthy.

> Note: This model performs binary classification only. It does not identify specific apple diseases.

---

## 🎯 Project Objectives

The main objectives of this project are to:

1. Understand the complete workflow of an image classification project.
2. Learn how to organize and preprocess image datasets.
3. Build a CNN model using TensorFlow and Keras.
4. Understand convolution, pooling, feature extraction, and classification.
5. Train and validate a deep learning model.
6. Evaluate model performance using classification metrics.
7. Predict new images using a trained model.
8. Develop a command-line interface for executing project operations.
9. Build an interactive dashboard for visualizing model performance and predictions.

---

## 📂 Project Structure

```text
CNN-model-Image-Binary-Classification/
│
├── app/
│   ├── dashboard.py
│   │
│   ├── pages/
│   │   ├── 1_Dataset.py
│   │   ├── 2_Training.py
│   │   ├── 3_Evaluation.py
│   │   └── 4_Prediction.py
│   │
│   └── components/
│       └── __init__.py
│
├── dataset/
│   ├── original/
│   │   ├── Healthy_leaf/
│   │   └── Unhealthy_leaf/
│   │
│   └── split/
│       ├── train/
│       │   ├── Healthy_leaf/
│       │   └── Unhealthy_leaf/
│       │
│       ├── validation/
│       │   ├── Healthy_leaf/
│       │   └── Unhealthy_leaf/
│       │
│       └── test/
│           ├── Healthy_leaf/
│           └── Unhealthy_leaf/
│
├── models/
│   ├── best_leaf_classifier.keras
│   └── leaf_classifier.keras
│
├── results/
│   ├── plots/
│   │   ├── training_history.png
│   │   └── confusion_matrix.png
│   │
│   ├── training_history.csv
│   ├── evaluation_report.txt
│   └── evaluation_metrics.json
│
├── src/
│   ├── augmentation.py
│   ├── check_dataset.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── split_dataset.py
│   └── train.py
│
├── main.py
├── experiment.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

*The dataset and generated results directories may be absent from a fresh Git clone because they are excluded from version control. Generate or provide these locally before running the corresponding operations.*

---

## 📊 Dataset Information

The project uses a collection of **600 apple leaf images** distributed equally between the two classes.

### Original Dataset

| Class | Number of Images |
|---|---:|
| Healthy Leaf | 300 |
| Unhealthy Leaf | 300 |
| **Total** | **600** |

### Dataset Splitting

The dataset is divided into three subsets:

| Dataset | Percentage | Healthy | Unhealthy | Total |
|---|---:|---:|---:|---:|
| Training | 70% | 210 | 210 | 420 |
| Validation | 15% | 45 | 45 | 90 |
| Testing | 15% | 45 | 45 | 90 |
| **Total** | **100%** | **300** | **300** | **600** |

### Purpose of Each Subset

- **Training:** Used to learn patterns and features from apple leaf images.
- **Validation:** Used to monitor model performance during training and support model selection.
- **Testing:** Used to evaluate the trained model on images not used for training or validation.

---

## ⚙️ Image Preprocessing

Before training, images are prepared for the CNN model.

The preprocessing pipeline includes:

1. Loading images from their respective class directories.
2. Resizing images to `128 × 128` pixels.
3. Converting images into RGB format.
4. Converting images into numerical tensors.
5. Normalizing pixel values using a Keras `Rescaling` layer.

### Pixel Normalization

Original image pixel values range from 0 to 255.

The model uses:

\[
x_{\text{normalized}} = \frac{x}{255}
\]

This transforms pixel values into the range:

\[
0 \leq x_{\text{normalized}} \leq 1
\]

Normalization is implemented inside the model, so prediction images should not be normalized a second time.

---

## 🔄 Data Augmentation

Data augmentation introduces controlled variations into training images to help the model learn more robust visual features.

The project uses:

| Augmentation | Purpose |
|---|---|
| Random Flip | Changes horizontal orientation |
| Random Rotation | Introduces slight rotational variations |
| Random Zoom | Changes image scale |
| Random Contrast | Adjusts image contrast |

These transformations are applied during training rather than permanently modifying the original dataset.

---

## 🧠 CNN Model Architecture

The LeafVision model uses a sequential CNN architecture.

### Architecture Overview

```text
Input Image
(128 × 128 × 3)
       |
       ▼
Rescaling (1./255)
       |
       ▼
Data Augmentation
       |
       ▼
Conv2D (32 filters)
       |
       ▼
MaxPooling2D
       |
       ▼
Conv2D (64 filters)
       |
       ▼
MaxPooling2D
       |
       ▼
Conv2D (128 filters)
       |
       ▼
MaxPooling2D
       |
       ▼
GlobalAveragePooling2D
       |
       ▼
Dense (128, ReLU)
       |
       ▼
Dropout (0.4)
       |
       ▼
Dense (1, Sigmoid)
       |
       ▼
Binary Classification
Healthy / Unhealthy
```

### Layer Explanation

| Layer | Function |
|---|---|
| Rescaling | Normalizes image pixel values |
| Conv2D | Extracts visual features from images |
| MaxPooling2D | Reduces spatial dimensions |
| GlobalAveragePooling2D | Converts feature maps into a compact representation |
| Dense | Learns higher-level classification patterns |
| Dropout | Helps reduce overfitting |
| Sigmoid | Produces a probability for the positive class |

### Model Configuration

| Parameter | Value |
|---|---|
| Input Shape | `(128, 128, 3)` |
| Convolution Filters | 32, 64, 128 |
| Dense Units | 128 |
| Dropout Rate | 0.4 |
| Output Units | 1 |
| Activation | Sigmoid |
| Loss | Binary Crossentropy |
| Optimizer | Adam |
| Maximum Epochs | 30 |

**Total trainable and non-trainable parameters:** 109,889.

---

## 🚀 Model Training

Training is performed using TensorFlow and Keras.

The model learns from the training dataset while its performance is monitored using the validation dataset.

### Training Callbacks

Three callbacks are implemented:

**1. EarlyStopping**

Monitors validation loss and stops training when the model stops improving. It also restores the best weights.

**2. ModelCheckpoint**

Saves the model with the best validation loss.

**3. ReduceLROnPlateau**

Reduces the learning rate when validation loss stops improving.

### Generated Training Artifacts

After training, the project generates:

- `models/best_leaf_classifier.keras`
- `models/leaf_classifier.keras`
- `results/training_history.csv`
- `results/plots/training_history.png`

The training history includes loss and accuracy values for both training and validation datasets.

---

## 📈 Model Evaluation

The model is evaluated using the separate test dataset.

### Test Results

<details>
<summary><strong>View evaluation results</strong></summary>

| Metric | Result |
|---|---:|
| Test Images | 90 |
| Correct Predictions | 77 |
| Incorrect Predictions | 13 |
| Test Accuracy | **85.56%** |
| Macro F1-score | **Approximately 0.8547** |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Healthy Leaf | 0.9211 | 0.7778 | 0.8434 |
| Unhealthy Leaf | 0.8077 | 0.9333 | 0.8660 |

### Confusion Matrix

```text
                 Predicted
                 Healthy  Unhealthy

Actual Healthy      35       10
Actual Unhealthy     3       42
```

### Interpretation

- 35 healthy leaves were correctly classified.
- 10 healthy leaves were incorrectly classified as unhealthy.
- 42 unhealthy leaves were correctly classified.
- 3 unhealthy leaves were incorrectly classified as healthy.

The model achieved a test accuracy of 85.56% on this dataset.

However, the test set contains only 90 images. These results should be treated as an initial experimental benchmark, not proof of reliable real-world performance.

</details>

---

## 🖥️ Interactive Dashboard

LeafVision includes a Streamlit frontend that provides a visual interface for exploring the machine learning workflow.

### Dashboard Features

| Page | Functionality |
|---|---|
| Overview | Displays system status and dataset distribution |
| Dataset | Visualizes dataset structure, class distribution, and sample images |
| Training | Displays training history, loss, and accuracy |
| Evaluation | Displays classification metrics and confusion matrix |
| Prediction | Uploads an image and displays its predicted class and probabilities |

### Dashboard Technologies

- Streamlit
- Pandas
- Matplotlib
- TensorFlow / Keras
- NumPy
- Pillow

---

## 💻 Command-Line Interface

The project also includes a menu-driven CLI through `main.py`.

It allows users to execute individual machine learning operations without manually running each Python script.

### Available Operations

| Option | Operation |
|---|---|
| 1 | Check Original Dataset |
| 2 | Preprocess Sample Images |
| 3 | Split Dataset |
| 4 | Train CNN Model |
| 5 | Evaluate CNN Model |
| 6 | Predict Test Images |
| 0 | Exit |

The CLI uses Python's `subprocess` module to execute the existing source scripts and displays execution status, elapsed time, and exit codes.

---

## 🛠️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rahat-codes/CNN-model-Image-Binary-Classification.git

cd CNN-model-Image-Binary-Classification
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the original images into the following directories:

```text
dataset/
└── original/
    ├── Healthy_leaf/
    └── Unhealthy_leaf/
```

Make sure each folder contains the appropriate images.

### 5. Run the CLI

```bash
python main.py
```

### 6. Run the Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Streamlit will provide a local URL in the terminal, normally:

```text
http://localhost:8501
```

---

## 🔬 Running Individual Scripts

You can also execute individual operations directly.

```bash
# Inspect the original dataset
python src/check_dataset.py

# Preprocess sample images
python src/preprocess.py

# Split the dataset
python src/split_dataset.py

# Train the CNN model
python src/train.py

# Evaluate the model
python src/evaluate.py

# Predict images
python src/predict.py
```

Run these operations in the appropriate order when starting from a fresh dataset.

---

## 📦 Dependencies

The project uses the following Python libraries:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

Potential future improvements include:

- Increase dataset size and diversity.
- Evaluate the model on independent external datasets.
- Experiment with deeper CNN architectures.
- Compare CNN performance with transfer learning models such as MobileNetV2 and EfficientNet.
- Apply hyperparameter tuning.
- Improve model generalization.
- Add Grad-CAM visualization to explain model predictions.
- Add prediction history and downloadable reports.
- Compare precision, recall, F1-score, and inference time across models.
- Investigate probability calibration and decision thresholds.

---

## 📚 Learning Outcomes

Through this project, I explored:

- Image classification using deep learning.
- CNN architecture and feature extraction.
- TensorFlow and Keras model implementation.
- Image preprocessing and data augmentation.
- Training, validation, and testing workflows.
- Model evaluation using classification metrics.
- Python project organization and modular programming.
- CLI development using subprocess execution.
- Interactive machine learning visualization using Streamlit.

---

## 👨‍💻 Author

**Rahat Md Mahadi Hasan**

AI and Computer Engineering Student  
Tongmyong University, Busan, South Korea

- GitHub: [@rahat-codes](https://github.com/rahat-codes)
- LinkedIn: [Rahat Md Mahadi Hasan](https://linkedin.com/in/rahat-mahadi-hasan)

---

## 📄 License

This project is intended for educational and research purposes.

---

**LeafVision — From Image Data to Intelligent Classification.** 🍃
