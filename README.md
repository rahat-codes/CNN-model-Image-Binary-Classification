# 🍃 LeafVision — Apple Leaf Classification Using CNN

**An end-to-end deep learning project for binary apple leaf image classification using Convolutional Neural Networks (CNN).**

LeafVision classifies apple leaf images into two categories — Healthy and Unhealthy — using a custom CNN built with TensorFlow and Keras.

The project includes a complete machine learning pipeline, a command-line interface, and an interactive Streamlit dashboard for exploring datasets, training results, model evaluation, and image predictions.

<p align="center">

  **🌐 Live Demo**

  [🚀 Open LeafVision Web Application](https://cnn-model-by-rahat.streamlit.app/)

</p>

---

## 🚀 Live Application

Try the deployed application:

**[https://cnn-model-by-rahat.streamlit.app/](https://cnn-model-by-rahat.streamlit.app/)**

The web application provides:

- Interactive project dashboard
- Dataset distribution visualization
- Training history visualization
- Model evaluation metrics
- Confusion matrix
- Apple leaf image upload
- Healthy / Unhealthy classification
- Prediction probabilities

---

## 📌 Project Overview

| Attribute | Details |
|---|---|
| Project | LeafVision |
| Task | Binary Image Classification |
| Domain | Computer Vision |
| Algorithm | Convolutional Neural Network |
| Framework | TensorFlow / Keras |
| Language | Python |
| Frontend | Streamlit |
| Image Input | 128 × 128 × 3 |
| Output Activation | Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Maximum Epochs | 30 |
| Deployment | Streamlit Community Cloud |

### Classification Classes

| Label | Class |
|---|---|
| 0 | Healthy Leaf |
| 1 | Unhealthy Leaf |

**Important:** This model distinguishes healthy from unhealthy leaves. It does not identify specific diseases or provide a confirmed agricultural diagnosis.

---

## 🎯 Project Objectives

The objectives of this project are to:

1. Understand the end-to-end workflow of an image classification system.
2. Implement image preprocessing and data augmentation.
3. Design and train a CNN using TensorFlow and Keras.
4. Understand convolutional layers and feature extraction.
5. Evaluate model performance using classification metrics.
6. Build a reusable image prediction pipeline.
7. Develop a menu-driven command-line interface.
8. Create an interactive machine learning dashboard.
9. Deploy the application to a publicly accessible web platform.

---

## 🖥️ Application Dashboard

The Streamlit application is organized into five main sections.

| Page | Description |
|---|---|
| Overview | Project overview, system status, and dataset distribution |
| Dataset | Dataset structure, class distribution, and sample images |
| Training | Training accuracy, validation accuracy, loss, and training history |
| Evaluation | Classification report, performance metrics, and confusion matrix |
| Prediction | Upload an apple leaf image and view the model prediction |

**Try it:** [Open the live LeafVision dashboard](https://cnn-model-by-rahat.streamlit.app/)

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
│       ├── validation/
│       └── test/
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

*Note: Dataset images and generated results may be excluded from Git using `.gitignore`. The trained model must be available for prediction and deployment.*

---

## 📊 Dataset Information

The project uses 600 apple leaf images, equally distributed between two classes.

### Original Dataset

| Class | Images |
|---|---:|
| Healthy Leaf | 300 |
| Unhealthy Leaf | 300 |
| **Total** | **600** |

### Dataset Distribution

| Split | Percentage | Healthy | Unhealthy | Total |
|---|---:|---:|---:|---:|
| Training | 70% | 210 | 210 | 420 |
| Validation | 15% | 45 | 45 | 90 |
| Testing | 15% | 45 | 45 | 90 |
| **Total** | **100%** | **300** | **300** | **600** |

### Dataset Purpose

- **Training:** Teaches the CNN to recognize image features.
- **Validation:** Monitors performance during training and supports model selection.
- **Testing:** Measures performance on images not used during training or validation.

---

## ⚙️ Image Preprocessing

The preprocessing pipeline prepares images for the CNN model.

1. Load images from their respective class directories.
2. Convert images into RGB format.
3. Resize images to 128 × 128 pixels.
4. Convert images into numerical tensors.
5. Normalize pixel values.

### Normalization

The model uses a rescaling layer:

\[
x_{\text{normalized}} = \frac{x}{255}
\]

This converts pixel values from the range 0–255 into the range 0–1.

Normalization is included inside the model. Prediction images should not be normalized a second time.

---

## 🔄 Data Augmentation

Data augmentation introduces variations into training images to improve the model's ability to learn visual features.

| Technique | Purpose |
|---|---|
| Random Flip | Changes horizontal orientation |
| Random Rotation | Introduces rotational variations |
| Random Zoom | Changes image scale |
| Random Contrast | Adjusts image contrast |

These transformations are applied during training without permanently modifying the original images.

---

## 🧠 CNN Architecture

The model uses a custom sequential CNN architecture.

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
Healthy / Unhealthy
```

### Layer Explanation

| Layer | Function |
|---|---|
| Rescaling | Normalizes image pixels |
| Conv2D | Extracts visual features |
| MaxPooling2D | Reduces spatial dimensions |
| GlobalAveragePooling2D | Compresses feature maps |
| Dense | Learns classification patterns |
| Dropout | Helps reduce overfitting |
| Sigmoid | Produces the positive-class probability |

### Model Configuration

| Parameter | Value |
|---|---|
| Input Shape | `(128, 128, 3)` |
| Convolution Filters | 32, 64, 128 |
| Dense Units | 128 |
| Dropout | 0.4 |
| Output Units | 1 |
| Output Activation | Sigmoid |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Maximum Epochs | 30 |
| Total Parameters | 109,889 |

---

## 🚀 Model Training

The model is trained using TensorFlow and Keras.

The training process uses the training dataset to learn visual patterns and the validation dataset to monitor generalization.

### Training Callbacks

**1. EarlyStopping**

Stops training when validation loss stops improving and restores the best weights.

**2. ModelCheckpoint**

Saves the model with the best validation loss.

**3. ReduceLROnPlateau**

Reduces the learning rate when validation loss stops improving.

### Generated Artifacts

After training, the project generates:

```text
models/
├── best_leaf_classifier.keras
└── leaf_classifier.keras

results/
├── training_history.csv
└── plots/
    └── training_history.png
```

---

## 📈 Model Evaluation

The trained model is evaluated using a separate test dataset containing 90 images.

### Test Performance

| Metric | Result |
|---|---:|
| Test Samples | 90 |
| Correct Predictions | 77 |
| Incorrect Predictions | 13 |
| Test Accuracy | **85.56%** |
| Macro F1-score | **0.8547** |

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

### Results Interpretation

- 35 healthy images were correctly classified.
- 10 healthy images were incorrectly classified as unhealthy.
- 42 unhealthy images were correctly classified.
- 3 unhealthy images were incorrectly classified as healthy.

The model achieved 85.56% accuracy on the test set.

**Research note:** The test set contains only 90 images. These results are an initial experimental benchmark and do not establish reliable real-world performance.

---

## 🖥️ Command-Line Interface

The project includes a menu-driven CLI through `main.py`.

It allows users to execute individual machine learning operations from the terminal.

| Option | Operation |
|---|---|
| 1 | Check Original Dataset |
| 2 | Preprocess Sample Images |
| 3 | Split Dataset |
| 4 | Train CNN Model |
| 5 | Evaluate CNN Model |
| 6 | Predict Test Images |
| 0 | Exit |

The CLI uses Python's `subprocess` module to execute existing scripts and displays execution status, elapsed time, and exit codes.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model implementation |
| NumPy | Numerical operations |
| Pandas | Training and evaluation data |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation metrics |
| Pillow | Image processing |
| Streamlit | Interactive frontend |
| Git & GitHub | Version control |
| Streamlit Community Cloud | Web deployment |

---

## 💻 Installation and Local Setup

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

### 3. Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 5. Prepare the Dataset

Organize your original images:

```text
dataset/
└── original/
    ├── Healthy_leaf/
    └── Unhealthy_leaf/
```

The dataset is not necessarily included in the GitHub repository.

### 6. Run the CLI

```powershell
python main.py
```

### 7. Run the Web Dashboard

```powershell
streamlit run app/dashboard.py
```

Open the local URL provided by Streamlit, usually:

`http://localhost:8501`

---

## 🔬 Running Individual Scripts

You can also run each operation independently:

```powershell
# Check original dataset
python src/check_dataset.py

# Preprocess images
python src/preprocess.py

# Split dataset
python src/split_dataset.py

# Train CNN model
python src/train.py

# Evaluate model
python src/evaluate.py

# Predict images
python src/predict.py
```

Run the scripts in the appropriate order when starting with a fresh dataset.

---

## ☁️ Deployment

LeafVision is deployed using **Streamlit Community Cloud**.

**Live Application:**

[https://cnn-model-by-rahat.streamlit.app/](https://cnn-model-by-rahat.streamlit.app/)

The application is connected to the GitHub repository and uses the following configuration:

| Setting | Value |
|---|---|
| Hosting | Streamlit Community Cloud |
| Repository | `rahat-codes/CNN-model-Image-Binary-Classification` |
| Branch | `main` |
| Entry Point | `app/dashboard.py` |

The deployed dashboard uses the saved trained model for image prediction. Local training and CLI operations remain separate from the hosted application.

---

## 🔮 Future Improvements

- Expand the dataset with more diverse leaf images.
- Test the model on independent external datasets.
- Compare custom CNN with MobileNetV2 and EfficientNet.
- Perform hyperparameter tuning.
- Add Grad-CAM for model explainability.
- Improve generalization and probability calibration.
- Add prediction history and downloadable reports.
- Compare model inference speed and performance.
- Explore multi-class disease classification.

---

## 📚 Learning Outcomes

This project helped explore the complete workflow of a deep learning application:

- Image classification using CNNs.
- Convolution and feature extraction.
- TensorFlow and Keras implementation.
- Image preprocessing and augmentation.
- Training, validation, and testing.
- Model evaluation and error analysis.
- Modular Python project development.
- CLI development.
- Interactive machine learning visualization.
- Cloud deployment using Streamlit.

---

## 👨‍💻 Author

**Rahat Md Mahadi Hasan**

AI and Computer Engineering Student  
Tongmyong University, Busan, South Korea

- GitHub: [@rahat-codes](https://github.com/rahat-codes)
- LinkedIn: [Rahat Md Mahadi Hasan](https://linkedin.com/in/rahat-mahadi-hasan)

---

## 📄 License

This project is developed for educational and research purposes.

---

<p align="center">

  **🍃 LeafVision**

  *From Image Data to Intelligent Classification.*

  [Live Demo](https://cnn-model-by-rahat.streamlit.app/) · [GitHub Repository](https://github.com/rahat-codes/CNN-model-Image-Binary-Classification)

</p>
