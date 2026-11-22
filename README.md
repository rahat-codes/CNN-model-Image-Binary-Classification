# 🍃 Apple Leaf Binary Image Classification Using CNN

<p align="center">
  <strong>A Deep Learning Project Using TensorFlow and Keras</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras" alt="Keras">
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-green" alt="Binary Classification">
</p>

## 1. Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify apple leaf images into two categories:

- **Healthy Leaf**
- **Unhealthy Leaf**

The primary objective is to understand and implement a complete deep learning workflow, starting from image preprocessing and dataset preparation to CNN training, model evaluation, and image prediction.

The model learns visual patterns from apple leaf images and uses those patterns to predict whether an input leaf belongs to the healthy or unhealthy category.

## 2. Project Objectives

- Understand the fundamentals of Convolutional Neural Networks.
- Perform image preprocessing and dataset preparation.
- Build a CNN architecture using TensorFlow and Keras.
- Train a binary image classification model.
- Evaluate model performance using standard classification metrics.
- Predict new images using the trained model.
- Analyze classification results and prediction errors.

## 3. Dataset Description

The dataset contains two image categories:

| Class | Description |
|---|---|
| Healthy Leaf | Images of healthy apple leaves |
| Unhealthy Leaf | Images of unhealthy apple leaves |

**Dataset distribution:**

| Dataset | Healthy | Unhealthy | Total |
|---|---:|---:|---:|
| Training | 210 | 210 | 420 |
| Validation | 45 | 45 | 90 |
| Testing | 45 | 45 | 90 |
| **Total** | **300** | **300** | **600** |

The dataset was divided into training, validation, and testing subsets.

- Training data: Used to learn model parameters.
- Validation data: Used to monitor performance during training.
- Testing data: Used for final model evaluation.

**Note:** Dataset images are not included in this GitHub repository. Users must provide the dataset locally using the expected directory structure.

## 4. CNN Model Architecture

The project uses a custom CNN architecture developed with TensorFlow/Keras.

### Architecture Overview

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 128 × 128 × 3 | Accept RGB images |
| Rescaling | 1/255 | Normalize pixel values |
| Random Flip | Horizontal | Data augmentation |
| Random Rotation | 0.08 | Data augmentation |
| Random Zoom | 0.1 | Data augmentation |
| Random Contrast | 0.1 | Data augmentation |
| Conv2D | 32 filters | Extract low-level features |
| MaxPooling2D | 2 × 2 | Reduce spatial dimensions |
| Conv2D | 64 filters | Extract intermediate features |
| MaxPooling2D | 2 × 2 | Reduce spatial dimensions |
| Conv2D | 128 filters | Extract higher-level features |
| MaxPooling2D | 2 × 2 | Reduce spatial dimensions |
| Global Average Pooling | - | Reduce feature maps |
| Dense | 128 neurons, ReLU | Learn classification patterns |
| Dropout | 0.4 | Reduce overfitting |
| Output | 1 neuron, Sigmoid | Binary classification |

**Total trainable and non-trainable parameters:** 109,889.

### Binary Classification

The output layer uses the sigmoid activation function.

The model produces a value between 0 and 1, representing its predicted probability for the Unhealthy Leaf class.

Classification threshold:

- Probability < 0.5 → Healthy Leaf
- Probability ≥ 0.5 → Unhealthy Leaf

## 5. Model Training

The model was trained using TensorFlow and Keras.

**Training configuration:**

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Input Image Size | 128 × 128 |
| Batch Size | 32 |
| Maximum Epochs | 30 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Output Activation | Sigmoid |
| Early Stopping | Enabled |
| Model Checkpoint | Enabled |

Early stopping and model checkpointing were used to monitor validation performance and preserve the best-performing model.

## 6. Experimental Results

The trained CNN was evaluated on 90 test images.

<details>
<summary><strong>Test Performance</strong></summary>

| Metric | Result |
|---|---:|
| Total Test Images | 90 |
| Correct Predictions | 77 |
| Incorrect Predictions | 13 |
| **Test Accuracy** | **85.56%** |

</details>

### Confusion Matrix

| Actual / Predicted | Healthy | Unhealthy |
|---|---:|---:|
| Healthy | 35 | 10 |
| Unhealthy | 3 | 42 |

### Classification Metrics

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Healthy | 92.11% | 77.78% | 84.34% |
| Unhealthy | 80.77% | 93.33% | 86.60% |

**Observations:**

- The model correctly classified 77 out of 90 test images.
- It performed better at identifying unhealthy leaves in terms of recall.
- Some healthy leaves were incorrectly classified as unhealthy.
- The results indicate that the model can learn useful visual patterns from the training dataset.

These results represent an initial experiment on a relatively small dataset. Further testing with larger and more diverse images is necessary to evaluate real-world generalization.

## 7. Project Structure

```text
CNN-model-Image-Binary-Classification/
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
│   └── predictions/
│       ├── Healthy_leaf/
│       └── Unhealthy_leaf/
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
├── experiment.ipynb
├── collect.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 8. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/rahat-codes/CNN-model-Image-Binary-Classification.git

cd CNN-model-Image-Binary-Classification
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Place the original images into the following directories:

```text
dataset/
└── original/
    ├── Healthy_leaf/
    └── Unhealthy_leaf/
```

Run the dataset preparation scripts according to the project workflow.

### Step 5: Train the Model

```bash
python src/train.py
```

### Step 6: Evaluate the Model

```bash
python src/evaluate.py
```

### Step 7: Predict and Classify Images

```bash
python src/predict.py
```

The prediction script classifies test images and copies them into the corresponding prediction folders:

```text
results/
└── predictions/
    ├── Healthy_leaf/
    └── Unhealthy_leaf/
```

## 9. Experiment Notebook

The repository includes `experiment.ipynb`, which documents the model experimentation process.

The notebook covers:

1. Importing libraries and loading the trained model.
2. Inspecting the CNN architecture.
3. Visualizing sample apple leaf images.
4. Performing individual image predictions.
5. Testing multiple images.
6. Evaluating the complete test dataset.
7. Generating a confusion matrix.
8. Analyzing experimental results.

## 10. Future Improvements

Potential improvements for future experiments include:

- Increasing dataset size and diversity.
- Experimenting with different CNN architectures.
- Applying transfer learning with MobileNetV2 or EfficientNet.
- Performing systematic hyperparameter tuning.
- Investigating overfitting and generalization.
- Testing the model on images collected under different lighting and environmental conditions.

## 11. Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- Pillow
- Jupyter Notebook

## 12. Author

**Rahat Md Mahadi Hasan**

AI and Computer Engineering Student  
Tongmyong University, Busan, South Korea

- GitHub: [rahat-codes](https://github.com/rahat-codes)
- LinkedIn: [Rahat Md Mahadi Hasan](https://linkedin.com/in/rahat-mahadi-hasan)

---

<p align="center">
  <strong>Built as a hands-on learning project in Deep Learning and Computer Vision.</strong>
</p>