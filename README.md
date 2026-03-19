---
title: Kidney Disease Classifier
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# 🏥 Kidney Disease Classification using Deep Learning

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/munimakbar/Kidney-Disease-Classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [!TIP]
> ### [🔗 View Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/munimakbar/Kidney-Disease-Classifier)

## 📺 App Preview
| Landing Page | Prediction Result |
| :---: | :---: |
| ![UI Preview 1](UI%20Screenshots/UI%20Prediction%202.png) | ![UI Preview 2](UI%20Screenshots/UI%20Prediction%201.png) |

## 🔬 Project Overview
This project is an end-to-end Deep Learning application designed to classify kidney CT scans as either **Tumor** or **Normal**. Built with TensorFlow and Flask, the application features a robust MLOps pipeline (DVC, MLflow), a premium web frontend, and advanced Out-of-Distribution (OOD) detection.

### 🧠 Model Architecture & Methodology
- **Base Model**: **VGG-16** (pre-trained on ImageNet)
- **Transfer Learning**: We froze the base convolutional layers and added a custom dense head (`Flatten -> Dense(2, softmax)`) to adapt the model for binary medical image classification.
- **Training Optimization**: 
  - **Callbacks**: `ModelCheckpoint` to save the best model and `ReduceLROnPlateau` to finely tune the learning rate when validation loss stalls.
  - **GPU Acceleration**: Configured natively for Windows using TensorFlow 2.10 and CUDA 11.2, achieving significantly faster training times over 50 epochs.

### 🛡️ Advanced Feature: Out-of-Distribution (OOD) Detection
A common issue with binary classifiers is that they forcefully categorize *any* uploaded image into one of the trained classes. 
To solve this, we implemented **Feature-Space Distance OOD Detection**:
1. During a one-time setup, the model extracts features from its penultimate layer (Flatten) across all training CT scans.
2. It calculates a **mean feature vector** and an **OOD threshold** ($\mu + 3\sigma$).
3. At inference, if a user uploads a non-medical image (e.g., a bicycle), the system calculates the cosine distance of the image's features against the mean vector.
4. If the distance exceeds the threshold, the UI flags the image as **"Invalid Input"** instead of making a false prediction.

### 💻 Premium User Interface
The project includes a custom-built, glassmorphic UI replacing the default template:
- **Responsive Design**: Clean layout built with raw CSS/JS (no heavy frontend frameworks).
- **Interactive Uploads**: Supports drag-and-drop file ingestion and client-side image previews.
- **Dynamic Results**: Displays color-coded badges (🔴 Tumor, 🟢 Normal, ⚠️ Invalid) and an animated confidence percentage bar.

---

## 🚀 Setup & Installation (GPU Optimized)

To run this project locally with GPU support on Windows, follow these exact steps:

### 1. Clone the repository
```bash
git clone https://github.com/MunimAkbar/Kidney-Disease-Classification-Deep-Learning-Project.git
cd Kidney-Disease-Classification-Deep-Learning-Project
```

### 2. Create the Conda Environment
We use Python 3.10 to maintain compatibility with TensorFlow 2.10 (the last version to natively support GPU on Windows).
```bash
conda create -p env python=3.10 -y
conda activate ./env
```

### 3. Install CUDA & cuDNN
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
```

### 4. Install Dependencies
```bash
pip install tensorflow-gpu==2.10.0
pip install -r requirements.txt
```

*(Note: Memory growth is automatically configured in `main.py` and `app.py` to prevent TF from hogging your entire VRAM).*

---

## 🏃‍♂️ How to Run

### Option A: Run the entire tracking & training pipeline
This will orchestrate data ingestion, model building, and training through DVC.
```bash
python main.py
```
*Alternatively, you can run `dvc repro` to execute only the changed stages.*

### Option B: Generate OOD Reference (Required before running the app)
If you have retrained the model, you **must** generate the feature references for the OOD detection to work.
```bash
python compute_features.py
```
*(This creates `model/feature_mean.npy` and `model/ood_threshold.npy`)*

### Option C: Run the Web Application
Start the Flask server to access the Prediction UI.
```bash
python app.py
```
Open your browser and navigate to: **http://127.0.0.1:8080**

![UI Prediction Screenshot](UI%20Screenshots/UI%20Prediction%201.png) *(Prediction UI Result)*
---

## 📊 MLOps Integration

### MLflow & DagsHub
We use **MLflow** tracked via **DagsHub** for experiment logging (metrics, parameters, and models).
To view the UI:
```bash
mlflow ui
```

### DVC (Data Version Control)
The pipeline is tracked using DVC. The stages are defined in `dvc.yaml`:
1. `data_ingestion`
2. `prepare_base_model`
3. `training`
4. `evaluation`

To visualize the DAG:
```bash
dvc dag
```

---

## 🌐 Deployment (Cloud & Local)

### 🏥 Option 1: Cloud Deployment (Hugging Face Spaces)
The application is pre-configured for free inference on Hugging Face using Docker.

1. **Create a Space**: On [Hugging Face](https://huggingface.co/spaces), create a new Space.
2. **SDK Choice**: Select **Docker** as the Space SDK and choose the **Blank** template.
3. **Push to Sync**: 
   - Connect your GitHub repository to the Space for automatic deployment.
   - Or push directly to the Hugging Face remote: `git push hf main`.

> [!NOTE]
> The provided `Dockerfile` is optimized to swap GPU-heavy dependencies for CPU-only versions during cloud deployment to ensure compatibility with the Hugging Face free tier.

### 💻 Option 2: Local Deployment (Docker)
If you have Docker installed locally, you can run the exact same environment as the cloud:
```bash
docker build -t kidney-classifier .
docker run -p 8080:8080 kidney-classifier
```
Access via: `http://localhost:8080`

### 🔧 Option 3: Standard Local Run (Direct)
Refer to the **"Setup & Installation"** section above for GPU-optimized local development.
