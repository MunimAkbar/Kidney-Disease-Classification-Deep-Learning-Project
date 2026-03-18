# Kidney Disease Classification using Deep Learning

![DeepVision AI Mockup](.assets/ui_mockup.png) *(Preview of the custom Prediction UI)*

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

<<<<<<< HEAD
![UI Prediction Screenshot](UI%20Screenshots/UI%20Prediction%201.png) *(Prediction UI Result)*
=======
>>>>>>> df1bec79fbc65aeebf1bbae8d6343d0b8c7a3309
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

## ☁️ Deployment (Hugging Face Spaces)

This project is configured out-of-the-box for free CPU inference deployment on **Hugging Face Spaces** via **Docker**.

We enforce a strict **hybrid workflow**:
1. **Train** the model locally on your GPU (using `main.py`).
2. **Deploy** only the Web UI and the lightweight inference model (`app.py` & `model.h5`) to the cloud.

### Deployment Steps:
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Docker** as the Space SDK and select the **Blank** template.
3. Upload this entire repository directly to the Hugging Face Space.

*(Note: The provided `Dockerfile` automatically swaps the required GPU TensorFlow for the CPU version to run cleanly on HF's free tier, and uses a pre-configured `.dockerignore` to keep your upload size small by ignoring the heavy training data).*
