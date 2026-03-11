# Kidney-Disease-Classification-Deep-Learning-Project
This repository contains my work on an amazing deep learning project 

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/MunimAkbar/Kidney-Disease-Classification-Deep-Learning-Project
```
### STEP 01- Create a conda environment for GPU Training
```bash
# Create a new environment with Python 3.10
conda create -p env python=3.10 -y

# Activate the environment
conda activate ./env

# Install CUDA Toolkit and cuDNN compatible with TF 2.10
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
```

### STEP 02- install the requirements
```bash
# Install GPU-enabled TensorFlow
pip install tensorflow-gpu==2.10.0

# Install remaining dependencies
pip install -r requirements.txt
```

### STEP 03- Running Training
```bash
# Run the training pipeline
python main.py
```

### STEP 04- Running the Application
```bash
# Finally run the flask app
python app.py
```

Now,
```bash
open up you local host and port
```

## MLflow


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)


Run this to export as env variables:

```bash


```

