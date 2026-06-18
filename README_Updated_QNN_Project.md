# Quantum Image Classification Using QNN and CNN

## Project Overview

This project compares a Quantum Neural Network (QNN) with a Classical Convolutional Neural Network (CNN) for image classification.

The project evaluates both models on:

- MNIST (Digit 0 vs Digit 1)
- Fashion-MNIST (T-Shirt vs Trouser)

A Streamlit dashboard is used to present the project, explain the concepts, execute experiments, and display results.

---

# Features

## Machine Learning Features

- Quantum Neural Network using Qiskit
- Classical CNN using TensorFlow
- Feature Extraction Pipeline
- Binary Image Classification
- Accuracy, Precision, Recall and F1 Score Evaluation

## Datasets

- MNIST
- Fashion-MNIST

## Dashboard Features

- Login Page
- Project Overview Page
- QNN Explanation Page
- CNN Explanation Page
- Dataset Information Page
- Run Experiment Page
- Live Terminal Output
- Results Page

---

# Project Structure

```text
Quantum_QNN_Project/
│
├── app.py
├── README.md
├── requirements.txt
│
├── src/
│   ├── main.py
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── evaluation.py
│   ├── cnn_model.py
│   ├── qnn_model.py
│   └── qnn_modelbackup.py
│
├── results/
│   └── experiment_results.txt
│
└── report/
```

---

# Technology Stack

- Python
- Qiskit
- Qiskit Machine Learning
- TensorFlow
- Scikit-Learn
- NumPy
- Streamlit

---

# Installation Guide (Fresh Laptop)

## Step 1: Clone Repository

```bash
git clone https://github.com/HarshithManish/Quantum-Image-Classification-QNN-vs-CNN.git
```

## Step 2: Open Project

```bash
cd Quantum-Image-Classification-QNN-vs-CNN
```

## Step 3: Create Virtual Environment

Mac/Linux

```bash
python3 -m venv qnn_env
```

Windows

```bash
python -m venv qnn_env
```

---

## Step 4: Activate Virtual Environment

Mac/Linux

```bash
source qnn_env/bin/activate
```

Windows

```bash
qnn_env\Scripts\activate
```

---

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is unavailable:

```bash
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install qiskit
pip install qiskit-machine-learning
pip install qiskit-algorithms
pip install matplotlib
pip install streamlit
```

---

# Running the Project

## Run Model Directly

```bash
python src/main.py
```

The program will:

1. Load MNIST
2. Train QNN
3. Train CNN
4. Evaluate Results
5. Load Fashion-MNIST
6. Train QNN
7. Train CNN
8. Evaluate Results
9. Save Results

---

## Run Dashboard

```bash
streamlit run app.py
```

---

# Dashboard Login

Username:

```text
admin
```

Password:

```text
admin
```

---

# Dashboard Flow

## 1. Project Overview

Displays:

- Project Objective
- Technology Stack
- Project Workflow
- Models Compared

---

## 2. QNN Page

Explains:

- Quantum Neural Networks
- Quantum Circuits
- Feature Encoding
- Qubits
- ZZFeatureMap
- RealAmplitudes Ansatz
- COBYLA Optimizer

Workflow:

```text
Image
 ↓
Feature Extraction
 ↓
Quantum Encoding
 ↓
Quantum Circuit
 ↓
Measurement
 ↓
Prediction
```

---

## 3. CNN Page

Explains:

- Convolutional Neural Networks
- Deep Learning for Images
- CNN Architecture

Workflow:

```text
Input Image
 ↓
Conv2D
 ↓
MaxPooling
 ↓
Flatten
 ↓
Dense
 ↓
Output
```

---

## 4. Dataset Page

### MNIST

- Handwritten Digits
- Binary Classification
- Digit 0 vs Digit 1

### Fashion-MNIST

- Clothing Dataset
- Binary Classification
- T-Shirt vs Trouser

---

## 5. Run Experiment Page

User clicks:

```text
Run Model
```

The dashboard:

- Executes src/main.py
- Shows Live Terminal Output
- Trains QNN
- Trains CNN
- Generates Results

---

## 6. Results Page

Displays:

- Latest Experiment Results
- MNIST Accuracy
- Fashion-MNIST Accuracy
- QNN vs CNN Comparison

---

# Models Used

## Quantum Neural Network

Configuration:

- 4 Qubits
- ZZFeatureMap
- RealAmplitudes Ansatz
- COBYLA Optimizer

Purpose:

To investigate the feasibility of quantum machine learning for image classification.

---

## Convolutional Neural Network

Architecture:

```text
Conv2D
 ↓
MaxPooling
 ↓
Flatten
 ↓
Dense
 ↓
Output Layer
```

Purpose:

Acts as the benchmark classical model.

---

# Current Results

## MNIST

| Model | Accuracy |
|---------|---------|
| QNN | 62% |
| CNN | 100% |

## Fashion-MNIST

| Model | Accuracy |
|---------|---------|
| QNN | 50% |
| CNN | 100% |

---

# Output File

After execution:

```text
results/experiment_results.txt
```

contains the latest experiment results.

---

# Future Improvements

- Improved QNN Architecture
- Additional Quantum Layers
- Better Feature Encoding
- Multi-Class Classification
- Quantum Circuit Visualization
- Accuracy Charts
- Confusion Matrices
- Enhanced Dashboard UI

---

# Author

Harshith Manish

Computer Science Engineering

Quantum Computing and Machine Learning Internship Project
