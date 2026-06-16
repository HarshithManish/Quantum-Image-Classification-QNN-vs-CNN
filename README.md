# Quantum Neural Network (QNN) for Fashion-MNIST Classification

## Overview

This project demonstrates image classification using a **Quantum Neural Network (QNN)** and compares its performance with a **Classical Convolutional Neural Network (CNN)**.

The objective is to explore the feasibility of quantum machine learning for image classification tasks and evaluate its performance against a well-established classical deep learning model.

The Fashion-MNIST dataset is used as the benchmark dataset.

---

## Project Objectives

- Load and preprocess Fashion-MNIST images.
- Reduce image dimensionality using feature extraction techniques.
- Encode classical features into quantum circuits.
- Train a Quantum Neural Network (QNN).
- Train a Classical Convolutional Neural Network (CNN).
- Compare the performance of both models using standard evaluation metrics.

---

## Dataset

**Fashion-MNIST**

The dataset contains grayscale images of fashion products belonging to different categories such as:

- T-shirt/Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

Image Size:
- 28 × 28 pixels
- Grayscale

---

## Technologies Used

### Quantum Computing
- Qiskit
- Qiskit Machine Learning

### Machine Learning
- TensorFlow / Keras
- Scikit-Learn
- NumPy

### Visualization
- Matplotlib

---

## Project Workflow

1. Load Fashion-MNIST images.
2. Perform feature extraction and dimensionality reduction.
3. Encode features into quantum circuits.
4. Train a Quantum Neural Network (QNN).
5. Train a Classical Convolutional Neural Network (CNN).
6. Evaluate both models using Accuracy, Precision, Recall, and F1 Score.
7. Compare the results.

---

## Results

### Quantum Neural Network (QNN)

| Metric | Value |
|----------|----------|
| Accuracy | 62.00% |
| Precision | 38.44% |
| Recall | 62.00% |
| F1 Score | 47.46% |

### Convolutional Neural Network (CNN)

| Metric | Value |
|----------|----------|
| Accuracy | 100.00% |
| Precision | 100.00% |
| Recall | 100.00% |
| F1 Score | 100.00% |

---

## Final Comparison

| Model | Accuracy |
|---------|---------|
| QNN | 62.00% |
| CNN | 100.00% |

The CNN achieved higher performance on the Fashion-MNIST dataset. However, the QNN successfully demonstrated that image data can be encoded into quantum circuits and classified using quantum machine learning techniques.

---

## How to Run

```bash
source qnn_env/bin/activate
python src/main.py
```

---

## Conclusion

This project successfully demonstrates the implementation of a Quantum Neural Network for image classification and compares its performance with a Classical CNN. While the CNN significantly outperformed the QNN on the Fashion-MNIST dataset, the project highlights the potential and current limitations of quantum machine learning.
