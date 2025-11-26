# Final Project in Deep Learning 🧠🌍

This repository contains my final project for a deep learning course.  
The project focuses on **image classification for land-use / land-cover scenes**, and compares different deep learning approaches in **Keras (TensorFlow)** and **PyTorch**, including **CNNs** and **Vision Transformers (ViT)**.

The work is organized as a series of Jupyter notebooks that follow the structure of the course: from problem framing and data loading, through baseline models, up to advanced architectures and a final evaluation.

---

## 🎯 Project Goals

- Build and train image classification models for a land classification task.
- Compare:
  - Memory-based vs. generator-based data loading.
  - Keras-based vs. PyTorch-based classifiers.
  - CNN-based models vs. Vision Transformers.
- Evaluate performance using accuracy, loss curves, and confusion matrices.
- Save and reuse the best-performing model.

---

## 📁 Repository Structure

- **`AI-capstone-M1L1-v1.ipynb`**  
  Initial capstone setup: problem definition, dataset overview, and environment setup.

- **`AI-capstone-M1L2-v1.ipynb`**  
  Data exploration and preprocessing: loading the data, basic statistics, and preparing train/validation/test splits.

- **`AI-capstone-M1L3-v1.ipynb`**  
  Baseline modeling and first experiments: simple models and evaluation to establish a baseline.

- **`Compare_Memory-Based_Versus_Generator-Based_Data_Loading.ipynb`**  
  Comparison between:
  - **Memory-based loading** (loading all data into memory), and  
  - **Generator-based loading** (using data generators / dataloaders).  
  Focus on performance, memory usage, and training behavior.

- **`Lab_M2L1_Train_and_Evaluate_a_Keras-Based_Classifier.ipynb`**  
  Implementation of an image classifier using **Keras**:
  - Convolutional neural network (CNN) architecture.
  - Model training, validation, and testing.
  - Metrics: accuracy, loss curves, confusion matrix, and classification report.

- **`Lab_M2L2_Implement_and_Test_a_PyTorch-Based_Classifier.ipynb`**  
  Implementation of a similar classifier in **PyTorch**:
  - Custom Dataset and DataLoader.
  - Model definition, training loop, and evaluation.
  - Comparison of training flow vs. Keras.

- **`Lab_M2L3_Comparative_Analysis_of_Keras_and_PyTorch_Models.ipynb`**  
  Side-by-side comparison between the Keras and PyTorch models:
  - Training curves.
  - Final accuracy and other metrics.
  - Discussion of pros and cons of each framework.

- **`Lab_M3L1_Vision_Transformers_in_Keras.ipynb`**  
  Vision Transformer (ViT) model using **Keras**:
  - Using patch embeddings and transformer blocks (or a pre-built ViT).
  - Training and evaluation on the land classification task.

- **`Lab_M3L2_Vision_Transformers_in_PyTorch.ipynb`**  
  Vision Transformer implementation / usage in **PyTorch**:
  - PyTorch-based ViT model.
  - Training, evaluation, and comparison to CNNs.

- **`lab_M4L1_Land_Classification_CNN-ViT_Integration_Evaluation.ipynb`**  
  Final integrated evaluation:
  - Compare **CNN vs. ViT** models.
  - Analyze metrics, confusion matrices, and example predictions.
  - Summarize findings and best-performing approach.

- **`ai_capstone_keras_best_model.model.keras`**  
  Saved **best Keras model** from the experiments, ready for loading and inference.

---

## 🧩 Main Techniques & Concepts

- Convolutional Neural Networks (CNNs) for image classification.
- Vision Transformers (ViT) for vision tasks.
- Keras vs. PyTorch workflows (model building, training, evaluation).
- Data preprocessing and augmentation.
- Memory-based vs. generator-based data loading.
- Model evaluation: accuracy, loss curves, ROC/Confusion Matrix, etc.

---

## 🛠️ Requirements

You will need a Python environment with the following (typical) packages:

- Python 3.x  
- Jupyter Notebook / JupyterLab  
- TensorFlow / Keras  
- PyTorch & torchvision  
- NumPy  
- pandas  
- scikit-learn  
- matplotlib  

You can install the core dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow torch torchvision
