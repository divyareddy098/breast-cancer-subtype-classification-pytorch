# 🧠 Deep Learning for Breast Cancer Subtype Classification (PyTorch)

## Project Goal
Developing a deep learning model to classify breast cancer subtypes from TCGA RNA-seq data and explore key predictive genes associated with different molecular subtypes.

---

## Overview
In this project, I built a neural network–based pipeline using TCGA RNA-seq data to classify breast cancer subtypes. The goal is to combine deep learning with genomic data analysis to understand subtype-specific transcriptional patterns.

This project focuses on clinically relevant breast cancer subtypes (e.g., Luminal A, Luminal B, HER2-enriched, Basal-like).

---

## Workflow Overview

1. Data preprocessing and normalization  
2. Feature selection and dataset preparation  
3. Deep learning model development (PyTorch)  
4. Model training and evaluation  
5. Model interpretability using SHAP  

---

## Key Features

- Classification of breast cancer subtypes using RNA-seq gene expression data  
- Deep learning model implemented in PyTorch  
- Handling of high-dimensional genomic datasets  
- Model evaluation using ROC-AUC and F1-score  
- Feature importance analysis using SHAP for biological interpretation  
- Modular and reproducible pipeline  

---

## Project Structure

breast-cancer-subtype-classification-pytorch/  
│── scripts/  
│── notebooks/  
│── data/  
│── results/  
│── figures/  
│── README.md  

---

## ⚙️ Workflow Details

### 1️⃣ Data Processing
- Load TCGA RNA-seq dataset  
- Normalize gene expression values  
- Prepare training and test datasets  

### 2️⃣ Model Development
- Build neural network architecture using PyTorch  
- Train model on gene expression features  

### 3️⃣ Model Evaluation
- Evaluate model performance using ROC-AUC and F1-score  
- Assess classification performance across subtypes  

### 4️⃣ Model Interpretability
- Apply SHAP to identify important genes  
- Visualize feature importance and gene contributions  

---

## Status

This project is currently in progress. Data preprocessing and model development pipelines are implemented, and model training and evaluation are ongoing.

---

## Expected Outputs

- ROC curve  
- Confusion matrix  
- SHAP summary plots  
- Identification of subtype-associated genes  

---

## Tools & Technologies

- Python  
- PyTorch  
- pandas, NumPy  
- scikit-learn  
- SHAP  
- matplotlib / plotly  

---

## Skills Demonstrated

- Deep learning for biological data  
- High-dimensional RNA-seq data analysis  
- Model evaluation and validation  
- Explainable AI (SHAP)  
- Data preprocessing and feature engineering  

---

## Impact

This project demonstrates the application of deep learning to genomic data for cancer subtype classification and provides a foundation for identifying potential biomarkers.

---

## Author

Divya Reddy  
MS Bioinformatics, Georgia Institute of Technology  
