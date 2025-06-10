# 🧠 Parkinson's Disease Detection via Voice Analysis

This project leverages machine learning and biomedical signal processing to enable **early detection of Parkinson’s Disease (PD)** using **voice measurements**. Built using Python, TensorFlow, and explainable AI (Grad-CAM), it provides a robust pipeline for classification, visualization, and real-world inference.

---

## 📚 Background

Parkinson’s Disease is a progressive neurodegenerative disorder that affects movement, often causing tremors and vocal impairments. Voice degradation occurs in early stages, making it a critical biomarker for diagnosis. This project uses the **UCI Parkinson's Disease Detection dataset**, featuring biomedical voice measurements from healthy and affected individuals, to train ML models capable of predicting PD with high accuracy.

---

## 🧰 Model Description

We implemented several classification models:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine**
- **K-Nearest Neighbors**
- **Deep Neural Networks**

The best-performing model (DNN) was further enhanced with:
- **Grad-CAM visualization** to explain prediction rationale
- **Model saving and loading** for reuse on new patient samples

---

## ⚙️ Technical Implementation

**Tools Used:**
- Python 3.10+
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn
- Grad-CAM (custom visualization)
- Streamlit (optional for web app interface)

**Key Features:**
- Data cleaning and feature selection
- Model comparison and evaluation (accuracy, precision, recall, F1)
- Explainability with Grad-CAM visualizations
- Predict from new patient voice data

---

## 💻 Installation & Requirements

Clone the repository:

```bash
git clone https://github.com/yourusername/parkinsons-voice-detection.git
cd parkinsons-voice-detection
