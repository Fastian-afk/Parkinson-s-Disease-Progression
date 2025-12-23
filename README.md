# 🧠 Parkinson’s Disease Detection via Voice Analysis
**Explainable Machine Learning for Early Neurodegenerative Diagnosis**

<p align="center">
  <img src="https://img.shields.io/badge/Healthcare-AI-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Disease%20Detection-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-orange?style=for-the-badge"/>
</p>

---

## 📌 Overview
This project applies **machine learning and biomedical signal analysis** to enable **early detection of Parkinson’s Disease (PD)** using **voice measurements**.

Voice degradation is a well-known early symptom of PD. By leveraging structured vocal biomarkers and explainable deep learning, this system predicts disease presence while providing **transparent, interpretable insights** into model decisions.

---

## 📚 Background
Parkinson’s Disease is a progressive neurodegenerative disorder that impacts motor control and speech production.  
Subtle changes in vocal frequency, amplitude, and jitter often appear **before severe clinical symptoms**, making voice analysis a valuable diagnostic signal.

This project uses the **UCI Parkinson’s Disease Detection Dataset**, containing biomedical voice measurements from both healthy individuals and PD patients.

---

## 🧠 Models Implemented
Multiple classical and deep learning models were evaluated:

- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- **Deep Neural Network (DNN)** ✅ *(best performing)*  

The final DNN model was enhanced with:
- 🔍 **Grad-CAM–based explainability**
- 💾 Model serialization for reuse on new patient samples

---

## 🔬 Key Features
- 🧹 Data cleaning and feature selection  
- 📊 Comparative evaluation across ML models  
- 🧠 Deep learning for complex pattern recognition  
- 🔍 Explainable AI using Grad-CAM visualizations  
- ♻ Reusable pipeline for real-world inference  

---

## 🛠 Tech Stack
<p align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="38"/>
</p>

**Visualization & Deployment**
- Matplotlib, Seaborn  
- Grad-CAM (custom implementation)  
- Streamlit (optional web interface)

---

## 📂 Project Structure

parkinsons-voice-detection/
│
├── data/                 # Voice measurement dataset
├── preprocess.py         # Data cleaning & feature selection
├── train_models.py       # Classical ML model training
├── dnn_model.py          # Deep Neural Network implementation
├── explain.py            # Grad-CAM visualizations
├── predict.py            # Inference on new patient samples
├── requirements.txt
└── README.md

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

git clone https://github.com/Fastian-afk/parkinsons-voice-detection.git
cd parkinsons-voice-detection

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Train & Evaluate Models

python train_models.py

### 4️⃣ Run Explainability

python explain.py

### 5️⃣ Predict on New Data

python predict.py

---

## 📊 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

This ensures robust assessment for **clinical decision-support relevance**.

---

## 🎯 Why This Project Matters

* Targets **early-stage Parkinson’s detection**
* Demonstrates **explainable AI in healthcare**
* Combines biomedical signals with deep learning
* Aligns with clinical trust and ethical AI principles

---

## 👨‍💻 Author

**Imaad Fazal**

📧 Email: [imdufazal@gmail.com](mailto:imdufazal@gmail.com)
🌐 Portfolio: [https://imaad-fazal-portfolio-hub.vercel.app/](https://imaad-fazal-portfolio-hub.vercel.app/)

---

## 📜 License

This project is released under the **MIT License**.
