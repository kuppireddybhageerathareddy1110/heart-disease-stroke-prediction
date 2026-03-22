# 🫀 Heart Disease Prediction using ML + DL + Explainable AI

## 📌 Overview

This project builds a **complete end-to-end machine learning system** for heart disease prediction using:

* ✅ 3 merged datasets
* ✅ 10 Machine Learning models
* ✅ 10 Deep Learning models
* ✅ Automatic best model selection
* ✅ Explainable AI (SHAP)
* ✅ Deployment-ready pipeline

---

## 📂 Project Structure

```
project/
│── data/
│   └── final_heart_clean.csv
│
│── models/
│   ├── best_model.pkl
│   ├── best_dl_model.h5
│   └── columns.pkl
│
│── outputs/
│   ├── shap_summary.png
│   ├── all_results.csv
│
│── main.py
│── README.md
```

---

## 📊 Datasets Used

This project combines **3 heart disease datasets**:

1. UCI Heart Disease Dataset
2. Cleveland Heart Disease Dataset
3. Kaggle Heart Dataset

All datasets are merged, cleaned, and standardized.

---

## ⚙️ Features

* Data preprocessing & cleaning
* Missing value handling
* Feature engineering
* One-hot encoding
* Model comparison (ML + DL)
* Best model selection
* Explainable AI (SHAP)
* Model saving for deployment

---

## 🤖 Machine Learning Models (10)

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost
* Extra Trees
* SVM
* KNN
* Naive Bayes
* LDA

---

## 🧠 Deep Learning Models (10)

* Dense Neural Networks (various architectures)
* Different combinations of:

  * Layers
  * Units
  * Dropout
  * Activations

---

## 🏆 Model Selection

* All 20 models are trained and evaluated
* Best model is selected based on **accuracy**

---

## 📈 Explainable AI (XAI)

* SHAP (SHapley Additive exPlanations)
* Generates:

  * Feature importance plots
  * Model interpretability

---

## 💾 Outputs

After running the script:

```
data/
 └── final_heart_clean.csv

models/
 ├── best_model.pkl OR best_dl_model.h5
 ├── columns.pkl

outputs/
 ├── shap_summary.png
 ├── all_results.csv
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib shap tensorflow joblib
```

---

### 2️⃣ Run the Script

```bash
python main.py
```

---

## 🔮 Prediction Function

```python
predict_input({
    "age": 52,
    "sex": "M",
    ...
})
```

---

## 🌐 Deployment Ready

This project is ready for:

* FastAPI / Flask backend
* React frontend
* Cloud deployment (AWS / Render / Vercel)

---

## ⚠️ Notes

* ML models often outperform DL for small datasets
* Deep learning requires larger datasets for best results

---

## 📌 Future Improvements

* Hyperparameter tuning (GridSearch / Optuna)
* XGBoost / LightGBM integration
* Real-time SHAP dashboard
* Docker deployment

---

## 👨‍💻 Author

Developed as a **full production ML + DL + XAI pipeline project**.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
