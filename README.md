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

# 🫀 Heart Disease Prediction System - Production Grade AI

> **Advanced machine learning system with explainable AI for cardiovascular risk assessment**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-red.svg)](https://github.com/slundberg/shap)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Frontend Guide](#-frontend-guide)
- [Model Performance](#-model-performance)
- [Explainability](#-explainability)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🎯 Overview

A **production-ready AI system** for predicting heart disease risk using advanced machine learning and explainable AI techniques. Built with FastAPI backend and modern vanilla JavaScript frontend.

### **What Makes This System Special?**

✅ **87.6% F1-Score** - Excellent prediction accuracy  
✅ **6 Types of AI Explanations** - SHAP + LIME for full transparency  
✅ **Production-Ready Code** - Validation, logging, error handling  
✅ **Modern Web Interface** - Responsive, interactive, beautiful  
✅ **Batch Processing** - Handle multiple patients efficiently  
✅ **Real-Time Predictions** - Sub-second response times  
✅ **Comprehensive Documentation** - Complete guides and examples  

---

## ✨ Features

### 🤖 **Machine Learning**
- **Random Forest Classifier** with 87.6% F1-score
- **5-Fold Cross-Validation** for robust performance
- **Feature Engineering** from 35 clinical parameters
- **Stratified Sampling** for balanced predictions
- **Confidence Intervals** (95% CI) for predictions

### 🔍 **Explainable AI**
- **SHAP Waterfall** - Individual feature contributions
- **SHAP Force Plot** - Interactive visualization
- **SHAP Summary** - Global feature importance
- **SHAP Dependence** - Feature relationship analysis
- **LIME** - Local interpretable model explanations
- **Risk Factor Identification** - Automatic highlighting

### 🌐 **API (FastAPI)**
- **17+ REST Endpoints** - Comprehensive functionality
- **Pydantic Validation** - Type-safe requests
- **Interactive Docs** - Swagger UI + ReDoc
- **CORS Support** - Cross-origin enabled
- **Health Checks** - Monitor system status
- **Batch Processing** - Multiple predictions
- **Comprehensive Logging** - Production monitoring

### 💻 **Frontend (Web UI)**
- **4 Main Sections** - Predict, Explain, Insights, About
- **Responsive Design** - Mobile, tablet, desktop
- **Real-Time Updates** - Instant predictions
- **Interactive Charts** - Plotly visualizations
- **Risk Stratification** - Color-coded levels
- **Sample Data Loader** - One-click demo
- **CSV Downloads** - Export reports

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                 USER INTERFACE                  │
│  (Responsive Web App - HTML/CSS/JavaScript)    │
└───────────────────┬─────────────────────────────┘
                    │ HTTP/JSON
                    ▼
┌─────────────────────────────────────────────────┐
│              FASTAPI BACKEND                    │
│  ┌──────────────┬──────────────┬─────────────┐ │
│  │  Prediction  │ Explainability│   Batch    │ │
│  │  Endpoints   │   (SHAP/LIME) │ Processing │ │
│  └──────────────┴──────────────┴─────────────┘ │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │  Random │ │  SHAP   │ │  LIME   │
  │  Forest │ │Explainer│ │Explainer│
  └─────────┘ └─────────┘ └─────────┘
        │           │           │
        └───────────┴───────────┘
                    ▼
        ┌───────────────────────┐
        │   Trained Model       │
        │   (87.6% F1-Score)   │
        │   2,141 Patients      │
        └───────────────────────┘
```

---

## 📦 Installation

### **Prerequisites**

- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

### **Step 1: Clone/Download Files**

Ensure you have all project files in one directory.

### **Step 2: Create Virtual Environment**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.7.2
joblib==1.3.2
shap==0.43.0
lime==0.2.0.1
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
python-multipart==0.0.6
```

---

## 🚀 Quick Start

### **3-Step Launch:**

#### **Step 1: Start API**
```bash
python api_advanced.py
```

**Expected output:**
```
✅ Model and artifacts loaded successfully
Model type: RandomForestClassifier
Features: 35
🚀 Heart Disease Prediction API started
Uvicorn running on http://0.0.0.0:8000
```

#### **Step 2: Open Frontend**
Simply double-click `index_advanced.html` in your file explorer

**OR** use a local server:
```bash
python -m http.server 8080
# Then open: http://localhost:8080/index_advanced.html
```

#### **Step 3: Test It!**
1. **Click** "Load Sample Patient" (high-risk patient pre-filled)
2. **Click** "Analyze Risk"
3. **View Results:** 🔴 HIGH RISK (85-95% probability)
4. **Click** "View Explanations"
5. **Explore** all 5 tabs (Waterfall, Force, LIME, Summary, Importance)

---

## 📚 API Documentation

### **Interactive Docs**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### **Example: Single Prediction**

```python
import requests

patient = {
    "age": 65,
    "sex": "Male",
    "chest_pain": "Asymptomatic",
    "restingbp_final": 160,
    "chol_final": 300,
    "maxhr_final": 130,
    "fasting_bs": "Yes",
    "resting_ecg": "ST-T Abnormality",
    "exercise_angina": "Yes",
    "oldpeak": 2.5,
    "st_slope": "Flat"
}

response = requests.post("http://localhost:8000/predict", json=patient)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Risk Factors: {result['risk_factors']}")
```

**Output:**
```
Prediction: Disease
Probability: 88.00%
Risk Level: HIGH
Risk Factors: ['oldpeak (2.5)', 'age (65.0)', 'exercise_angina (1.0)']
```

### **All Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/explain/waterfall` | SHAP waterfall plot |
| POST | `/explain/force` | SHAP force plot |
| POST | `/explain/lime` | LIME explanation |
| POST | `/explain/lime/html` | LIME HTML (interactive) |
| GET | `/explain/summary` | Global SHAP summary |
| GET | `/explain/importance` | Feature importance |
| POST | `/explain/dependence/{feature}` | SHAP dependence plot |
| GET | `/model/info` | Model details |
| GET | `/features/list` | Feature list |

---

## 🎨 Frontend Guide

### **4 Main Sections:**

#### 1️⃣ **🔮 Predict**
- Patient input form (11 clinical parameters)
- Field validation (age 18-120, BP 80-200, etc.)
- Sample patient loader for instant demo
- Real-time prediction with animated results
- **Risk stratification:**
  - 🟢 LOW (0-40%)
  - 🟡 MEDIUM (40-70%)
  - 🔴 HIGH (70-100%)

#### 2️⃣ **📊 Explain** (5 Tabs)
- **Waterfall** - Feature contributions (SHAP)
- **Force Plot** - Interactive visualization (SHAP)
- **LIME** - Model-agnostic explanation + weights table
- **Global Summary** - Population-level importance (SHAP)
- **Feature Importance** - Ranked features (SHAP)

#### 3️⃣ **💡 Insights**
- Cardiovascular health tips
- Nutrition guidelines
- Exercise recommendations
- Medical follow-up advice

#### 4️⃣ **ℹ️ About**
- Model information
- Performance metrics
- Technology stack
- Important disclaimers

---

## 📊 Model Performance

### **Dataset**
- **Total:** 2,141 patients
- **Training:** 1,712 (80%)
- **Test:** 429 (20%)
- **Features:** 35 clinical parameters

### **Metrics** (Test Set)

| Metric | Score |
|--------|-------|
| **F1-Score** | **87.6%** ⭐ |
| **Accuracy** | 86.5% |
| **Precision** | 86.9% |
| **Recall** | 88.4% |
| **Cross-Validation** | 85.0% ± 1.8% |

### **Confusion Matrix**

```
              Predicted
              No    Yes
Actual  No  [ 166    31 ]
        Yes [  27   205 ]
```

### **Top 10 Features**

1. **oldpeak** - ST depression
2. **age** - Patient age
3. **maxhr_final** - Maximum heart rate
4. **chol_final** - Cholesterol
5. **restingbp_final** - Resting BP
6. **exercise_angina** - Exercise-induced angina
7. **chest_pain** - Chest pain type
8. **st_slope** - ST slope
9. **sex** - Gender
10. **fasting_bs** - Fasting blood sugar

---

## 🔬 Explainability

### **SHAP (SHapley Additive exPlanations)**

**5 Visualization Types:**

1. **Waterfall** - Individual feature contributions
2. **Force Plot** - Interactive push/pull forces
3. **Summary** - Global beeswarm plot
4. **Importance** - Ranked bar chart
5. **Dependence** - Feature interactions

### **LIME (Local Interpretable Model-agnostic Explanations)**

- Creates simple linear model around prediction
- Model-agnostic (works with any ML model)
- Feature weights table
- Validates SHAP results

### **Why Use Both?**

| SHAP | LIME |
|------|------|
| Exact contributions | Approximate |
| Tree-optimized | Model-agnostic |
| Local + Global | Local only |
| Slower | Faster |

**Best Practice:** If SHAP and LIME agree → High confidence ✅

---

## 🚢 Deployment

### **Docker**

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api_advanced:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

### **Cloud Platforms**

**AWS (Elastic Beanstalk):**
```bash
eb init -p python-3.10 heart-disease-api
eb create heart-disease-env
eb deploy
```

**Google Cloud (Cloud Run):**
```bash
gcloud run deploy heart-disease-api \
  --source . --platform managed --region us-central1
```

**Heroku:**
```bash
heroku create heart-disease-api
git push heroku main
```

---

## 📁 Project Structure

```
heart-disease-ai/
│
├── api_advanced.py              # FastAPI backend
├── index_advanced.html          # Frontend HTML
├── style_advanced.css           # Frontend CSS
├── script_advanced.js           # Frontend JavaScript
├── requirements.txt             # Dependencies
├── README.md                    # This file
│
├── models/                      # Trained models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│   └── needs_scaling.pkl
│
├── data/                        # Training data
│   └── final_heart_clean.csv
│
└── outputs/                     # Visualizations
    ├── model_comparison.csv
    └── *.png
```

---

## ⚙️ Configuration

### **Change API Port**

```python
# In api_advanced.py
uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed to 8001
```

```javascript
// In script_advanced.js
const API_BASE = 'http://127.0.0.1:8001';  // Match port
```

### **Change Risk Thresholds**

```python
# In api_advanced.py
def get_risk_level(probability: float) -> str:
    if probability >= 0.7:  # HIGH threshold
        return "HIGH"
    elif probability >= 0.4:  # MEDIUM threshold
        return "MEDIUM"
    else:
        return "LOW"
```

### **Change Colors**

```css
/* In style_advanced.css */
:root {
    --primary: #2563eb;  /* Primary color */
    --danger: #ef4444;   /* High risk */
    --success: #10b981;  /* Low risk */
}
```

---

## 🐛 Troubleshooting

### **Port Already in Use**

```bash
# Windows - Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8000 | xargs kill -9
```

### **Model Files Not Found**

- Ensure files are in `models/` directory
- Check file paths in `api_advanced.py`
- Update paths to absolute paths if needed

### **CORS Errors**

```python
# In api_advanced.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all
)
```

### **SHAP/LIME Not Working**

```bash
# Reinstall with correct versions
pip install shap==0.43.0 lime==0.2.0.1
```

### **Scikit-learn Version Warning**

```bash
# Match model version
pip install scikit-learn==1.7.2
```

---

## 📄 License

**Educational Use Only**

⚠️ **Disclaimer:**
- NOT FDA approved
- NOT for clinical use
- NOT a substitute for medical advice
- Use as decision support only
- Consult healthcare professionals

---

## 🙏 Acknowledgments

**Datasets:**
- UCI Heart Disease Dataset
- Cleveland Heart Disease Database

**Libraries:**
- FastAPI, Scikit-learn, SHAP, LIME, Plotly

**Resources:**
- Interpretable ML (Christoph Molnar)
- SHAP Paper (Lundberg & Lee, 2017)

---

## 🗺️ Roadmap

### **Completed ✅**
- [x] Random Forest (87.6% F1)
- [x] FastAPI backend (17+ endpoints)
- [x] SHAP + LIME explanations
- [x] Modern web interface
- [x] Comprehensive docs

### **Planned 🎯**
- [ ] Unit tests
- [ ] Database integration
- [ ] User authentication
- [ ] PDF reports
- [ ] Mobile app

---

## 📊 API Performance

| Operation | Time |
|-----------|------|
| Single Prediction | ~100ms |
| Batch (10 patients) | ~500ms |
| SHAP Waterfall | ~200ms |
| LIME Explanation | ~300ms |

---

## 🎓 Learn More

**Machine Learning:**
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Random Forests Explained](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

**Explainable AI:**
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Tutorial](https://github.com/marcotcr/lime)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

**FastAPI:**
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

---

<div align="center">

**Built with ❤️ for better healthcare through AI**

**Version 2.0.0 | March 2026**

[⬆ Back to Top](#-heart-disease-prediction-system---production-grade-ai)

</div>










## 👨‍💻 Author

Developed as a **full production ML + DL + XAI pipeline project**.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
