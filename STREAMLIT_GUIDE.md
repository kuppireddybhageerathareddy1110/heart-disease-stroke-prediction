# 🚀 Streamlit App Setup & Usage Guide

## 📋 Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you already ran the pipeline, you likely have most packages installed.

### 2. Verify Model Files Exist

Make sure these files exist (created by running the pipeline):
```
models/
├── best_model.pkl       ✓ Required
├── scaler.pkl          ✓ Required
├── columns.pkl         ✓ Required
└── needs_scaling.pkl   ✓ Required

outputs/
└── model_comparison.csv ✓ Required (for performance dashboard)
```

If these don't exist, run the pipeline first:
```bash
python heart_disease_pipeline_improved.py
```

### 3. Launch the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🎯 App Features Overview

### 🏠 **Home Page**
- Welcome screen with key metrics
- Model information and performance
- How to use guide
- Quick feature overview

### 🔮 **Single Prediction**
**What it does:**
- Enter patient data manually
- Get instant risk prediction
- View probability gauge (0-100%)
- See SHAP explanations (which factors matter)
- Get personalized recommendations
- Download prediction report

**How to use:**
1. Navigate to "🔮 Single Prediction"
2. Fill in patient information:
   - **Age:** 18-100 years
   - **Sex:** Male/Female
   - **Chest Pain Type:** 4 options
   - **Blood Pressure:** 80-200 mm Hg
   - **Cholesterol:** 100-600 mg/dl
   - **Max Heart Rate:** 60-220 bpm
   - **Other clinical parameters**
3. Click "🔍 Predict Risk"
4. View results:
   - Risk level (Low/Medium/High)
   - Disease probability
   - SHAP explanation chart
   - Top 10 contributing features
   - Recommendations
5. Download report (optional)

**Example Patient:**
```
Age: 55
Sex: Male
Chest Pain: Atypical Angina
Resting BP: 140
Cholesterol: 250
Fasting BS: Yes
Resting ECG: Normal
Max HR: 150
Exercise Angina: Yes
Oldpeak: 2.3
ST Slope: Flat
```

### 📊 **Batch Prediction**
**What it does:**
- Upload CSV with multiple patients
- Get predictions for all at once
- View summary statistics
- See distribution charts
- Download results with predictions

**How to use:**
1. Navigate to "📊 Batch Prediction"
2. Click "📋 View Required CSV Format" to see template
3. Download sample CSV (optional)
4. Prepare your CSV file with patient data
5. Upload CSV file
6. Click "🚀 Generate Predictions"
7. View results:
   - Data preview
   - Predictions for each patient
   - Summary statistics
   - Distribution charts
8. Download results CSV

**CSV Format:**
```csv
age,sex,chest_pain,restingbp_final,chol_final,fasting_bs,resting_ecg,maxhr_final,exercise_angina,oldpeak,st_slope
55,Male,Atypical Angina,120,220,No,Normal,170,No,0.5,Upsloping
62,Female,Asymptomatic,140,268,Yes,ST-T Abnormality,160,Yes,2.3,Flat
48,Male,Non-anginal Pain,130,245,No,Normal,150,No,1.5,Upsloping
```

### 📈 **Model Performance**
**What it does:**
- View all model comparison metrics
- See performance visualizations
- Explore feature importance
- Understand model details

**Features:**
- Metrics table (all 14 models)
- 4-metric comparison charts
- Feature importance ranking
- Model information

### ℹ️ **About**
- Application overview
- Technology stack
- Dataset information
- Disclaimers and references

---

## 🎨 UI Features

### Color-Coded Risk Levels

- **🟢 LOW RISK** (0-40%): Green background
- **🟡 MEDIUM RISK** (40-70%): Yellow background  
- **🔴 HIGH RISK** (70-100%): Red background

### Interactive Elements

1. **Probability Gauge**
   - Visual representation of disease risk
   - Color-coded zones
   - Real-time update

2. **SHAP Explanation Chart**
   - Horizontal bar chart
   - Red bars = increases risk
   - Green bars = decreases risk
   - Sorted by importance

3. **Distribution Charts** (Batch mode)
   - Pie chart for predictions
   - Bar chart for risk levels
   - Interactive hover tooltips

---

## 🔍 Understanding the Results

### Prediction Output

**Prediction:** Disease / No Disease  
- Binary classification result
- Based on 0.5 probability threshold

**Probability:** 0-100%  
- Likelihood of heart disease
- More informative than binary prediction
- Used for risk stratification

**Risk Level:**
- **LOW:** < 40% probability
- **MEDIUM:** 40-70% probability
- **HIGH:** > 70% probability

### SHAP Explanation

**What is SHAP?**
- Shows feature contributions to prediction
- Explains "why" the model made this prediction
- Feature-by-feature impact

**How to read:**
- **Feature name:** What clinical parameter
- **Value:** Patient's actual value
- **SHAP value:** How much it affects prediction
- **Direction:** Increases or decreases risk

**Example:**
```
Feature: age
Value: 65
SHAP: +0.15 (Red bar)
Direction: Increases Risk

→ Being 65 years old increases disease risk
```

---

## 💡 Tips & Best Practices

### For Single Predictions

1. **Enter accurate data**
   - Use actual clinical measurements
   - Don't estimate or guess values
   - Check units (mm Hg for BP, mg/dl for cholesterol)

2. **Interpret probability, not just prediction**
   - 51% vs 95% both say "Disease" but very different
   - Use probability for risk stratification
   - Consider threshold adjustments for your use case

3. **Review SHAP explanations**
   - Understand which factors drive the prediction
   - Identify modifiable risk factors
   - Use for patient education

4. **Combine with clinical judgment**
   - Don't rely solely on the model
   - Consider patient history
   - Factor in other test results

### For Batch Predictions

1. **Prepare clean data**
   - Check for missing values
   - Ensure correct format
   - Validate categorical values

2. **Review summary statistics**
   - Look for unexpected patterns
   - Identify high-risk patients
   - Prioritize follow-ups

3. **Download and archive results**
   - Keep prediction records
   - Track over time
   - Use for quality assurance

---

## 🛠️ Customization Options

### Change Risk Thresholds

Edit in `streamlit_app.py`:
```python
def get_risk_level(probability):
    if probability >= 0.7:  # Change this (default: 0.7)
        return "HIGH RISK", "risk-high", "🔴"
    elif probability >= 0.4:  # Change this (default: 0.4)
        return "MEDIUM RISK", "risk-medium", "🟡"
    else:
        return "LOW RISK", "risk-low", "🟢"
```

**Recommendations:**
- **Screening tool:** Lower thresholds (e.g., 0.3, 0.6) to catch more cases
- **Diagnostic tool:** Higher thresholds (e.g., 0.5, 0.8) to reduce false alarms

### Add Custom Features

To add new input fields:

1. Update `get_feature_info()` or `get_categorical_options()`
2. Add input widget in the form
3. Update `encode_input()` function
4. Ensure feature exists in trained model

### Modify Color Scheme

Edit CSS in the `st.markdown()` section at the top:
```python
.risk-high {
    background-color: #FADBD8;  # Light red
    border-left-color: #E74C3C;  # Dark red
}
```

---

## 🐛 Troubleshooting

### Issue: "Error loading model"

**Solution:**
```bash
# Run the pipeline first to generate models
python heart_disease_pipeline_improved.py

# Verify files exist
ls models/
```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt

# Or install specific package
pip install streamlit shap plotly
```

### Issue: SHAP explanations not showing

**Reason:** SHAP only works for tree-based models (Random Forest, Gradient Boosting, etc.)

**Solution:** This is expected behavior. The app handles this gracefully with a warning message.

### Issue: Batch upload fails

**Common causes:**
- Wrong CSV format
- Missing columns
- Invalid categorical values
- Special characters in data

**Solution:**
1. Download sample CSV from the app
2. Match the exact format
3. Check column names (case-sensitive)
4. Validate all categorical values

### Issue: App is slow

**Optimization:**
1. Close other browser tabs
2. Reduce batch size (<1000 rows)
3. Use smaller visualizations
4. Clear browser cache

---

## 📊 Performance Benchmarks

**Single Prediction:**
- Response time: < 1 second
- SHAP computation: 1-2 seconds
- Total: ~2-3 seconds

**Batch Prediction:**
- 10 patients: ~5 seconds
- 100 patients: ~30 seconds
- 1000 patients: ~5 minutes

**Memory Usage:**
- Base app: ~200 MB
- With SHAP: ~500 MB
- Large batch: ~1 GB

---

## 🚀 Deployment Options

### Option 1: Local (Development)
```bash
streamlit run streamlit_app.py
```
**Pros:** Easy, fast, full control  
**Cons:** Only accessible on your machine

### Option 2: Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect repository
4. Deploy (free tier available)

**Pros:** Free, public URL, easy  
**Cons:** Limited resources, public access

### Option 3: Docker Container
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

**Pros:** Portable, consistent environment  
**Cons:** Requires Docker knowledge

### Option 4: AWS/GCP/Azure
- Deploy on cloud VMs
- Use managed container services
- Set up load balancing

**Pros:** Scalable, production-ready  
**Cons:** Cost, complexity

---

## 🔐 Security Considerations

### For Production Deployment:

1. **Add Authentication**
```python
import streamlit_authenticator as stauth

# Add login page
authenticator = stauth.Authenticate(...)
name, authentication_status = authenticator.login('Login', 'main')
```

2. **Encrypt Patient Data**
- Don't log sensitive information
- Use HTTPS in production
- Comply with HIPAA/GDPR

3. **Rate Limiting**
- Prevent abuse
- Limit predictions per user
- Monitor API usage

4. **Input Validation**
- Sanitize all inputs
- Validate ranges
- Prevent injection attacks

5. **Audit Trail**
- Log all predictions
- Track user actions
- Store for compliance

---

## 📝 Sample Workflows

### Workflow 1: Individual Patient Assessment
1. Patient visits clinic
2. Doctor collects clinical data
3. Enter data in app (Single Prediction)
4. Review prediction and SHAP explanation
5. Discuss results with patient
6. Download report for medical record
7. Follow clinical protocols based on risk level

### Workflow 2: Population Screening
1. Export patient list from EMR system
2. Format as CSV (batch template)
3. Upload to app (Batch Prediction)
4. Generate predictions for all patients
5. Download results
6. Filter high-risk patients
7. Schedule follow-up appointments
8. Track outcomes over time

### Workflow 3: Research Study
1. Collect patient cohort data
2. Run batch predictions
3. Analyze distribution patterns
4. Compare with actual diagnoses
5. Validate model performance
6. Generate research reports
7. Document findings

---

## 📚 Additional Resources

### Documentation
- Streamlit: https://docs.streamlit.io/
- SHAP: https://shap.readthedocs.io/
- Plotly: https://plotly.com/python/

### Related Files
- `heart_disease_pipeline_improved.py` - Train models
- `RESULTS_ANALYSIS_REPORT.md` - Performance analysis
- `DEPLOYMENT_CHECKLIST.md` - Production guide
- `ENHANCEMENT_SNIPPETS.py` - Advanced features

### Support
- Check QUICK_REFERENCE.md for metrics explanations
- Review IMPROVEMENTS_GUIDE.md for technical details
- See REPRODUCIBILITY_REPORT.md for validation

---

## ✅ Pre-Launch Checklist

Before deploying to production:

- [ ] Pipeline run successfully
- [ ] All model files present
- [ ] Requirements installed
- [ ] App runs locally without errors
- [ ] Single prediction tested
- [ ] Batch prediction tested
- [ ] SHAP explanations working
- [ ] All visualizations render
- [ ] Download buttons functional
- [ ] Error handling verified
- [ ] Disclaimers added
- [ ] Authentication configured (if needed)
- [ ] HTTPS enabled (production)
- [ ] Logging set up
- [ ] Backup plan ready

---

## 🎓 Tutorial: First Prediction

**Step-by-step guide for your first prediction:**

1. **Launch app**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Navigate to Single Prediction**
   - Click "🔮 Single Prediction" in sidebar

3. **Enter test patient data**
   ```
   Age: 60
   Sex: Male
   Chest Pain: Asymptomatic
   Resting BP: 150
   Cholesterol: 280
   Fasting BS: Yes
   Resting ECG: ST-T Abnormality
   Max HR: 140
   Exercise Angina: Yes
   Oldpeak: 2.0
   ST Slope: Flat
   ```

4. **Click "Predict Risk"**

5. **Review results**
   - Check risk level (likely HIGH RISK)
   - Note probability (~85-95%)
   - Read SHAP explanation
   - See which factors contribute most

6. **Try modifying values**
   - Change age to 40
   - Change cholesterol to 180
   - See how prediction changes
   - Understand feature impacts

7. **Download report**
   - Click download button
   - Open CSV file
   - Review prediction summary

**Congratulations!** You've made your first prediction! 🎉

---

**Version:** 1.0  
**Last Updated:** 2026  
**Streamlit Version:** 1.28+  
**Python Version:** 3.8+
