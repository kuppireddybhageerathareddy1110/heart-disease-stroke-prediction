"""
Heart Disease Prediction Web Application
==========================================
A Streamlit app with explainable AI for heart disease prediction.

Features:
- Interactive patient data input
- Real-time predictions with probability
- SHAP explanations for predictions
- Feature importance visualization
- Model performance dashboard
- Batch prediction upload
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E74C3C;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495E;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498DB;
    }
    .risk-high {
        background-color: #FADBD8;
        border-left-color: #E74C3C;
    }
    .risk-medium {
        background-color: #FCF3CF;
        border-left-color: #F39C12;
    }
    .risk-low {
        background-color: #D5F4E6;
        border-left-color: #27AE60;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL AND ARTIFACTS
# ==========================================

@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts."""
    try:
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/columns.pkl")
        needs_scaling = joblib.load("models/needs_scaling.pkl")
        
        return model, scaler, feature_names, needs_scaling
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure you've run the pipeline first to generate model files.")
        st.stop()

# Load model
model, scaler, feature_names, needs_scaling = load_model_artifacts()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_feature_info():
    """Get information about each feature for the input form."""
    return {
        'age': {
            'label': 'Age',
            'min': 18, 'max': 100, 'default': 50,
            'help': 'Patient age in years'
        },
        'oldpeak': {
            'label': 'ST Depression (Oldpeak)',
            'min': 0.0, 'max': 6.0, 'default': 1.0,
            'help': 'ST depression induced by exercise relative to rest'
        },
        'restingbp_final': {
            'label': 'Resting Blood Pressure',
            'min': 80, 'max': 200, 'default': 120,
            'help': 'Resting blood pressure in mm Hg'
        },
        'chol_final': {
            'label': 'Cholesterol',
            'min': 100, 'max': 600, 'default': 200,
            'help': 'Serum cholesterol in mg/dl'
        },
        'maxhr_final': {
            'label': 'Maximum Heart Rate',
            'min': 60, 'max': 220, 'default': 150,
            'help': 'Maximum heart rate achieved'
        }
    }

def get_categorical_options():
    """Get options for categorical features."""
    return {
        'sex': ['Male', 'Female'],
        'chest_pain': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
        'fasting_bs': ['No (≤ 120 mg/dl)', 'Yes (> 120 mg/dl)'],
        'resting_ecg': ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'],
        'exercise_angina': ['No', 'Yes'],
        'st_slope': ['Upsloping', 'Flat', 'Downsloping']
    }

def encode_input(age, sex, oldpeak, chest_pain, restingbp, chol, maxhr, 
                 fasting_bs, resting_ecg, exercise_angina, st_slope):
    """Encode user inputs into feature format expected by model."""
    
    # Create a dictionary with all features initialized to 0
    data = {feature: 0 for feature in feature_names}
    
    # Set numeric features
    data['age'] = age
    data['oldpeak'] = oldpeak
    data['restingbp_final'] = restingbp
    data['chol_final'] = chol
    data['maxhr_final'] = maxhr
    
    # Encode categorical features (one-hot)
    if sex == 'Male':
        if 'sex_M' in data:
            data['sex_M'] = 1
    
    # Chest pain type (assuming encoding: 0=TA, 1=ATA, 2=NAP, 3=ASY)
    cp_mapping = {
        'Typical Angina': 0,
        'Atypical Angina': 1, 
        'Non-anginal Pain': 2,
        'Asymptomatic': 3
    }
    cp_value = cp_mapping[chest_pain]
    for i in range(1, 4):  # cp_final_1, cp_final_2, cp_final_3
        key = f'cp_final_{i}'
        if key in data:
            data[key] = 1 if i == cp_value else 0
    
    # Fasting blood sugar
    if fasting_bs == 'Yes (> 120 mg/dl)' and 'fbs_final_Yes' in data:
        data['fbs_final_Yes'] = 1
    
    # Resting ECG
    ecg_mapping = {'Normal': 0, 'ST-T Abnormality': 1, 'LV Hypertrophy': 2}
    ecg_value = ecg_mapping[resting_ecg]
    for i in range(1, 3):
        key = f'restecg_final_{i}'
        if key in data:
            data[key] = 1 if i == ecg_value else 0
    
    # Exercise angina
    if exercise_angina == 'Yes' and 'exang_final_Y' in data:
        data['exang_final_Y'] = 1
    
    # ST slope
    slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    slope_value = slope_mapping[st_slope]
    for i in range(1, 3):
        key = f'slope_final_{i}'
        if key in data:
            data[key] = 1 if i == slope_value else 0
    
    # Convert to DataFrame
    df = pd.DataFrame([data])[feature_names]
    return df

def predict(patient_data):
    """Make prediction for patient data."""
    # Scale if needed
    if needs_scaling:
        patient_data_processed = scaler.transform(patient_data)
    else:
        patient_data_processed = patient_data.values
    
    # Predict
    prediction = model.predict(patient_data_processed)[0]
    probability = model.predict_proba(patient_data_processed)[0]
    
    return prediction, probability

def get_risk_level(probability):
    """Determine risk level from probability."""
    if probability >= 0.7:
        return "HIGH RISK", "risk-high", "🔴"
    elif probability >= 0.4:
        return "MEDIUM RISK", "risk-medium", "🟡"
    else:
        return "LOW RISK", "risk-low", "🟢"

# ==========================================
# SIDEBAR - NAVIGATION
# ==========================================

st.sidebar.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", 
     "📈 Model Performance", "ℹ️ About"]
)

# ==========================================
# PAGE: HOME
# ==========================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">🫀 Heart Disease Prediction System</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #EBF5FB; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h2 style='color: #2E86C1; margin-top: 0;'>Welcome to the AI-Powered Heart Disease Predictor</h2>
        <p style='font-size: 1.1rem; color: #34495E;'>
            This application uses advanced machine learning to predict the risk of heart disease 
            based on clinical parameters. Our model has achieved <strong>87.6% F1-Score</strong> 
            with <strong>explainable AI</strong> features.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate Predictions</h3>
            <p>87.6% F1-Score</p>
            <p>86.5% Accuracy</p>
            <p>88.4% Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Explainable AI</h3>
            <p>SHAP explanations</p>
            <p>Feature importance</p>
            <p>Risk factor analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-Time</h3>
            <p>Instant predictions</p>
            <p>Interactive interface</p>
            <p>Batch processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Information
    st.markdown('<p class="sub-header">📊 Model Information</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Model Type:** Random Forest Classifier  
        **Training Samples:** 1,712 patients  
        **Test Samples:** 429 patients  
        **Features Used:** {len(feature_names)}  
        **Scaling Required:** {'Yes' if needs_scaling else 'No'}
        """)
    
    with col2:
        st.success("""
        **Performance Metrics:**
        - **Precision:** 86.9% (Few false alarms)
        - **Recall:** 88.4% (Catches most cases)
        - **F1-Score:** 87.6% (Excellent balance)
        - **Cross-Validation:** 85.0% ± 1.8%
        """)
    
    # How to Use
    st.markdown('<p class="sub-header">🚀 How to Use</p>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Single Prediction:** Enter patient data manually for instant prediction
    2. **Batch Prediction:** Upload a CSV file for multiple predictions
    3. **Model Performance:** View detailed performance metrics and visualizations
    4. **Explainability:** Understand which factors drive each prediction
    
    ---
    
    ⚠️ **Disclaimer:** This tool is for educational and research purposes only. 
    It should not replace professional medical advice, diagnosis, or treatment.
    """)

# ==========================================
# PAGE: SINGLE PREDICTION
# ==========================================

elif page == "🔮 Single Prediction":
    st.markdown('<p class="main-header">🔮 Single Patient Prediction</p>', unsafe_allow_html=True)
    
    st.markdown("Enter patient information below to get a heart disease risk assessment.")
    
    # Input Form
    with st.form("prediction_form"):
        st.markdown("### 👤 Patient Information")
        
        col1, col2 = st.columns(2)
        
        feature_info = get_feature_info()
        cat_options = get_categorical_options()
        
        with col1:
            age = st.number_input(
                feature_info['age']['label'],
                min_value=feature_info['age']['min'],
                max_value=feature_info['age']['max'],
                value=feature_info['age']['default'],
                help=feature_info['age']['help']
            )
            
            sex = st.selectbox("Sex", cat_options['sex'])
            
            chest_pain = st.selectbox(
                "Chest Pain Type",
                cat_options['chest_pain'],
                help="Type of chest pain experienced"
            )
            
            restingbp = st.number_input(
                feature_info['restingbp_final']['label'],
                min_value=feature_info['restingbp_final']['min'],
                max_value=feature_info['restingbp_final']['max'],
                value=feature_info['restingbp_final']['default'],
                help=feature_info['restingbp_final']['help']
            )
            
            chol = st.number_input(
                feature_info['chol_final']['label'],
                min_value=feature_info['chol_final']['min'],
                max_value=feature_info['chol_final']['max'],
                value=feature_info['chol_final']['default'],
                help=feature_info['chol_final']['help']
            )
        
        with col2:
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar",
                cat_options['fasting_bs'],
                help="Fasting blood sugar level"
            )
            
            resting_ecg = st.selectbox(
                "Resting ECG",
                cat_options['resting_ecg'],
                help="Resting electrocardiographic results"
            )
            
            maxhr = st.number_input(
                feature_info['maxhr_final']['label'],
                min_value=feature_info['maxhr_final']['min'],
                max_value=feature_info['maxhr_final']['max'],
                value=feature_info['maxhr_final']['default'],
                help=feature_info['maxhr_final']['help']
            )
            
            exercise_angina = st.selectbox(
                "Exercise-Induced Angina",
                cat_options['exercise_angina'],
                help="Angina induced by exercise"
            )
            
            oldpeak = st.number_input(
                feature_info['oldpeak']['label'],
                min_value=feature_info['oldpeak']['min'],
                max_value=feature_info['oldpeak']['max'],
                value=feature_info['oldpeak']['default'],
                step=0.1,
                help=feature_info['oldpeak']['help']
            )
            
            st_slope = st.selectbox(
                "ST Slope",
                cat_options['st_slope'],
                help="Slope of peak exercise ST segment"
            )
        
        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)
    
    # Make Prediction
    if submitted:
        with st.spinner("Analyzing patient data..."):
            # Encode inputs
            patient_data = encode_input(
                age, sex, oldpeak, chest_pain, restingbp, chol, maxhr,
                fasting_bs, resting_ecg, exercise_angina, st_slope
            )
            
            # Get prediction
            prediction, probability = predict(patient_data)
            disease_prob = probability[1]
            risk_level, risk_class, risk_icon = get_risk_level(disease_prob)
            
            # Display Results
            st.markdown("---")
            st.markdown("## 📋 Prediction Results")
            
            # Risk Assessment Card
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div class="metric-card {risk_class}" style="text-align: center; padding: 2rem;">
                    <h1 style="font-size: 3rem; margin: 0;">{risk_icon}</h1>
                    <h2 style="margin: 1rem 0;">{risk_level}</h2>
                    <h3 style="color: #34495E;">Disease Probability: {disease_prob:.1%}</h3>
                    <p style="margin-top: 1rem; color: #7F8C8D;">
                        {'High risk of heart disease detected' if prediction == 1 else 'Low risk of heart disease'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability Gauge
            st.markdown("### 📊 Risk Probability")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=disease_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Disease Risk %", 'font': {'size': 24}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#D5F4E6'},
                        {'range': [40, 70], 'color': '#FCF3CF'},
                        {'range': [70, 100], 'color': '#FADBD8'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Explanation (if tree-based model)
            st.markdown("### 🔍 Explainability: What's Driving This Prediction?")
            
            try:
                # Generate SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(patient_data)
                
                # Handle different SHAP formats
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]  # For disease class
                else:
                    shap_vals = shap_values[0]
                
                # Create waterfall plot
                st.markdown("#### Feature Contributions")
                
                # Sort features by absolute SHAP value
                feature_shap = list(zip(feature_names, shap_vals, patient_data.values[0]))
                feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Take top 10
                top_features = feature_shap[:10]
                
                # Create horizontal bar chart
                fig = go.Figure()
                
                colors = ['#E74C3C' if val > 0 else '#27AE60' for _, val, _ in top_features]
                
                fig.add_trace(go.Bar(
                    y=[f[0] for f in top_features],
                    x=[f[1] for f in top_features],
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f'{f[1]:.3f}' for f in top_features],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Top 10 Features Contributing to Prediction",
                    xaxis_title="SHAP Value (Impact on Prediction)",
                    yaxis_title="Feature",
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature values table
                st.markdown("#### Feature Values")
                
                feature_df = pd.DataFrame({
                    'Feature': [f[0] for f in top_features],
                    'Value': [f'{f[2]:.2f}' for f in top_features],
                    'Impact': [f'{f[1]:.4f}' for f in top_features],
                    'Direction': ['Increases Risk' if f[1] > 0 else 'Decreases Risk' for f in top_features]
                })
                
                st.dataframe(feature_df, use_container_width=True)
                
                # Interpretation
                st.info("""
                **How to interpret:**
                - **Red bars** (positive SHAP values) increase disease risk
                - **Green bars** (negative SHAP values) decrease disease risk
                - **Larger bars** have more impact on the prediction
                """)
                
            except Exception as e:
                st.warning("SHAP explanations are only available for tree-based models.")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if prediction == 1:
                st.error("""
                **⚠️ High Risk Detected:**
                - Consult with a cardiologist as soon as possible
                - Consider comprehensive cardiac evaluation
                - Review lifestyle factors (diet, exercise, stress)
                - Monitor blood pressure and cholesterol regularly
                - Follow medical advice for risk reduction
                """)
            else:
                st.success("""
                **✅ Low Risk:**
                - Maintain healthy lifestyle habits
                - Regular health check-ups recommended
                - Continue monitoring cardiovascular health
                - Stay physically active
                - Eat a heart-healthy diet
                """)
            
            # Download Report
            st.markdown("### 📄 Download Report")
            
            report_data = {
                'Patient_Age': [age],
                'Sex': [sex],
                'Chest_Pain': [chest_pain],
                'Resting_BP': [restingbp],
                'Cholesterol': [chol],
                'Max_HR': [maxhr],
                'Prediction': ['Disease' if prediction == 1 else 'No Disease'],
                'Probability': [f'{disease_prob:.2%}'],
                'Risk_Level': [risk_level]
            }
            
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Download Prediction Report",
                data=csv,
                file_name=f"heart_disease_prediction_{age}y_{sex}.csv",
                mime="text/csv"
            )

# ==========================================
# PAGE: BATCH PREDICTION
# ==========================================

elif page == "📊 Batch Prediction":
    st.markdown('<p class="main-header">📊 Batch Prediction</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a CSV file with patient data to get predictions for multiple patients at once.
    """)
    
    # Sample Template
    with st.expander("📋 View Required CSV Format"):
        st.markdown("""
        Your CSV file should contain the following columns:
        
        **Numeric Features:**
        - age
        - oldpeak
        - restingbp_final
        - chol_final
        - maxhr_final
        
        **Categorical Features (use exact values):**
        - sex: Male, Female
        - chest_pain: Typical Angina, Atypical Angina, Non-anginal Pain, Asymptomatic
        - fasting_bs: No, Yes
        - resting_ecg: Normal, ST-T Abnormality, LV Hypertrophy
        - exercise_angina: No, Yes
        - st_slope: Upsloping, Flat, Downsloping
        """)
        
        # Create sample template
        sample_data = {
            'age': [55, 62, 48],
            'sex': ['Male', 'Female', 'Male'],
            'chest_pain': ['Atypical Angina', 'Asymptomatic', 'Non-anginal Pain'],
            'restingbp_final': [120, 140, 130],
            'chol_final': [220, 268, 245],
            'fasting_bs': ['No', 'Yes', 'No'],
            'resting_ecg': ['Normal', 'ST-T Abnormality', 'Normal'],
            'maxhr_final': [170, 160, 150],
            'exercise_angina': ['No', 'Yes', 'No'],
            'oldpeak': [0.5, 2.3, 1.5],
            'st_slope': ['Upsloping', 'Flat', 'Upsloping']
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Sample CSV",
            csv_sample,
            "sample_patients.csv",
            "text/csv"
        )
    
    # File Upload
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} patients.")
            
            # Show preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head())
            
            # Process predictions
            if st.button("🚀 Generate Predictions", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    predictions = []
                    probabilities = []
                    risk_levels = []
                    
                    for idx, row in df.iterrows():
                        try:
                            # Encode row
                            patient_data = encode_input(
                                row['age'], row['sex'], row['oldpeak'],
                                row['chest_pain'], row['restingbp_final'],
                                row['chol_final'], row['maxhr_final'],
                                row['fasting_bs'], row['resting_ecg'],
                                row['exercise_angina'], row['st_slope']
                            )
                            
                            # Predict
                            pred, prob = predict(patient_data)
                            disease_prob = prob[1]
                            risk, _, _ = get_risk_level(disease_prob)
                            
                            predictions.append('Disease' if pred == 1 else 'No Disease')
                            probabilities.append(f'{disease_prob:.2%}')
                            risk_levels.append(risk)
                            
                        except Exception as e:
                            predictions.append('Error')
                            probabilities.append('N/A')
                            risk_levels.append('N/A')
                    
                    # Add results to dataframe
                    df['Prediction'] = predictions
                    df['Probability'] = probabilities
                    df['Risk_Level'] = risk_levels
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(df)
                    
                    # Summary statistics
                    st.markdown("### 📈 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total = len(df)
                    disease_count = (df['Prediction'] == 'Disease').sum()
                    high_risk = (df['Risk_Level'] == 'HIGH RISK').sum()
                    medium_risk = (df['Risk_Level'] == 'MEDIUM RISK').sum()
                    
                    with col1:
                        st.metric("Total Patients", total)
                    with col2:
                        st.metric("Disease Predicted", disease_count)
                    with col3:
                        st.metric("High Risk", high_risk)
                    with col4:
                        st.metric("Medium Risk", medium_risk)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Prediction distribution
                        pred_counts = df['Prediction'].value_counts()
                        fig = px.pie(
                            values=pred_counts.values,
                            names=pred_counts.index,
                            title="Prediction Distribution",
                            color_discrete_sequence=['#27AE60', '#E74C3C']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Risk level distribution
                        risk_counts = df['Risk_Level'].value_counts()
                        fig = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            title="Risk Level Distribution",
                            color=risk_counts.index,
                            color_discrete_map={
                                'LOW RISK': '#27AE60',
                                'MEDIUM RISK': '#F39C12',
                                'HIGH RISK': '#E74C3C'
                            }
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.markdown("### 📄 Download Results")
                    
                    csv_result = df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        csv_result,
                        "batch_predictions_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV matches the required format.")

# ==========================================
# PAGE: MODEL PERFORMANCE
# ==========================================

elif page == "📈 Model Performance":
    st.markdown('<p class="main-header">📈 Model Performance Dashboard</p>', unsafe_allow_html=True)
    
    # Load performance data
    try:
        results_df = pd.read_csv("outputs/model_comparison.csv", index_col=0)
        
        st.markdown("### 🏆 Model Comparison")
        
        # Display metrics table
        st.dataframe(
            results_df.style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        
        # Visualizations
        st.markdown("### 📊 Performance Visualizations")
        
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score']
        )
        
        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            values = results_df[metric].sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=values.values,
                    y=values.index,
                    orientation='h',
                    name=metric.capitalize(),
                    marker_color='steelblue'
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🌳 Feature Importance")
            
            importances = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(
                feature_imp_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features",
                color='Importance',
                color_continuous_scale='Blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        st.markdown("### ℹ️ Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Model Type:** {type(model).__name__}  
            **Number of Features:** {len(feature_names)}  
            **Best F1-Score:** {results_df.loc['Random Forest', 'f1']:.4f}  
            **Best Accuracy:** {results_df.loc['Random Forest', 'accuracy']:.4f}
            """)
        
        with col2:
            st.success("""
            **Training Details:**
            - 5-fold cross-validation
            - Stratified train/test split
            - Feature scaling applied
            - Random state: 42
            """)
    
    except Exception as e:
        st.error("Performance data not found. Please run the pipeline first.")

# ==========================================
# PAGE: ABOUT
# ==========================================

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Application</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🫀 Heart Disease Prediction System
    
    This application uses machine learning to predict the risk of heart disease based on clinical parameters.
    
    ### 🎯 Model Performance
    
    Our Random Forest model achieves:
    - **87.6% F1-Score** - Excellent balance of precision and recall
    - **86.5% Accuracy** - High overall correctness
    - **88.4% Recall** - Catches 88% of disease cases
    - **86.9% Precision** - Low false alarm rate
    
    ### 🔬 Technology Stack
    
    - **Machine Learning:** Scikit-learn, Random Forest
    - **Explainable AI:** SHAP, LIME
    - **Web Framework:** Streamlit
    - **Visualization:** Plotly, Matplotlib, Seaborn
    - **Data Processing:** Pandas, NumPy
    
    ### 📊 Dataset
    
    The model was trained on **2,141 patient records** from three merged heart disease datasets:
    - Comprehensive clinical features
    - Balanced class distribution (54% disease, 46% no disease)
    - High-quality curated data
    
    ### ⚠️ Important Disclaimers
    
    - This tool is for **educational and research purposes only**
    - **Not FDA approved** for clinical use
    - Should **not replace professional medical advice**
    - Use as a **decision support tool**, not sole decision maker
    - Always consult with healthcare professionals
    
    ### 📚 References
    
    - Random Forests (Breiman, 2001)
    - SHAP Values (Lundberg & Lee, 2017)
    - Heart Disease UCI Dataset
    
    ### 👥 Credits
    
    Developed using state-of-the-art machine learning techniques and best practices
    in explainable AI.
    
    ---
    
    **Version:** 1.0  
    **Last Updated:** 2026
    """)
    
    # Contact form placeholder
    st.markdown("### 📬 Feedback")
    st.info("Have questions or suggestions? We'd love to hear from you!")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7F8C8D; padding: 1rem;'>
    <p>Heart Disease Prediction System | Powered by Machine Learning & Explainable AI</p>
    <p>⚠️ For educational purposes only | Not a substitute for professional medical advice</p>
</div>
""", unsafe_allow_html=True)
