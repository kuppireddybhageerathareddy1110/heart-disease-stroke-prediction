from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

data = pd.read_csv("healthcare-dataset-stroke-data.csv")

data["bmi"] = data["bmi"].fillna(data["bmi"].median())
data = data[data["gender"] != "Other"]

data = pd.get_dummies(
    data,
    columns=[
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status"
    ],
    drop_first=True
)

X = data.drop(["stroke","id"],axis=1).astype(float)

model_columns = X.columns

explainer = shap.TreeExplainer(model)


@app.get("/")
def home():
    return {"message":"Stroke Prediction API Running"}


# ---------------------
# Prediction endpoint
# ---------------------
@app.post("/predict")
def predict(data:dict):

    df = pd.DataFrame([data])

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    num_cols = ["age","avg_glucose_level","bmi"]
    df[num_cols] = scaler.transform(df[num_cols])

    df = df.astype(float)

    prob = model.predict_proba(df)[0][1]

    return {
        "stroke_probability":float(prob),
        "risk_level":"High Risk" if prob>0.5 else "Low Risk"
    }


# ---------------------
# Local SHAP explanation
# ---------------------
@app.post("/explain")
def explain(data: dict):

    df = pd.DataFrame([data])

    # Ensure all model columns exist
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    # Scale numeric columns
    num_cols = ["age", "avg_glucose_level", "bmi"]
    df[num_cols] = scaler.transform(df[num_cols])

    df = df.astype(float)

    # Compute SHAP values
    shap_values = explainer(df)

    values = shap_values.values[0, :, 1]
    base = shap_values.base_values[0, 1]

    explanation = shap.Explanation(
        values=values,
        base_values=base,
        data=df.iloc[0],
        feature_names=model_columns
    )

    # Larger figure for better visualization
    plt.figure(figsize=(12,6))

    shap.plots.waterfall(explanation, show=False)

    # Fix label clipping
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)

    buf = BytesIO()

    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=150
    )

    buf.seek(0)

    img = base64.b64encode(buf.read()).decode("utf-8")

    plt.close()

    return {"plot": img}

# ---------------------
# Global SHAP summary plot
# ---------------------
@app.get("/summary_plot")

def summary_plot():

    sample = X.sample(200)

    shap_values = explainer(sample)

    values = shap_values.values[:,:,1]

    plt.figure(figsize=(8,6))

    shap.summary_plot(values,sample,show=False)

    buf = BytesIO()
    plt.savefig(buf,format="png")
    buf.seek(0)

    img = base64.b64encode(buf.read()).decode("utf-8")

    plt.close()

    return {"plot":img}

@app.get("/feature_importance")

def feature_importance():

    sample = X.sample(200)

    shap_values = explainer(sample)

    values = shap_values.values[:,:,1]

    plt.figure(figsize=(8,6))

    shap.summary_plot(
        values,
        sample,
        plot_type="bar",
        show=False
    )

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img = base64.b64encode(buf.read()).decode("utf-8")

    plt.close()

    return {"plot": img}

import io

import io

@app.post("/force_plot")
def force_plot(data: dict):

    df = pd.DataFrame([data])

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    num_cols = ["age","avg_glucose_level","bmi"]
    df[num_cols] = scaler.transform(df[num_cols])

    df = df.astype(float)

    shap_values = explainer(df)

    values = shap_values.values[0,:,1]
    base = shap_values.base_values[0,1]

    force = shap.force_plot(
        base,
        values,
        df.iloc[0],
        feature_names=model_columns
    )

    html_buffer = io.StringIO()
    shap.save_html(html_buffer, force)

    html_content = html_buffer.getvalue()

    return {"plot": html_content}