# ==============================
# IMPROVED HEART DISEASE AI PIPELINE
# ==============================

import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    classification_report, precision_recall_curve, f1_score,
    precision_score, recall_score, accuracy_score
)

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# DL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping

# LIME
from lime.lime_tabular import LimeTabularExplainer

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==============================
# CREATE FOLDERS
# ==============================

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

print("📂 Loading datasets...")
df1 = pd.read_csv("heart.csv")
df2 = pd.read_csv("heart_disease_uci (1).csv")
df3 = pd.read_csv("cleveland_heart_disease.csv")

df1.columns = df1.columns.str.lower()
df2.columns = df2.columns.str.lower()
df3.columns = df3.columns.str.lower()

df = pd.concat([df1, df2, df3], ignore_index=True)
print(f"✅ Loaded {len(df)} total records")

# ==============================
# FEATURE ENGINEERING
# ==============================

print("\n🔧 Engineering features...")
df["cp_final"] = df["chestpaintype"].fillna(df["cp"])
df["restingbp_final"] = df["restingbp"].fillna(df["trestbps"])
df["chol_final"] = df["cholesterol"].fillna(df["chol"])
df["maxhr_final"] = df["maxhr"].fillna(df["thalch"]).fillna(df["thalach"])
df["target_final"] = df["heartdisease"].fillna(df["num"]).fillna(df["target"])

df["fbs_final"] = df["fastingbs"].fillna(df["fbs"])
df["restecg_final"] = df["restingecg"].fillna(df["restecg"])
df["exang_final"] = df["exerciseangina"].fillna(df["exang"])
df["slope_final"] = df["st_slope"].fillna(df["slope"])

df = df[
    [
        "age", "sex", "oldpeak", "cp_final", "restingbp_final",
        "chol_final", "maxhr_final", "target_final",
        "fbs_final", "restecg_final", "exang_final", "slope_final"
    ]
]

# ==============================
# CLEANING
# ==============================

print("🧹 Cleaning data...")
num_cols = ["age", "oldpeak", "restingbp_final", "chol_final", "maxhr_final"]
cat_cols = ["sex", "cp_final", "fbs_final", "restecg_final", "exang_final", "slope_final"]

df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

df["sex"] = df["sex"].replace({"Male": "M", "Female": "F", 1.0: "M", 0.0: "F"})
df["fbs_final"] = df["fbs_final"].replace({True: "Yes", False: "No", 1: "Yes", 0: "No"})
df["exang_final"] = df["exang_final"].replace({True: "Y", False: "N"})
df["target_final"] = df["target_final"].apply(lambda x: 1 if x > 0 else 0)

# Check class balance
print(f"\n📊 Class distribution:")
print(df["target_final"].value_counts())
print(f"Class balance: {df['target_final'].value_counts(normalize=True).to_dict()}")

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

df.to_csv("data/final_heart_clean.csv", index=False)
print("✅ Clean dataset saved")

# ==============================
# SPLIT & SCALE
# ==============================

print("\n✂️ Splitting data...")
X = df.drop("target_final", axis=1)
y = df["target_final"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for SVM, KNN, Neural Networks)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# ==============================
# ML MODELS WITH CROSS-VALIDATION
# ==============================

print("\n🤖 Training ML models with 5-fold CV...")

ml_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis()
}

# Models that benefit from scaling
scale_models = ["SVM", "KNN", "Logistic Regression", "LDA"]

ml_results = {}
cv_scores = {}
test_metrics = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in ml_models.items():
    # Use scaled or unscaled data
    X_tr = X_train_scaled if name in scale_models else X_train
    X_te = X_test_scaled if name in scale_models else X_test
    
    # Cross-validation
    scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy')
    cv_scores[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    
    # Train on full training set
    model.fit(X_tr, y_train)
    
    # Test set predictions
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    
    # Compute multiple metrics
    test_metrics[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    ml_results[name] = test_metrics[name]['f1']  # Use F1 for model selection
    
    print(f"{name:20s} | CV: {cv_scores[name]['mean']:.4f} (±{cv_scores[name]['std']:.4f}) | "
          f"Test F1: {test_metrics[name]['f1']:.4f}")

# ==============================
# DL MODELS WITH EARLY STOPPING
# ==============================

print("\n🧠 Training Deep Learning models...")

X_train_dl = X_train_scaled.astype(np.float32).values
X_test_dl = X_test_scaled.astype(np.float32).values

def build_model(units_1, units_2=0, dropout=0, batch_norm=False):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(units_1, activation="relu"))
    
    if batch_norm:
        model.add(BatchNormalization())
    
    if dropout > 0:
        model.add(Dropout(dropout))
    
    if units_2 > 0:
        model.add(Dense(units_2, activation="relu"))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout > 0:
            model.add(Dropout(dropout))
    
    model.add(Dense(1, activation="sigmoid"))
    return model

dl_configs = [
    (64, 0, 0, False),
    (128, 64, 0.3, False),
    (128, 64, 0.3, True),
    (256, 128, 0.4, True)
]

dl_results = {}
dl_histories = {}

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for i, cfg in enumerate(dl_configs):
    model_name = f"DL_{i+1}"
    print(f"\nTraining {model_name} with config {cfg[:3]}...")
    
    model = build_model(*cfg)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    history = model.fit(
        X_train_dl, y_train,
        epochs=100,  # Increased with early stopping
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_test_dl, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    dl_results[model_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    dl_histories[model_name] = history
    
    print(f"{model_name}: F1 = {dl_results[model_name]['f1']:.4f}, "
          f"Acc = {dl_results[model_name]['accuracy']:.4f}")
    
    # Save training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.legend()
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.legend()
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig(f"outputs/dl_{i+1}_training.png", dpi=150)
    plt.close()
    
    # Save model
    model.save(f"models/dl_{i+1}_model.keras")

# ==============================
# BEST MODEL SELECTION
# ==============================

print("\n🏆 Selecting best model...")

# Combine all F1 scores
all_f1_scores = {**{k: v['f1'] for k, v in test_metrics.items()}, 
                 **{k: v['f1'] for k, v in dl_results.items()}}

best_name = max(all_f1_scores, key=all_f1_scores.get)
print(f"\n🥇 BEST MODEL: {best_name} (F1 = {all_f1_scores[best_name]:.4f})")

# Save best ML model (if it's ML)
if best_name in ml_models:
    best_model = ml_models[best_name]
    X_best = X_train_scaled if best_name in scale_models else X_train
    best_model.fit(X_best, y_train)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(X.columns.tolist(), "models/columns.pkl")
    joblib.dump(best_name in scale_models, "models/needs_scaling.pkl")
    print(f"✅ Best model saved: {best_name}")

# ==============================
# COMPREHENSIVE RESULTS TABLE
# ==============================

print("\n📊 COMPREHENSIVE RESULTS:")
print("=" * 80)
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("=" * 80)

for name in ml_models.keys():
    m = test_metrics[name]
    print(f"{name:<20} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
          f"{m['recall']:<10.4f} {m['f1']:<10.4f}")

for name in dl_results.keys():
    m = dl_results[name]
    print(f"{name:<20} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
          f"{m['recall']:<10.4f} {m['f1']:<10.4f}")

print("=" * 80)

# Save results to CSV
results_df = pd.DataFrame({**test_metrics, **dl_results}).T
results_df.to_csv("outputs/model_comparison.csv")

# ==============================
# VISUALIZATIONS
# ==============================

print("\n📈 Creating visualizations...")

# 1. Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
all_metrics = {**test_metrics, **dl_results}

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    values = [all_metrics[m][metric] for m in all_metrics.keys()]
    names = list(all_metrics.keys())
    
    bars = ax.bar(range(len(names)), values, color='steelblue')
    bars[names.index(best_name)].set_color('gold')
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/comprehensive_comparison.png", dpi=150)
plt.close()

# 2. Cross-validation scores
plt.figure(figsize=(12, 6))
cv_means = [cv_scores[name]['mean'] for name in ml_models.keys()]
cv_stds = [cv_scores[name]['std'] for name in ml_models.keys()]
names = list(ml_models.keys())

plt.bar(range(len(names)), cv_means, yerr=cv_stds, capsize=5, color='coral')
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('ML Models - 5-Fold Cross-Validation Scores')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/cv_scores.png", dpi=150)
plt.close()

# ==============================
# CONFUSION MATRIX (BEST MODEL)
# ==============================

if best_name in ml_models:
    X_eval = X_test_scaled if best_name in scale_models else X_test
    y_pred = best_model.predict(X_eval)
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {best_name}')
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

# ==============================
# ROC CURVE (TOP 5 MODELS)
# ==============================

plt.figure(figsize=(10, 8))

# Plot for top 5 ML models
top_5_ml = sorted(test_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)[:5]

for name, _ in top_5_ml:
    X_eval = X_test_scaled if name in scale_models else X_test
    model = ml_models[name]
    y_prob = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Top 5 Models')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("outputs/roc_curves.png", dpi=150)
plt.close()

# ==============================
# FEATURE IMPORTANCE (if tree-based)
# ==============================

if best_name in ["Random Forest", "Gradient Boosting", "Extra Trees", "Decision Tree"]:
    print("\n🌳 Analyzing feature importance...")
    
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), X.columns[indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance - {best_name}')
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()

# ==============================
# SHAP ANALYSIS (if tree-based)
# ==============================

if best_name in ["Random Forest", "Gradient Boosting", "Extra Trees", "AdaBoost"]:
    print("\n🔍 Computing SHAP values...")
    
    X_eval = X_test_scaled if best_name in scale_models else X_test
    X_train_eval = X_train_scaled if best_name in scale_models else X_train
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train_eval)
    
    # Handle different SHAP formats
    if isinstance(shap_values, list):
        shap_plot = shap_values[1]
    else:
        shap_plot = shap_values
    
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_plot, X_train_eval, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot
    plt.figure()
    shap.summary_plot(shap_plot, X_train_eval, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()

# ==============================
# LIME EXPLANATION
# ==============================

if best_name in ml_models:
    print("\n🔬 Generating LIME explanation...")
    
    X_eval = X_test_scaled if best_name in scale_models else X_test
    X_train_eval = X_train_scaled if best_name in scale_models else X_train
    
    explainer_lime = LimeTabularExplainer(
        X_train_eval.values,
        feature_names=X.columns.tolist(),
        class_names=["No Disease", "Disease"],
        mode="classification"
    )
    
    # Explain first test instance
    exp = explainer_lime.explain_instance(
        X_eval.iloc[0].values,
        best_model.predict_proba,
        num_features=10
    )
    
    exp.save_to_file("outputs/lime_explanation.html")
    print("✅ LIME explanation saved")

# ==============================
# CLASSIFICATION REPORT
# ==============================

if best_name in ml_models:
    X_eval = X_test_scaled if best_name in scale_models else X_test
    y_pred = best_model.predict(X_eval)
    
    print("\n📋 Classification Report (Best Model):")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

# ==============================
# SAVE SUMMARY
# ==============================

summary = {
    'best_model': best_name,
    'best_f1_score': all_f1_scores[best_name],
    'total_samples': len(df),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'num_features': X.shape[1],
    'class_distribution': y.value_counts().to_dict()
}

with open('outputs/pipeline_summary.txt', 'w') as f:
    f.write("HEART DISEASE PREDICTION PIPELINE SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

print("\n" + "=" * 60)
print("🎉 PIPELINE COMPLETE!")
print("=" * 60)
print(f"✅ Best Model: {best_name}")
print(f"✅ F1-Score: {all_f1_scores[best_name]:.4f}")
print(f"✅ All outputs saved to 'outputs/' directory")
print(f"✅ Models saved to 'models/' directory")
print("=" * 60)
