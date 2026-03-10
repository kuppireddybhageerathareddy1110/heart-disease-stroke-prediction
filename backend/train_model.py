import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Fill missing BMI
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Remove rare gender
df = df[df['gender'] != 'Other']

# Encode categorical
cat_cols = [
    'gender',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'
]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Features / target
X = df.drop(['stroke','id'], axis=1)
y = df['stroke']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

# Scale numeric columns
num_cols = ['age','avg_glucose_level','bmi']

scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train model
model = RandomForestClassifier()

model.fit(X_train,y_train)

# Save model and scaler
joblib.dump(model,"model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("Model training completed and saved.")