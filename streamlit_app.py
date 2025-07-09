import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------------------
# Sample training data
data = pd.DataFrame({
    'YearsExperience': [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
    'EducationLevel': [1, 2, 2, 3, 3, 1, 2, 2, 3, 3],
    'JobRole': ['Analyst', 'Developer', 'Manager', 'Manager', 'Manager',
                'Analyst', 'Developer', 'Developer', 'Manager', 'Director'],
    'Salary': [30000, 50000, 70000, 90000, 105000, 35000, 55000, 75000, 95000, 120000]
})

X = data.drop('Salary', axis=1)
y = data['Salary']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['YearsExperience', 'EducationLevel']),
        ('cat', OneHotEncoder(), ['JobRole'])
    ]
)

# ------------------------------
# Train ML model
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', LinearRegression())])
lr_pipeline.fit(X, y)

# ------------------------------
# Train DL model
X_transformed = preprocessor.fit_transform(X)
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_transformed.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_transformed, y, epochs=100, verbose=0)

# ------------------------------
# Streamlit App
st.title("👔 Employee Salary Predictor")

st.write("### Enter Employee Details")

years_exp = st.slider("Years of Experience", 0, 20, 3)
education_level = st.selectbox("Education Level", {
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
})

job_role = st.selectbox("Job Role", ['Analyst', 'Developer', 'Manager', 'Director'])

if st.button("Predict Salary"):
    input_df = pd.DataFrame({
        'YearsExperience': [years_exp],
        'EducationLevel': [education_level],
        'JobRole': [job_role]
    })

    # Predict using ML model
    ml_salary = lr_pipeline.predict(input_df)[0]

    # Predict using DL model
    input_transformed = preprocessor.transform(input_df)
    dl_salary = model.predict(input_transformed)[0][0]

    st.success(f"💡 ML Predicted Salary: ₹{int(ml_salary):,}")
    st.success(f"🧠 DL Predicted Salary: ₹{int(dl_salary):,}")

