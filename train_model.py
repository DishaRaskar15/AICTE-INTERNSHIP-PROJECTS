import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample data: [experience, education_level, age, branch_encoded]
X = np.array([
    [1.0, 1, 22, 0],
    [2.0, 2, 25, 1],
    [0.5, 1, 21, 2],
    [3.0, 3, 30, 0],
    [4.0, 2, 28, 3],
    [1.5, 1, 24, 4],
    [5.0, 3, 32, 1],
    [2.5, 2, 27, 5],
])

# Target: salary in ₹
y = np.array([
    300000, 500000, 250000, 700000, 650000, 400000, 800000, 550000
])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save to salary_model.pkl
joblib.dump(model, "salary_model.pkl")
print("✅ Model saved as salary_model.pkl")
