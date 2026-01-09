import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Real-style economic dataset
# -----------------------------
data = {
    "income": [12000, 15000, 18000, 22000, 25000, 30000, 35000, 40000, 45000, 50000],
    "education_years": [8, 10, 12, 12, 14, 16, 16, 18, 18, 20],
    "employed": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["income", "education_years"]]
y = df["employed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

