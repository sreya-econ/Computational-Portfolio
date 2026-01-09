# --------------------------------------------------
# Random Forest Regression
# Country Panel Data with Cross-Validation
# --------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# Load Panel Data
# --------------------------------------------------
data = pd.read_csv("worldbank_panel_data.csv")

# --------------------------------------------------
# Create Country Fixed Effects (Dummies)
# --------------------------------------------------
country_dummies = pd.get_dummies(data["country"], drop_first=True)

X_numeric = data[["investment", "inflation", "trade", "unemployment"]]
X = pd.concat([X_numeric, country_dummies], axis=1)

y = data["gdp_per_capita"]

# --------------------------------------------------
# Random Forest Model
# --------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42
)

# --------------------------------------------------
# K-Fold Cross Validation
# --------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rmse = -cross_val_score(
    rf,
    X,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error"
)

cv_r2 = cross_val_score(
    rf,
    X,
    y,
    cv=kf,
    scoring="r2"
)

print("Cross-Validation Results")
print("-------------------------")
print(f"Average RMSE: {cv_rmse.mean():.2f}")
print(f"Average R²: {cv_r2.mean():.3f}")

# --------------------------------------------------
# Fit Final Model
# --------------------------------------------------
rf.fit(X, y)
y_pred = rf.predict(X)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importance.plot(kind="barh")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.show()

# --------------------------------------------------
# Actual vs Predicted
# --------------------------------------------------
plt.figure()
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel("Actual GDP per Capita")
plt.ylabel("Predicted GDP per Capita")
plt.title("Actual vs Predicted GDP per Capita")
plt.show()
