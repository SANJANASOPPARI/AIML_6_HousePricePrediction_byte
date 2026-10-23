import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset directly using data and target attributes
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# 2. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# 6. Residual Plot
residuals = y_test.values - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.3, color="teal", edgecolor="k", s=20)
plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
plt.title("Residual Plot - House Price Prediction", fontsize=14)
plt.xlabel("Predicted Median House Value ($100,000s)", fontsize=11)
plt.ylabel("Residuals (Actual - Predicted)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("residual_plot_task6.png", dpi=300)
plt.close()

# 7. Save Model & Scaler Artifacts
joblib.dump(model, "house_price_regression_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

# 8. 5 Sample Predictions vs Actual Values
comparison_df = pd.DataFrame(
    {
        "Actual Price ($100k)": np.round(y_test.values[:5], 3),
        "Predicted Price ($100k)": np.round(y_pred[:5], 3),
        "Absolute Error": np.round(np.abs(y_test.values[:5] - y_pred[:5]), 3),
    }
)

print("\n--- 5 Sample Predictions vs Actuals ---")
print(comparison_df.to_string(index=False))