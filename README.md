# AIML_6_HousePricePrediction_byte

## Project Overview
A multivariate Linear Regression model trained on the California Housing dataset to predict median house values using demographic and geographical indicators.

## Evaluation Metrics
- **Root Mean Squared Error (RMSE):** ~0.7456
- **Mean Absolute Error (MAE):** ~0.5332
- **Coefficient of Determination (R²):** ~0.5758

## Included Artifacts
- `task6_house_prices.py`: Complete pipeline script (data loading, preprocessing, training, and evaluation).
- `house_price_regression_model.pkl`: Serialized Scikit-Learn linear regression model.
- `feature_scaler.pkl`: StandardScaler transformer fitted on training features.
- `residual_plot_task6.png`: Diagnostic plot displaying model error distribution.

## Reproduction Instructions
1. Install requirements:
   ```bash
   pip install scikit-learn pandas numpy matplotlib joblib