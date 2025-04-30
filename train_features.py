
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def train_evaluate_models(X, y):
    print("Splitting data (train/test)...")
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Normalize where needed
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['RandomForest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'MAE': mean_absolute_error(y_test, rf_pred),
        'R2': r2_score(y_test, rf_pred),
        'model': rf
    }
    print("RandomForest feature importance:")
    print(dict(zip(X.columns, rf.feature_importances_)))

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    results['Ridge'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred)),
        'MAE': mean_absolute_error(y_test, ridge_pred),
        'R2': r2_score(y_test, ridge_pred),
        'model': ridge
    }

    # Extra Trees
    et = ExtraTreesRegressor(n_estimators=150, random_state=42)
    et.fit(X_train, y_train)
    et_pred = et.predict(X_test)
    results['ExtraTrees'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, et_pred)),
        'MAE': mean_absolute_error(y_test, et_pred),
        'R2': r2_score(y_test, et_pred),
        'model': et
    }

   
    return results