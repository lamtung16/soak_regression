import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

np.random.seed(123)

# --- read dataset name ---
params_df = pd.read_csv("params.csv")
dataset = params_df.iloc[int(sys.argv[1])]['dataset']


# --- Load dataset ---
df = pd.read_csv(f"data/{dataset}.csv")
feature_cols = [c for c in df.columns if c not in ["subset", "y"]]


# --- Model dictionary ---
def featureless_model(X_train, X_test, y_train, y_test):
    # Scale the target
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    
    # Predict the mean of the scaled training target
    y_pred_scaled = np.repeat(y_train_scaled.mean(), len(y_test_scaled))
    
    # Metrics on scaled target
    mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)
    
    return mse_scaled, mae_scaled


def cv_gam_model(X_train, X_test, y_train, y_test):
    # Scale the target
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Pipeline for X features
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter=10000, tol=0.1))
    ])

    # Grid search for best L1 alpha
    param_grid = {'lasso__alpha': [0.01, 1, 2, 10]}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train_scaled)

    # Predictions (scaled)
    y_pred_scaled = grid.predict(X_test)

    # Metrics on scaled target
    mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)
    
    return mse_scaled, mae_scaled


model_dict = {
    "featureless": featureless_model,
    "cv_gam": cv_gam_model
}


# --- record results ---
results = []

for subset in np.unique(df["subset"]):
    subset_idx = np.where(df["subset"] == subset)[0]
    kf = KFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(kf.split(subset_idx)):
        test_indices = subset_idx[test_idx]
        same_indices = subset_idx[train_idx]
        other_indices = np.where(df["subset"] != subset)[0]
        all_indices = np.concatenate([same_indices, other_indices])
        
        train_dict = {
            "same": same_indices,
            "other": other_indices,
            "all": all_indices
        }
        
        for category, train_indices in train_dict.items():
            for model in ["featureless", "cv_gam"]:
                mse, mae = model_dict[model](
                    X_train = df[feature_cols].values[train_indices],
                    X_test  = df[feature_cols].values[test_indices],
                    y_train = df["y"].values[train_indices],
                    y_test  = df["y"].values[test_indices])
                
                results.append({
                    "subset": subset,
                    "category": category,
                    "test_fold": fold + 1,
                    "model": model,
                    "mse": mse,
                    "mae": mae
                })

results_df = pd.DataFrame(results)
results_df.to_csv(f"results/{dataset}.csv", index=False)