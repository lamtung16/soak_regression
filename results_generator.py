import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, FunctionTransformer
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline, FeatureUnion

np.random.seed(123)

# --- read dataset name ---
params_df = pd.read_csv("params.csv")
dataset = params_df.iloc[int(sys.argv[1])]['dataset']

# --- Load dataset ---
df = pd.read_csv(f"data/{dataset}.csv")
feature_cols = [c for c in df.columns if c not in ["subset", "y"]]

# Scale features
feature_scaler = MinMaxScaler()
df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])

# Scale target
target_scaler = MinMaxScaler()
df["y"] = target_scaler.fit_transform(df[["y"]])

# --- Model dictionary ---
def featureless_model(X_train, X_test, y_train, y_test):
    y_pred = np.repeat(y_train.mean(), len(y_test))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae


def cv_gam_model(X_train, X_test, y_train, y_test):
    features = FeatureUnion([
        ('poly', PolynomialFeatures()),
        ('exp_poly', Pipeline([
            ('poly', PolynomialFeatures()),
            ('exp', FunctionTransformer(np.exp, validate=False))
        ]))
    ])

    pipeline = Pipeline([
        ('features', features),
        ('scaler', MinMaxScaler()),
        ('lasso', Lasso(max_iter=100000, tol=0.001))
    ])

    # Grid search for best alpha
    param_grid = {'lasso__alpha': [0.001 * 2 ** i for i in range(16)]}
    grid = GridSearchCV(pipeline, param_grid, cv=4, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    return mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)


model_dict = {
    "featureless": featureless_model,
    "cv_gam": cv_gam_model
}

# --- record results ---
results = []

for subset in np.unique(df["subset"]):
    subset_idx = np.where(df["subset"] == subset)[0]
    if(len(subset_idx) < 10):
        continue
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
                    X_train=df[feature_cols].values[train_indices],
                    X_test=df[feature_cols].values[test_indices],
                    y_train=df["y"].values[train_indices],
                    y_test=df["y"].values[test_indices]
                )
                
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