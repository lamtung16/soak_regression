import sys
import os
import numpy as np
import pandas as pd
from soak_low_level import SOAKFold

# --- Read dataset name from params.csv ---
params = np.genfromtxt("params.csv", delimiter=",", dtype=None, names=True, encoding="utf-8")
idx = int(sys.argv[1])
dataset = params[idx]["dataset"]
subset_col = params[idx]["subset_col"]
target_col = params[idx]["target_col"]
log_target = bool(params[idx]["log_target"])
n_folds = 5
n_seeds = 3

# Load data
data = np.genfromtxt(f'data/{dataset}.csv.xz', delimiter=',', dtype=None, names=True, encoding=None, invalid_raise=False)
X = np.column_stack([data[name] for name in data.dtype.names if name.startswith('X_')])
y = data[target_col]
subset_vec = data[subset_col]

# model list
model_list = ['featureless', 'linear', 'linear_BasExp', 'tree']

# replace orignal label by its log scale
if log_target:
    y = np.log(y)
y = (y - np.mean(y)) / np.std(y)

# --- Initialize soak object ---
soak_obj = SOAKFold(n_splits=n_folds)

# --- Analyze all subsets ---
results = []
for subset_value, category, fold_id, X_train, y_train, X_test, y_test in soak_obj.split(X, y, subset_vec):
    for model in model_list:
        y_pred, (rmse, mae) = soak_obj.model_eval(X_train, y_train, X_test, y_test, model)
        predictions_path = f"predictions/{dataset}/{subset_col}"
        os.makedirs(predictions_path, exist_ok=True)
        np.savez_compressed(f"{predictions_path}/{subset_value}.{category}.{fold_id}.{model}.{False}.{0}.{len(y_train)}.npz",y=y)
        results.append({
                    "subset": subset_value,
                    "category": category,
                    "fold_id": fold_id,
                    "model": model,
                    "downsample": False,
                    "seed_id": 0,
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                    "rmse": rmse,
                    "mae": mae,
                })


for seed_id in range(n_seeds):
    for subset_value, category, fold_id, X_train, y_train, X_test, y_test in soak_obj.split(X, y, subset_vec):
        for model in model_list:
            for n in [((soak_obj.n_splits - 1)/soak_obj.n_splits) * np.sum(subset_vec == subset_value), np.sum(subset_vec != subset_value)]:
                try:
                    if int(n) < len(y_train) - 1:
                        idx = np.random.choice(len(y_train), size=int(n), replace=False)
                        y_pred, (rmse, mae) = soak_obj.model_eval(X_train[idx], y_train[idx], X_test, y_test, model)
                        np.savez_compressed(f"{predictions_path}/{subset_value}.{category}.{fold_id}.{model}.{True}.{seed_id + 1}.{len(idx)}.npz",y=y)
                        results.append({
                                    "subset": subset_value,
                                    "category": category,
                                    "fold_id": fold_id,
                                    "model": model,
                                    "downsample": True,
                                    "seed_id": seed_id + 1,
                                    "train_size": len(idx),
                                    "test_size": len(y_test),
                                    "rmse": rmse,
                                    "mae": mae,
                                })
                except:
                    continue

# --- Save CV results ---
results_path = f"results/{dataset}/"
os.makedirs(results_path, exist_ok=True)
pd.DataFrame(results).to_csv(f"{results_path}/{subset_col}.csv", index=False)