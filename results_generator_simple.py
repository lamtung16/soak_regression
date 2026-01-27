import sys
import os
import numpy as np
import pandas as pd
from soak_low_level import SOAKFold
import time

# --- Read dataset name from params.csv ---
params = pd.read_csv("params.csv")
row_idx = int(sys.argv[1])
row = params.iloc[row_idx]
dataset = row["dataset"]
subset_col = row["subset_col"]
target_col = row["target_col"]
log_target = bool(row["log_target"])
n_folds = 10
n_seeds = 10

# Load data
data = np.genfromtxt(f'data/{dataset}.csv.xz', delimiter=',', dtype=None, names=True, encoding=None, invalid_raise=False)
X = np.column_stack([data[name] for name in data.dtype.names if name.startswith('X_')])
y = data[target_col]
subset_vec = data[subset_col]

# model list
model_list = ['featureless', 'linear', 'tree']

# replace orignal label by its log scale
if log_target:
    y = np.log(y)
y = (y - np.mean(y)) / np.std(y)

# --- Initialize soak object ---
soak_obj = SOAKFold(n_splits=n_folds)

# --- Analyze all subsets ---
results = []
for subset_value, category, fold_id, train_idx, test_idx in soak_obj.split_idx(subset_vec):
    for model in model_list:
        start_time = time.time()
        y_pred, (rmse, mae) = soak_obj.model_eval(X[train_idx], y[train_idx], X[test_idx], y[test_idx], model)
        elapsed_time = time.time() - start_time
        results.append({
                        "subset": subset_value,
                        "category": category,
                        "fold_id": fold_id,
                        "model": model,
                        "downsample": False,
                        "seed_id": 0,
                        "train_size": len(train_idx),
                        "test_size": len(test_idx),
                        "rmse": rmse,
                        "mae": mae,
                        "train_time": elapsed_time,
                    })


for seed_id in range(n_seeds):
    for subset_value, category, fold_id, train_idx, test_idx in soak_obj.split_idx(subset_vec):
        for model in model_list:
            rng = np.random.default_rng(seed_id)
            shuffled_idx = train_idx.copy()
            rng.shuffle(shuffled_idx)
            n = min(np.floor(((n_folds-1)/n_folds) * np.sum(subset_vec == subset_value)), np.sum(subset_vec != subset_value))
            if int(n) < len(train_idx) - 1:
                idx = shuffled_idx[:int(n)]
                start_time = time.time()
                y_pred, (rmse, mae) = soak_obj.model_eval(X[idx], y[idx], X[test_idx], y[test_idx], model)
                elapsed_time = time.time() - start_time
                results.append({
                                "subset": subset_value,
                                "category": category,
                                "fold_id": fold_id,
                                "model": model,
                                "downsample": True,
                                "seed_id": seed_id + 1,
                                "train_size": len(idx),
                                "test_size": len(test_idx),
                                "rmse": rmse,
                                "mae": mae,
                                "train_time": elapsed_time,
                            })

# --- Save CV results ---
results_path = f"results/{dataset}/"
os.makedirs(results_path, exist_ok=True)
pd.DataFrame(results).to_csv(f"{results_path}/{subset_col}.csv", index=False)