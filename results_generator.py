import polars as pl
import lzma
import sys
import os
from soak import soak

# --- Read dataset name from params.csv ---
params_df = pl.read_csv("params.csv")
idx = int(sys.argv[1])
dataset = params_df[idx, "dataset"]
subset_col = params_df[idx, "subset_col"]
target_col = params_df[idx, "target_col"]

# --- Load dataset efficiently from .xz ---
file_path = f"data/{dataset}.csv.xz"
with lzma.open(file_path, mode="rb") as f:
    df = pl.read_csv(f, encoding="latin1")

# --- Initialize soak object ---
soak_obj = soak(df, subset_col, target_col, models=['featureless', 'linear'])

# --- Analyze all subsets ---
subset_list = sorted(df[subset_col].unique().to_list())
soak_obj.analyze(subset_list)

# --- Save CV results ---
results_path = f"results/{dataset}"
os.makedirs(results_path, exist_ok=True)
soak_obj.results_df.write_csv(f"{results_path}/{subset_col}.csv")

# --- Generate figures ---
figs = soak_obj.plot_results()

# --- Save figures manually ---
figures_dir = f"figures/{dataset}/{subset_col}"
os.makedirs(figures_dir, exist_ok=True)
for subset_name, fig in figs.items():
    fig.savefig(f"{figures_dir}/{subset_name}.png")