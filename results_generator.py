from soak import soak
import pandas as pd
import sys
import os

# --- read dataset name ---
params_df = pd.read_csv("params.csv")
dataset = params_df.iloc[int(sys.argv[1])]['dataset']

# --- Load dataset ---
df = pd.read_csv(f"data/{dataset}.csv")

# --- Initialize soak object ---
soak_obj = soak(df)

# --- Analyze all subsets ---
subset_list = sorted(df['subset'].unique().tolist())
soak_obj.analyze(subset_list)

# --- Save CV results ---
results_path = f"results/{dataset}.csv"
os.makedirs("results", exist_ok=True)
soak_obj.results_df.to_csv(results_path, index=False)

# --- Generate figures ---
figs = soak_obj.plot_results()

# --- Save figures manually ---
figures_dir = f"figures/{dataset}"
os.makedirs(figures_dir, exist_ok=True)
for subset_name, fig in figs.items():
    fig.savefig(f"{figures_dir}/{subset_name}.png")