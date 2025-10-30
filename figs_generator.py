import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- read dataset name ---
params_df = pd.read_csv("params.csv")
dataset = params_df.iloc[int(sys.argv[1])]['dataset']

# --- Load results and data ---
df = pd.read_csv(f"results/{dataset}.csv")
df_data = pd.read_csv(f"data/{dataset}.csv")

# --- Compute number of observations per subset from data/{dataset}.csv ---
# Ensure consistent subset names (strip spaces)
if 'subset' not in df_data.columns:
    raise KeyError(f"'subset' column not found in data/{dataset}.csv")

subset_counts = df_data['subset'].astype(str).str.strip().value_counts().to_dict()

# --- Prepare results dataframe ---
df['subset'] = df['subset'].astype(str).str.strip()
df['category'] = df['category'].astype(str).str.strip()
df['model'] = df['model'].astype(str).str.strip()

# Take log of mse
df['log_mse'] = np.log10(df['mse'])

subsets = df['subset'].unique()
categories = df['category'].unique()
models = df['model'].unique()

# Create output directory for this dataset
output_dir = f"figures/{dataset}"
os.makedirs(output_dir, exist_ok=True)

# --- Aggregate by subset, category, model ---
df_mean = df.groupby(['subset', 'category', 'model'], sort=False).agg(
    mse_mean=('log_mse', 'mean'),
    mse_sd=('log_mse', 'std')
).reset_index()

# --- Generate one figure per subset ---
for subset in subsets:
    n_obs = subset_counts.get(subset, 0)  # number of observations from data/{dataset}.csv

    fig, axes = plt.subplots(
        len(categories),
        2,
        figsize=(10, 1.1 * len(categories)),
        sharex=True
    )

    if len(categories) == 1:
        axes = np.array([axes])  # ensure consistent indexing

    for i, category in enumerate(categories):
        for j in range(2):
            ax_err = axes[i, j]
            ax_err.grid(alpha=0.2)

            data_mean = df_mean[(df_mean['category'] == category) & (df_mean['subset'] == subset)]
            data = df[(df['category'] == category) & (df['subset'] == subset)]

            model_names = data['model'].unique()
            model_to_y = {m: yk for yk, m in enumerate(model_names)}

            ax_err.set_ylim(-0.5, len(models) - 0.5)
            if i == len(categories) - 1:
                ax_err.set_xlabel('log(MSE)')
            if j == 0:
                ax_err.set_ylabel(category, rotation=0, labelpad=15, va='center', fontweight='bold')

            ax_err.set_yticklabels([])

            # --- Left column: means with error bars ---
            if j == 0:
                for k, row in enumerate(data_mean.itertuples()):
                    if row.model == 'featureless':
                        ax_err.errorbar(row.mse_mean, k, xerr=row.mse_sd, fmt='o',
                                        color='black', markerfacecolor='white', capsize=2, markersize=5)
                    elif row.model == 'cv_gam':
                        ax_err.errorbar(row.mse_mean, k, xerr=row.mse_sd, fmt='o',
                                        color='black', markerfacecolor='black', capsize=2, markersize=5)
            # --- Right column: all points ---
            else:
                for model in model_names:
                    vals = data.loc[data['model'] == model, 'log_mse']
                    y_pos = model_to_y[model]
                    if model == 'featureless':
                        ax_err.scatter(vals, np.ones_like(vals)*y_pos,
                                       facecolor='white', edgecolor='black', s=20)
                    elif model == 'cv_gam':
                        ax_err.scatter(vals, np.ones_like(vals)*y_pos,
                                       facecolor='black', edgecolor='black', s=20)

    # Global legend and title
    handles = [
        plt.Line2D([], [], marker='o', color='black', markerfacecolor='black',
                   linestyle='', markersize=5, label='cv_gam'),
        plt.Line2D([], [], marker='o', color='black', markerfacecolor='white',
                   linestyle='', markersize=5, label='featureless')
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=8)
    fig.suptitle(f'dataset: {dataset} | test subset: {subset} (n = {n_obs})',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plt.savefig(f"{output_dir}/{subset}.png")
    plt.close(fig)