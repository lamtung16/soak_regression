import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.cm as cm

def soak_plot_multiple_models(results_df, models = None, figsize = (16, 5)):
    if models == None:
        models = sorted(results_df["model"].unique())
    df = results_df[(results_df['seed_id'] == 0) & (results_df['model'].isin(models))].copy()
    df = df.sort_values("model")
    subsets = df["subset"].unique()
    categories = ["all", "same", "other"]

    # Aggregate mean and std
    df_mean = (
        df.groupby(["subset", "category", "model"], as_index=False)
          .agg(
              rmse_mean=("rmse", "mean"),
              rmse_sd=("rmse", "std"),
              mae_mean=("mae", "mean"),
              mae_sd=("mae", "std")
          )
    )

    # Colors
    cmap = cm.get_cmap("tab10")
    model_colors = {model: cmap(i % 10) for i, model in enumerate(models)}

    fig_dict = {}

    for subset in subsets:
        df_subset = df[df["subset"] == subset]
        df_mean_subset = df_mean[df_mean["subset"] == subset]
        n_obs = np.max(df_subset['test_size'])*np.max(df_subset['fold_id'])
        fig, axes = plt.subplots(len(categories), 4, figsize=figsize, sharex=False)

        # x-axis limits
        rmse_min = df_mean_subset["rmse_mean"].min() - 2 * df_mean_subset["rmse_sd"].max()
        rmse_max = df_mean_subset["rmse_mean"].max() + 2 * df_mean_subset["rmse_sd"].max()
        mae_min = df_mean_subset["mae_mean"].min() - 2 * df_mean_subset["mae_sd"].max()
        mae_max = df_mean_subset["mae_mean"].max() + 2 * df_mean_subset["mae_sd"].max()

        for i, category in enumerate(categories):
            data = df_subset[df_subset["category"] == category]
            data_mean = df_mean_subset[df_mean_subset["category"] == category]

            models_cat = sorted(data["model"].unique())
            model_to_y = {m: y for y, m in enumerate(models_cat)}

            for j, metric in enumerate(["rmse", "rmse", "mae", "mae"]):
                ax = axes[i, j]
                ax.grid(alpha=0.1)
                ax.set_ylim(-0.5, len(models_cat) - 0.5)
                ax.set_yticks([])

                # Select axis range
                if metric == "rmse":
                    ax.set_xlim(rmse_min, rmse_max)
                else:
                    ax.set_xlim(mae_min, mae_max)

                if j == 0:
                    ax.set_ylabel(category, rotation=0, labelpad=15, va="center", fontweight="bold")
                else:
                    ax.set_ylabel("")

                # x-axis label only on last row
                if i == len(categories) - 1:
                    ax.set_xlabel(f"{metric.upper()}")

                # Mean + SD (even columns 0,2)
                if j % 2 == 0:
                    for _, row in data_mean.iterrows():
                        color = model_colors[row["model"]]
                        ax.errorbar(
                            row[f"{metric}_mean"],
                            model_to_y[row["model"]],
                            xerr=row[f"{metric}_sd"],
                            fmt="o",
                            color=color,
                            capsize=2,
                            markersize=5
                        )
                # Raw points (odd columns 1,3)
                else:
                    for model in models_cat:
                        vals = data.loc[data["model"] == model, f"{metric}"].to_numpy()
                        ax.scatter(
                            vals,
                            np.ones_like(vals) * model_to_y[model],
                            facecolor=model_colors[model],
                            edgecolor=model_colors[model],
                            s=20
                        )
        # Legend and title
        handles = [Line2D([], [], marker="o", linestyle="", markersize=5, color=color, markerfacecolor=color, label=m) for m, color in reversed(list(model_colors.items()))]
        fig.legend(handles=handles, loc="upper right", fontsize=7)
        fig.suptitle(f"Subset: {subset} (n={n_obs})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig_dict[subset] = fig
    return fig_dict



def soak_plot_one_model(results_df, subset_value, model, metric, figsize=(6, 2.5)):
    _, counts = np.unique(results_df['downsample'], return_counts=True)
    df = results_df[(results_df['subset'] == subset_value) & (results_df['model'] == model)].copy()
    df = df.groupby(['subset', 'category', 'model', 'downsample', 'train_size']).agg(
        avg=(f'{metric}', 'mean'),
        sd=(f'{metric}', lambda x: x.std(ddof=0)),
        test_size=('test_size', 'min'),
    ).reset_index()
    agg_df = (
    df[df["downsample"] == False]
    .groupby(["category", "model", "subset"], as_index=False)
    .agg({
        "train_size": "min",
        "avg": "mean",
        "sd": "mean",
        "test_size": "min"
    })
    )
    agg_df["downsample"] = False
    final_df = pd.concat([agg_df, df[df["downsample"] == True].copy()], ignore_index=True).sort_values(['category', 'train_size'])

    plt.figure(figsize=figsize)
    plt.ylim(-0.5, final_df.shape[0] - 0.5)
    for _, row in final_df.iterrows():
        _x = row['avg']
        _y = f"{row['category']}.{row['train_size']}"
        _xerr = row['sd']
        _color = 'red' if row['downsample'] else 'black'
        plt.errorbar(x = _x, y = _y, xerr=_xerr, fmt='o', color=_color)
        plt.text(_x, _y, f"{_x:.4f}±{_xerr:.4f}", ha='center', va='bottom', color=_color)
    plt.plot([], [], 'o', color='red',   label='reduced')
    plt.plot([], [], 'o', color='black', label='full')
    plt.yticks(final_df.apply(lambda r: f"{r['category']}.{r['train_size']}", axis=1), fontsize=9)
    plt.xlabel(f"{metric.upper()}")
    plt.title(f"subset: {final_df['subset'][0]} || "
              f"model: {final_df['model'][0]} || "
              f"{results_df['fold_id'].max()} folds || "
              f"{int(counts.max()/counts.min())} random seeds", 
              fontsize=10, x=0.35)
    plt.legend(bbox_to_anchor=(1, 1.2))
    plt.grid(alpha=0.5)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.show()