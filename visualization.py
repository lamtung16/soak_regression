import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from scipy.stats import ttest_ind


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



def soak_plot_one_model(results_df, subset_value, model, metric='rmse', figsize=(6, 2.5)):
    _, counts = np.unique(results_df['downsample'], return_counts=True)
    df = results_df[(results_df['subset'] == subset_value) & (results_df['model'] == model)].copy()
    def summarize(df, group_cols, extra_aggs=None):
        aggs = {
            'avg': (metric, 'mean'),
            'sd': (metric, lambda x: x.std()),
            'test_size': ('test_size', 'min'),
        }
        if extra_aggs:
            aggs.update(extra_aggs)

        return (
            df.groupby(group_cols)
            .agg(**aggs)
            .reset_index()
        )

    df_ds = summarize(
        df.query('downsample'),
        ['subset', 'category', 'model', 'downsample', 'train_size']
    )

    df_not_ds = summarize(
        df.query('~downsample'),
        ['subset', 'category', 'model', 'downsample'],
        extra_aggs={'train_size': ('train_size', 'min')}
    )

    final_df = (
        pd.concat([df_not_ds, df_ds], ignore_index=True)
        .sort_values(['category', 'train_size'])
    )

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
    plt.xlabel(f"{metric.upper()} mean ± 1sd")
    plt.title(f"subset: {final_df['subset'][0]} || "
              f"model: {final_df['model'][0]} || "
              f"{results_df['fold_id'].max()} folds || "
              f"{int(counts.max()/counts.min())} random seeds", 
              fontsize=10, x=0.35)
    plt.legend(bbox_to_anchor=(1, 1.2))
    plt.grid(alpha=0.5)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.show()




def soak_plot_one_model_extend(results_df, subset_value, model, metric="rmse", figsize=(6, 2.5)):
    # filter
    category_order = ["all", "all-same", "same", "other-same", "other"]
    df = results_df[
        (results_df["subset"] == subset_value) &
        (results_df["model"] == model) &
        (~results_df["downsample"])
    ].copy()
    base_cats = ["other", "same", "all"]

    # base summary
    summary = (
        df.groupby("category", observed=False)[metric]
        .agg(mean="mean", std="std")
        .reindex(base_cats)
        .reset_index()
    )

    # calculate p-values for combined categories
    def pval(cat1, cat2):
        x = df.loc[df["category"] == cat1, metric]
        y = df.loc[df["category"] == cat2, metric]
        t_stat, p = ttest_ind(x, y, equal_var=False)  # Welch's t-test
        return p

    mean_same = df.loc[df["category"] == "same", metric].mean()
    mean_other = df.loc[df["category"] == "other", metric].mean()
    mean_all = df.loc[df["category"] == "all", metric].mean()

    combined = pd.DataFrame({
        "category": ["other-same", "all-same"],
        "mean": [
            (mean_same + mean_other) / 2,
            (mean_same + mean_all) / 2,
        ],
        "std": [
            abs(mean_other - mean_same) / 2,
            abs(mean_all - mean_same) / 2,
        ],
        "p_value": [
            pval("other", "same"),
            pval("all", "same")
        ]
    })

    # merge base summary (no p-values for single categories)
    summary["p_value"] = np.nan
    summary = pd.concat([summary, combined], ignore_index=True)
    summary = (
        summary.assign(category=lambda x: pd.Categorical(x["category"], category_order, ordered=True))
        .sort_values("category")
        .reset_index(drop=True)
    )

    # y positions
    y_pos = {cat: i for i, cat in enumerate(category_order)}

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in summary.iterrows():
        y = y_pos[row["category"]]
        mean = row["mean"]
        sd = row["std"]
        color = 'black' if i % 2 == 0 else 'grey'
        text = f"{mean:.5f} ± {sd:.5f}" if i % 2 == 0 else f"P = {row['p_value']:.4f}"
        marker_size = 4 if i % 2 == 0 else 0
        ax.errorbar(mean, y, xerr=sd, fmt="o", color=color, markersize=marker_size)
        ax.text(mean, y + 0.15, text, ha="center", va="bottom", fontsize=8)

    # y-axis formatting
    ax.set_yticks([y_pos[c] for c in category_order])
    ax.set_yticklabels(category_order, fontsize=9)
    ax.set_ylim(-0.5, len(category_order) - 0.2)

    # labels & title
    ax.set_xlabel(metric.upper(), fontsize=8)
    ax.set_title(f"subset: {subset_value} | model: {model} | {df['fold_id'].nunique()} folds", fontsize=9)
    ax.grid(alpha=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def process_df(results_df, subset_value, model, metric, downsample=False):
    if downsample == False:
        df = results_df[
        (results_df["subset"] == subset_value) &
        (results_df["model"] == model) &
        (results_df["downsample"]==downsample)
        ].copy()
    else:
        df1 = results_df[
        (results_df["subset"] == subset_value) &
        (results_df["model"] == model) &
        (results_df["downsample"]==downsample)
        ]
        df2 = results_df[
        (results_df["subset"] == subset_value) &
        (results_df["model"] == model) &
        (results_df["category"]== [item for item in ["other", "same", "all"] if item not in set(df1['category'])][0])
        ]
        df2 = pd.concat([df2] * df1['seed_id'].unique().__len__(), ignore_index=True)
        df = pd.concat([df1, df2], ignore_index=True)

    base_cats = ["other", "same", "all"]

    # base summary
    summary = (
        df.groupby("category", observed=False)[metric]
        .agg(mean="mean", std=lambda x: x.std(ddof=0))
        .reindex(base_cats)
        .reset_index()
    )

    # calculate p-values for combined categories
    def pval(cat1, cat2):
        x = df.loc[df["category"] == cat1, metric]
        y = df.loc[df["category"] == cat2, metric]
        t_stat, p = ttest_ind(x, y, equal_var=False)  # Welch's t-test
        return p
    
    mean_same = df.loc[df["category"] == "same", metric].mean()
    mean_other = df.loc[df["category"] == "other", metric].mean()
    mean_all = df.loc[df["category"] == "all", metric].mean()

    combined = pd.DataFrame({
        "category": ["other-same", "all-same"],
        "mean": [
            (mean_same + mean_other) / 2,
            (mean_same + mean_all) / 2,
        ],
        "std": [
            abs(mean_other - mean_same) / 2,
            abs(mean_all - mean_same) / 2,
        ],
        "p_value": [
            pval("other", "same"),
            pval("all", "same")
        ]
    })

    # merge base summary (no p-values for single categories)
    category_order = ["all", "all-same", "same", "other-same", "other"]
    summary["p_value"] = np.nan
    summary = pd.concat([summary, combined], ignore_index=True)
    summary = (
        summary.assign(category=lambda x: pd.Categorical(x["category"], category_order, ordered=True))
        .sort_values("category")
        .reset_index(drop=True)
    )
    return summary


def soak_plot_matrix(results_df, models, subset_values, metric='rmse', subplot_size=(5, 2), extend=(0.01, 0.01)):   
    sizes = subplot_size
    category_order = ["all", "all-same", "same", "other-same", "other"]
    y_pos = {cat: i for i, cat in enumerate(category_order)}

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=len(subset_values),
        figsize=(sizes[0]*len(subset_values), sizes[1]*len(models)),
        sharex=True,
        sharey=True
    )

    for row_idx, model in enumerate(models):
        for col_idx, subset_value in enumerate(subset_values):
            ax = axes[row_idx, col_idx]

            summary = process_df(results_df, subset_value, model, metric)

            for i, row in summary.iterrows():
                y = y_pos[row["category"]]
                mean = row["mean"]
                sd = row["std"]

                color = "black" if i % 2 == 0 else "grey"
                text = (f"{mean:.4f} ± {2*sd:.4f}" if i % 2 == 0 else f"P = {row['p_value']:.5f}")
                marker_size = 4.5 if i % 2 == 0 else 0
                _sd = 2*sd if i % 2 == 0 else sd
                ax.errorbar(mean, y, xerr=_sd, fmt="o", color=color, markersize=marker_size, linewidth=3)
                ax.text(mean, y + 0.12, text, ha="center", va="bottom", fontsize=10)
                left, right = ax.get_xlim()
                ax.set_xlim(left - extend[0], right + extend[1])

            # --- LEFT: category labels on every subplot ---
            ax.set_yticks([y_pos[c] for c in category_order])
            ax.set_yticklabels(category_order, fontsize=12)
            ax.set_ylim(-0.5, len(category_order) - 0.2)

            ax.grid(alpha=0.5)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.tick_params(axis="x", labelsize=12)

            # --- TOP: subset titles ---
            if row_idx == 0:
                ax.set_title(f"{subset_value} ({results_df[results_df['subset']==subset_value].iloc[0]['test_size']*len(set(results_df['fold_id']))} rows)", fontsize=12)

            # --- RIGHT: model labels ---
            if col_idx == len(subset_values) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"model: {model}", fontsize=12, rotation=270, labelpad=15)

    fig.supxlabel(f"{metric.upper()} (mean ± 2sd) over {set(results_df['fold_id']).__len__()} folds", fontsize=13)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    plt.close(fig)
    return fig


def soak_plot_matrix_downsample(results_df, models, subset_values, metric='rmse', subplot_size=(5, 4), extend=(0.01, 0.01)):   
    n_random_seeds = len(set(results_df['seed_id']))-1
    n_folds = set(results_df['fold_id']).__len__()

    sizes = subplot_size
    category_order = ["all", "all-same", "same", "other-same", "other"]
    y_pos = {cat: i for i, cat in enumerate(category_order)}

    fig, axes = plt.subplots(
        nrows= 2*len(models),
        ncols=len(subset_values),
        figsize=(sizes[0]*len(subset_values), sizes[1]*len(models)),
        sharex='col',
        sharey=True
    )
    for row_idx, model in enumerate(models * 2):
        for col_idx, subset_value in enumerate(subset_values):
            ax = axes[row_idx, col_idx]
            if row_idx < len(models):
                summary = process_df(results_df, subset_value, model, metric, False)
            else:
                summary = process_df(results_df, subset_value, model, metric, True)
            for i, row in summary.iterrows():
                y = y_pos[row["category"]]
                mean = row["mean"]
                sd = row["std"]
                if row_idx < len(models):
                    color = "black" if row['category'] in ["other", "same", "all"] else "grey"
                else:
                    color = "red" if row['category'] in ["other", "same", "all"] else "pink"
                text = (f"{mean:.4f} ± {2*sd:.4f}" if row['category'] in ["other", "same", "all"] else "P < 0.0001" if row['p_value'] < 0.0001 else f"P = {row['p_value']:.4f}")
                marker_size = 4.5 if row['category'] in ["other", "same", "all"] else 0
                _sd = 2*sd if row['category'] in ["other", "same", "all"] else sd
                ax.errorbar(mean, y, xerr=_sd, fmt="o", color=color, markersize=marker_size, linewidth=3)
                ax.text(mean, y + 0.12, text, ha="center", va="bottom", fontsize=10)
                left, right = ax.get_xlim()
                ax.set_xlim(left - extend[0], right + extend[1])

            # --- LEFT: category labels on every subplot ---
            ax.set_yticks([y_pos[c] for c in category_order])
            ax.set_yticklabels(category_order, fontsize=12)
            ax.set_ylim(-0.5, len(category_order) - 0.2)

            ax.grid(alpha=0.5)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.tick_params(axis="x", labelsize=12)

            # --- TOP: subset titles ---
            if row_idx == 0:
                ax.set_title(f"{subset_value} ({results_df[results_df['subset']==subset_value].iloc[0]['test_size']*len(set(results_df['fold_id']))} rows)", fontsize=12)

            # --- RIGHT: model labels ---
            if col_idx == len(subset_values) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"{model}", fontsize=12, rotation=270, labelpad=15)

    fig.supxlabel(f"{metric.upper()} (mean ± 2sd) over {n_folds} folds using {n_random_seeds} random seeds for downsample", fontsize=13)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    plt.plot([], [], 'o', color='black', label='full')
    plt.plot([], [], 'o', color='red',   label='downsample')
    plt.legend()
    plt.close(fig)
    return fig


def soak_plot_one_model_extend_downsample(results_df, subset_value, model, metric="rmse", figsize=(6, 2.5)):
    df = results_df[
        (results_df["subset"] == subset_value) &
        (results_df["model"] == model)
    ].copy()
    df["category"] = (
        df["category"]
        + "."
        + df["downsample"].map({False: "full", True: "ds"})
    )

    cats = set(df["category"].unique())
    has_other_ds = "other.ds" in cats

    # middle element
    if not has_other_ds and "other.full" in cats:
        mid = "other.full"
    else:
        mid = "same.full"

    # build sorted list
    sorted_cats = [
        "same.ds" if mid == "other.full" else "other.ds",
        "same.full" if mid == "other.full" else "other.full",
        mid,
        "all.full",
        "all.ds",
    ]

    # keep only existing categories (safety)
    sorted_cats = [c for c in sorted_cats if c in cats]

    # base summary
    summary = (
        df.groupby("category", observed=False)[metric]
        .agg(mean="mean", std="std")
        .reindex(sorted_cats)
        .reset_index()
    )

    # calculate p-values for combined categories
    def pval(cat1, cat2):
        x = df.loc[df["category"] == cat1, metric]
        y = df.loc[df["category"] == cat2, metric]
        t_stat, p = ttest_ind(x, y, equal_var=False)  # Welch's t-test
        return p

    combined = pd.DataFrame({
        "category": [f"{sorted_cats[1]}-{sorted_cats[0]}", f"{sorted_cats[1]}-{sorted_cats[2]}", f"{sorted_cats[3]}-{sorted_cats[2]}", f"{sorted_cats[3]}-{sorted_cats[4]}"],
        "mean": [
            (summary.iloc[1]['mean'] + summary.iloc[0]['mean']) / 2,
            (summary.iloc[1]['mean'] + summary.iloc[2]['mean']) / 2,
            (summary.iloc[3]['mean'] + summary.iloc[2]['mean']) / 2,
            (summary.iloc[3]['mean'] + summary.iloc[4]['mean']) / 2
        ],
        "std": [
            abs(summary.iloc[1]['mean'] - summary.iloc[0]['mean']) / 2,
            abs(summary.iloc[1]['mean'] - summary.iloc[2]['mean']) / 2,
            abs(summary.iloc[3]['mean'] - summary.iloc[2]['mean']) / 2,
            abs(summary.iloc[3]['mean'] - summary.iloc[4]['mean']) / 2
        ],
        "p_value": [
            pval(sorted_cats[1], sorted_cats[0]),
            pval(sorted_cats[1], sorted_cats[2]),
            pval(sorted_cats[3], sorted_cats[2]),
            pval(sorted_cats[3], sorted_cats[4]),
        ]
    })

    n = len(sorted_cats) + len(combined['category'].to_list())
    category_order = [None] * n

    category_order[::2] = sorted_cats
    category_order[1::2] = combined['category'].to_list()

    summary["p_value"] = np.nan
    final_df = pd.concat([summary, combined], ignore_index=True)
    final_df = (
        final_df.assign(category=lambda x: pd.Categorical(x["category"], category_order, ordered=True))
        .sort_values("category")
        .reset_index(drop=True)
    )

    # y positions
    y_pos = {cat: i for i, cat in enumerate(category_order)}

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in final_df.iterrows():
        y = y_pos[row["category"]]
        mean = row["mean"]
        sd = row["std"]
        color = 'black' if i % 2 == 0 else 'grey'
        text = f"{mean:.5f} ± {sd:.5f}" if i % 2 == 0 else f"P = {row['p_value']:.4f}"
        marker_size = 4 if i % 2 == 0 else 0
        ax.errorbar(mean, y, xerr=sd, fmt="o", color=color, markersize=marker_size)
        ax.text(mean, y + 0.15, text, ha="center", va="bottom", fontsize=8)

    # y-axis formatting
    ax.set_yticks([y_pos[c] for c in category_order])
    ax.set_yticklabels(category_order, fontsize=9)
    ax.set_ylim(-0.5, len(category_order) - 0.2)

    # labels & title
    ax.set_xlabel(f"{metric.upper()} (mean ± 2sd)", fontsize=8)
    ax.set_title(f"subset: {subset_value} | model: {model} | {df['fold_id'].nunique()} folds", fontsize=9)
    ax.grid(alpha=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def soak_plot_one_model_extend_downsample_2(results_df, subset_value, model, metric="rmse", figsize=(6, 2.5)):
    df = results_df[
        (results_df["subset"] == subset_value) &
        (results_df["model"] == model)
    ].copy()
    df["category"] = (
        df["category"]
        + "."
        + df["downsample"].map({False: "full", True: "ds"})
    )
    cats = set(df["category"].unique())
    has_other_ds = "other.ds" in cats

    # middle element
    if not has_other_ds and "other.full" in cats:
        mid = "other.full"
    else:
        mid = "same.full"

    # build sorted list
    sorted_cats = [
        "same.ds" if mid == "other.full" else "other.ds",
        "same.full" if mid == "other.full" else "other.full",
        mid,
        "all.full",
        "all.ds",
    ]

    # keep only existing categories (safety)
    sorted_cats = [c for c in sorted_cats if c in cats]

    # base summary
    summary = (
        df.groupby("category", observed=False)
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            train_size=("train_size", "min"),
        )
        .reindex(sorted_cats)
        .reset_index())

    # calculate p-values for combined categories
    def pval(cat1, cat2):
        x = df.loc[df["category"] == cat1, metric]
        y = df.loc[df["category"] == cat2, metric]
        t_stat, p = ttest_ind(x, y, equal_var=False)  # Welch's t-test
        return p

    combined = pd.DataFrame({
        "category": [f"{sorted_cats[2]}-{sorted_cats[0]}", f"{sorted_cats[1]}-{sorted_cats[0]}", f"{sorted_cats[1]}-{sorted_cats[2]}", f"{sorted_cats[3]}-{sorted_cats[2]}", f"{sorted_cats[3]}-{sorted_cats[4]}", f"{sorted_cats[2]}-{sorted_cats[4]}"],
        "mean": [
            (summary.iloc[2]['mean'] + summary.iloc[0]['mean']) / 2,
            (summary.iloc[1]['mean'] + summary.iloc[0]['mean']) / 2,
            (summary.iloc[1]['mean'] + summary.iloc[2]['mean']) / 2,
            (summary.iloc[3]['mean'] + summary.iloc[2]['mean']) / 2,
            (summary.iloc[3]['mean'] + summary.iloc[4]['mean']) / 2,
            (summary.iloc[2]['mean'] + summary.iloc[4]['mean']) / 2
        ],
        "std": [
            abs(summary.iloc[2]['mean'] - summary.iloc[0]['mean']) / 2,
            abs(summary.iloc[1]['mean'] - summary.iloc[0]['mean']) / 2,
            abs(summary.iloc[1]['mean'] - summary.iloc[2]['mean']) / 2,
            abs(summary.iloc[3]['mean'] - summary.iloc[2]['mean']) / 2,
            abs(summary.iloc[3]['mean'] - summary.iloc[4]['mean']) / 2,
            abs(summary.iloc[2]['mean'] - summary.iloc[4]['mean']) / 2,
        ],
        "p_value": [
            pval(sorted_cats[2], sorted_cats[0]),
            pval(sorted_cats[1], sorted_cats[0]),
            pval(sorted_cats[1], sorted_cats[2]),
            pval(sorted_cats[3], sorted_cats[2]),
            pval(sorted_cats[3], sorted_cats[4]),
            pval(sorted_cats[2], sorted_cats[4])
        ]
    })

    n = len(sorted_cats) + len(combined['category'].to_list())
    category_order = [None] * n

    category_order[::2] = combined['category'].to_list()
    category_order[1::2] = sorted_cats

    combined["train_size"] = np.nan
    summary["p_value"] = np.nan
    final_df = pd.concat([summary, combined], ignore_index=True)
    final_df = (
        final_df.assign(category=lambda x: pd.Categorical(x["category"], category_order, ordered=True))
        .sort_values("category")
        .reset_index(drop=True)
    )

    final_df["category"] = final_df.apply(
        lambda row: f"{row['category']}.{int(row['train_size'])}" 
        if pd.notnull(row['train_size']) 
        else row['category'], 
        axis=1
    )

    # final_df["category"] = final_df["category"].str.replace(r'\.full\.|\.ds\.', '.', regex=True)

    # y positions
    category_order = final_df["category"]
    y_pos = {cat: i for i, cat in enumerate(category_order)}

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in final_df.iterrows():
        y = y_pos[row["category"]]
        mean = row["mean"]
        sd = row["std"]
        is_acc_row = "-" not in str(row["category"])
        color = 'black' if is_acc_row else 'grey'
        text = f"{mean:.5f} ± {sd:.5f}" if is_acc_row else f"P = {row['p_value']:.4f}"
        marker_size = 4 if is_acc_row else 0
        ax.errorbar(mean, y, xerr=sd, fmt="o", color=color, markersize=marker_size)
        ax.text(mean, y + 0.15, text, ha="center", va="bottom", fontsize=8)

    # y-axis formatting
    ax.set_yticks([y_pos[c] for c in category_order])
    ax.set_yticklabels(category_order, fontsize=9)
    ax.set_ylim(-0.5, len(category_order) - 0.2)

    # labels & title
    ax.set_xlabel(f"{metric.upper()} (mean ± 2sd)", fontsize=8)
    ax.set_title(f"subset: {subset_value} | model: {model} | {df['fold_id'].nunique()} folds", fontsize=9)
    ax.grid(alpha=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def soak_plot_one_model_extend_downsample_3(results_df, subset_value, model, metric="rmse", figsize=(6, 2.5)):
    df = results_df[
        (results_df["subset"] == subset_value) &
        (results_df["model"] == model)
    ].copy()
    df["category"] = (
        df["category"]
        + "."
        + df["downsample"].map({False: "full", True: "ds"})
    )
    cats = set(df["category"].unique())
    
    sorted_cats = [
        "other.ds" if "other.ds" in cats else "other.full",
        "same.ds" if "same.ds" in cats else "other.full",
        "same.full",
        "all.full",
        "all.ds",
    ]

    # keep only existing categories (safety)
    sorted_cats = [c for c in sorted_cats if c in cats]

    # base summary
    summary = (
        df.groupby("category", observed=False)
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            train_size=("train_size", "min"),
        )
        .reindex(sorted_cats)
        .reset_index())

    # calculate p-values for combined categories
    def pval(cat1, cat2):
        x = df.loc[df["category"] == cat1, metric]
        y = df.loc[df["category"] == cat2, metric]
        t_stat, p = ttest_ind(x, y, equal_var=False)  # Welch's t-test
        return p

    combined = pd.DataFrame({
        "category": [f"{sorted_cats[2]}-{sorted_cats[0]}", f"{sorted_cats[1]}-{sorted_cats[0]}", f"{sorted_cats[1]}-{sorted_cats[2]}", f"{sorted_cats[3]}-{sorted_cats[2]}", f"{sorted_cats[3]}-{sorted_cats[4]}", f"{sorted_cats[2]}-{sorted_cats[4]}"],
        "mean": [
            (summary.iloc[2]['mean'] + summary.iloc[0]['mean']) / 2,
            (summary.iloc[1]['mean'] + summary.iloc[0]['mean']) / 2,
            (summary.iloc[1]['mean'] + summary.iloc[2]['mean']) / 2,
            (summary.iloc[3]['mean'] + summary.iloc[2]['mean']) / 2,
            (summary.iloc[3]['mean'] + summary.iloc[4]['mean']) / 2,
            (summary.iloc[2]['mean'] + summary.iloc[4]['mean']) / 2
        ],
        "std": [
            abs(summary.iloc[2]['mean'] - summary.iloc[0]['mean']) / 2,
            abs(summary.iloc[1]['mean'] - summary.iloc[0]['mean']) / 2,
            abs(summary.iloc[1]['mean'] - summary.iloc[2]['mean']) / 2,
            abs(summary.iloc[3]['mean'] - summary.iloc[2]['mean']) / 2,
            abs(summary.iloc[3]['mean'] - summary.iloc[4]['mean']) / 2,
            abs(summary.iloc[2]['mean'] - summary.iloc[4]['mean']) / 2,
        ],
        "p_value": [
            pval(sorted_cats[2], sorted_cats[0]),
            pval(sorted_cats[1], sorted_cats[0]),
            pval(sorted_cats[1], sorted_cats[2]),
            pval(sorted_cats[3], sorted_cats[2]),
            pval(sorted_cats[3], sorted_cats[4]),
            pval(sorted_cats[2], sorted_cats[4])
        ]
    })

    n = len(sorted_cats) + len(combined['category'].to_list())
    category_order = [None] * n

    category_order[::2] = combined['category'].to_list()
    category_order[1::2] = sorted_cats

    combined["train_size"] = np.nan
    summary["p_value"] = np.nan
    final_df = pd.concat([summary, combined], ignore_index=True)
    final_df = (
        final_df.assign(category=lambda x: pd.Categorical(x["category"], category_order, ordered=True))
        .sort_values("category")
        .reset_index(drop=True)
    )

    final_df["category"] = final_df.apply(
        lambda row: f"{row['category']}.{int(row['train_size'])}" 
        if pd.notnull(row['train_size']) 
        else row['category'], 
        axis=1
    )

    # final_df["category"] = final_df["category"].str.replace(r'\.full\.|\.ds\.', '.', regex=True)

    # y positions
    category_order = final_df["category"]
    y_pos = {cat: i for i, cat in enumerate(category_order)}

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in final_df.iterrows():
        y = y_pos[row["category"]]
        mean = row["mean"]
        sd = row["std"]
        is_acc_row = "-" not in str(row["category"])
        color = 'black' if is_acc_row else 'grey'
        text = f"{mean:.5f} ± {sd:.5f}" if is_acc_row else f"P = {row['p_value']:.4f}"
        marker_size = 4 if is_acc_row else 0
        ax.errorbar(mean, y, xerr=sd, fmt="o", color=color, markersize=marker_size)
        ax.text(mean, y + 0.15, text, ha="center", va="bottom", fontsize=8)

    # y-axis formatting
    ax.set_yticks([y_pos[c] for c in category_order])
    ax.set_yticklabels(category_order, fontsize=9)
    ax.set_ylim(-0.5, len(category_order) - 0.2)

    # labels & title
    ax.set_xlabel(f"{metric.upper()} (mean ± 2sd)", fontsize=8)
    ax.set_title(f"subset: {subset_value} | model: {model} | {df['fold_id'].nunique()} folds", fontsize=9)
    ax.grid(alpha=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def soak_plot_one_model_extend_downsample_4(results_df, subset_value, model, metric="rmse", figsize=(6, 2.5)):
    df = results_df[
        (results_df["subset"] == subset_value) &
        (results_df["model"] == model)
    ].copy()
    df["category"] = (
        df["category"]
        + "."
        + df["downsample"].map({False: "full", True: "ds"})
    )
    cats = set(df["category"].unique())
    
    if "same.ds" in cats:
        sorted_cats = ["other.full","same.full",  "same.ds",   "all.ds",   "all.full"]
    else:
        sorted_cats = ["other.ds",  "other.full", "same.full", "all.full", "all.ds"]

    # base summary
    summary = (
        df.groupby("category", observed=False)
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            train_size=("train_size", "min"),
        )
        .reindex(sorted_cats)
        .reset_index())

    # calculate p-values for combined categories
    def pval(cat1, cat2):
        x = df.loc[df["category"] == cat1, metric]
        y = df.loc[df["category"] == cat2, metric]
        t_stat, p = ttest_ind(x, y, equal_var=False)  # Welch's t-test
        return p

    combined = pd.DataFrame({
        "category": [f"{sorted_cats[2]}-{sorted_cats[0]}", f"{sorted_cats[1]}-{sorted_cats[0]}", f"{sorted_cats[1]}-{sorted_cats[2]}", f"{sorted_cats[3]}-{sorted_cats[2]}", f"{sorted_cats[3]}-{sorted_cats[4]}", f"{sorted_cats[2]}-{sorted_cats[4]}"],
        "mean": [
            (summary.iloc[2]['mean'] + summary.iloc[0]['mean']) / 2,
            (summary.iloc[1]['mean'] + summary.iloc[0]['mean']) / 2,
            (summary.iloc[1]['mean'] + summary.iloc[2]['mean']) / 2,
            (summary.iloc[3]['mean'] + summary.iloc[2]['mean']) / 2,
            (summary.iloc[3]['mean'] + summary.iloc[4]['mean']) / 2,
            (summary.iloc[2]['mean'] + summary.iloc[4]['mean']) / 2
        ],
        "std": [
            abs(summary.iloc[2]['mean'] - summary.iloc[0]['mean']) / 2,
            abs(summary.iloc[1]['mean'] - summary.iloc[0]['mean']) / 2,
            abs(summary.iloc[1]['mean'] - summary.iloc[2]['mean']) / 2,
            abs(summary.iloc[3]['mean'] - summary.iloc[2]['mean']) / 2,
            abs(summary.iloc[3]['mean'] - summary.iloc[4]['mean']) / 2,
            abs(summary.iloc[2]['mean'] - summary.iloc[4]['mean']) / 2,
        ],
        "p_value": [
            pval(sorted_cats[2], sorted_cats[0]),
            pval(sorted_cats[1], sorted_cats[0]),
            pval(sorted_cats[1], sorted_cats[2]),
            pval(sorted_cats[3], sorted_cats[2]),
            pval(sorted_cats[3], sorted_cats[4]),
            pval(sorted_cats[2], sorted_cats[4])
        ]
    })

    n = len(sorted_cats) + len(combined['category'].to_list())
    category_order = [None] * n

    category_order[::2] = combined['category'].to_list()
    category_order[1::2] = sorted_cats

    combined["train_size"] = np.nan
    summary["p_value"] = np.nan
    final_df = pd.concat([summary, combined], ignore_index=True)
    final_df = (
        final_df.assign(category=lambda x: pd.Categorical(x["category"], category_order, ordered=True))
        .sort_values("category")
        .reset_index(drop=True)
    )

    final_df["category"] = final_df.apply(
        lambda row: f"{row['category']}.{int(row['train_size'])}" 
        if pd.notnull(row['train_size']) 
        else row['category'], 
        axis=1
    )

    # final_df["category"] = final_df["category"].str.replace(r'\.full\.|\.ds\.', '.', regex=True)

    # y positions
    category_order = final_df["category"]
    y_pos = {cat: i for i, cat in enumerate(category_order)}

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in final_df.iterrows():
        y = y_pos[row["category"]]
        mean = row["mean"]
        sd = row["std"]
        is_acc_row = "-" not in str(row["category"])
        color = 'black' if is_acc_row else 'grey'
        text = f"{mean:.5f} ± {sd:.5f}" if is_acc_row else f"P = {row['p_value']:.4f}"
        marker_size = 4 if is_acc_row else 0
        ax.errorbar(mean, y, xerr=sd, fmt="o", color=color, markersize=marker_size)
        ax.text(mean, y + 0.15, text, ha="center", va="bottom", fontsize=8)

    # y-axis formatting
    ax.set_yticks([y_pos[c] for c in category_order])
    ax.set_yticklabels(category_order, fontsize=9)
    ax.set_ylim(-0.5, len(category_order) - 0.2)

    # labels & title
    ax.set_xlabel(f"{metric.upper()} (mean ± 2sd)", fontsize=8)
    ax.set_title(f"subset: {subset_value} | model: {model} | {df['fold_id'].nunique()} folds", fontsize=9)
    ax.grid(alpha=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def soak_plot_one_model_downsample_split(results_df, subset_value, model, metric="rmse", figsize=(6, 2.5)):

    def pval(cat1, cat2):
        x = df.loc[df["category"] == cat1, metric]
        y = df.loc[df["category"] == cat2, metric]
        t_stat, p = ttest_ind(x, y, equal_var=False)
        return p

    df = results_df[
        (results_df["subset"] == subset_value) &
        (results_df["model"] == model)
    ].copy()
    df["category"] = (
        df["category"]
        + "."
        + df["downsample"].map({False: "full", True: "ds"})
    )

    cats = set(df["category"].unique())
    sorted_cats_full = ["other.full", "same.full", "all.full"]
    sorted_cats_ds = ["other.ds", "same.full", "all.ds"]
    if "same.ds" in cats:
        sorted_cats_ds = ["other.full", "same.ds", "all.ds"]

    dfs = [None, None]
    for i, sorted_cats in enumerate([sorted_cats_full, sorted_cats_ds]):
        # base summary
        summary = (
            df.groupby("category", observed=False)
            .agg(
                mean=(metric, "mean"),
                std=(metric, "std"),
                train_size=("train_size", "min"),
            )
            .reindex(sorted_cats)
            .reset_index())

        combined = pd.DataFrame({
            "category": [f"{sorted_cats[0]}-{sorted_cats[1]}", f"{sorted_cats[2]}-{sorted_cats[1]}"],
            "mean": [
                (summary.iloc[0]['mean'] + summary.iloc[1]['mean']) / 2,
                (summary.iloc[2]['mean'] + summary.iloc[1]['mean']) / 2
            ],
            "std": [
                abs(summary.iloc[0]['mean'] - summary.iloc[1]['mean']) / 2,
                abs(summary.iloc[2]['mean'] - summary.iloc[1]['mean']) / 2
            ],
            "p_value": [
                pval(sorted_cats[0], sorted_cats[1]),
                pval(sorted_cats[2], sorted_cats[1]),
            ]
        })

        n = len(sorted_cats) + len(combined['category'].to_list())
        category_order = [None] * n
        category_order[::2] = sorted_cats
        category_order[1::2] = combined['category'].to_list()

        combined["train_size"] = np.nan
        summary["p_value"] = np.nan

        final = pd.concat([summary, combined], ignore_index=True)
        final = (
            final.assign(category=lambda x: pd.Categorical(x["category"], category_order, ordered=True))
            .sort_values("category")
            .reset_index(drop=True)
        )
        final["category"] = final.apply(
            lambda row: f"{row['category']}.{int(row['train_size'])}" 
            if pd.notnull(row['train_size']) 
            else row['category'], 
            axis=1
        )
        dfs[i] = final

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for idx, ax in enumerate(axes):
        df = dfs[idx]
        category_order = df['category'].unique().tolist()
        y_pos = {cat: i for i, cat in enumerate(category_order)}
        for i, row in df.iterrows():
            y = y_pos[row["category"]]
            mean = row["mean"]
            sd = row["std"]
            color = 'black' if i % 2 == 0 else 'grey'
            text = f"{mean:.5f} ± {sd:.5f}" if i % 2 == 0 else f"P = {row['p_value']:.4f}"
            marker_size = 4 if i % 2 == 0 else 0
            ax.errorbar(mean, y, xerr=sd, fmt="o", color=color, markersize=marker_size)
            ax.text(mean, y + 0.15, text, ha="center", va="bottom", fontsize=8)

        # y-axis formatting
        ax.set_yticks([y_pos[c] for c in category_order])
        ax.set_yticklabels(category_order, fontsize=9)
        ax.set_ylim(-0.5, len(category_order) - 0.2)

        # labels & title
        ax.set_title("full" if idx==0 else "downsample", fontsize=10)
        ax.grid(alpha=0.5)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.tick_params(axis='x', labelsize=9)

    fig.supxlabel(f"{metric.upper()} (mean ± 2sd) | subset: {subset_value} | model: {model} | {set(results_df['fold_id']).__len__()} test folds | {len(set(results_df['seed_id']))-1} random seeds for downsample", fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig