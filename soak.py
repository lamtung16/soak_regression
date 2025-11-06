import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

class soak:
    def __init__(self, df, subset_col, target_col, n_folds=5, models=None):
        self.df = df
        self.subset_col = subset_col
        self.target_col = target_col
        self.n_folds = n_folds

        # Identify feature columns
        self.feature_cols = [c for c in df.columns if c.startswith("X_")]

        # --- Store scalers ---
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Fit scalers only
        self.feature_scaler.fit(df.select(self.feature_cols).to_numpy())
        self.target_scaler.fit(df.select(self.target_col).to_numpy().reshape(-1, 1))

        # --- Model selection ---
        all_models = {
            "featureless": self._featureless_model,
            "linear": self._linear_model,
            "gam": self._gam_model,
        }
        if models is None:
            models = list(all_models.keys())
        self.model_dict = {name: all_models[name] for name in models if name in all_models}

        self.results_df = None

    # --- Utility to get scaled NumPy arrays ---
    def _get(self, cols, scale=True):
        arr = self.df.select(cols).to_numpy()
        if scale:
            if cols == [self.target_col]:
                arr = self.target_scaler.transform(arr.reshape(-1, 1)).flatten()
            else:
                arr = self.feature_scaler.transform(arr)
        return arr

    # --- Model methods ---
    def _featureless_model(self, train_idx, test_idx):
        mse = mean_squared_error(self._get([self.target_col], scale=True)[test_idx], np.repeat(self._get([self.target_col], scale=True)[train_idx].mean(), len(test_idx)))
        mae = mean_absolute_error(self._get([self.target_col], scale=True)[test_idx], np.repeat(self._get([self.target_col], scale=True)[train_idx].mean(), len(test_idx)))
        return mse, mae 

    def _linear_model(self, train_idx, test_idx):
        model = LinearRegression()
        model.fit(self._get(self.feature_cols, scale=True)[train_idx], self._get([self.target_col], scale=True)[train_idx])
        mse = mean_squared_error(self._get([self.target_col], scale=True)[test_idx], model.predict(self._get(self.feature_cols, scale=True)[test_idx]))
        mae = mean_absolute_error(self._get([self.target_col], scale=True)[test_idx], model.predict(self._get(self.feature_cols, scale=True)[test_idx]))
        return mse, mae 

    def _gam_model(self, train_idx, test_idx):
        pipeline = Pipeline([
            ('poly', PolynomialFeatures()),
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-2, 2, 20), cv=4))
        ])
        pipeline.fit(self._get(self.feature_cols, scale=True)[train_idx], self._get([self.target_col], scale=True)[train_idx])
        mse = mean_squared_error(self._get([self.target_col], scale=True)[test_idx], pipeline.predict(self._get(self.feature_cols, scale=True)[test_idx]))
        mae = mean_absolute_error(self._get([self.target_col], scale=True)[test_idx], pipeline.predict(self._get(self.feature_cols, scale=True)[test_idx]))
        return mse, mae

    # --- Main analysis ---
    def analyze(self, subset_names):
        # Use a list to accumulate small Polars DataFrames
        results_chunks = []

        subset_col_np = self._get([self.subset_col], scale=False).flatten()

        for subset_name in subset_names:
            subset_idx = np.where(subset_col_np == subset_name)[0]
            other_indices = np.where(subset_col_np != subset_name)[0]
            kf = KFold(n_splits=self.n_folds)

            for fold, (train_idx, test_idx) in enumerate(kf.split(subset_idx)):
                test_indices = subset_idx[test_idx]
                same_indices = subset_idx[train_idx]
                all_indices = np.concatenate([same_indices, other_indices])

                train_dict = {
                    "all": all_indices,
                    "same": same_indices,
                    "other": other_indices,
                }

                # Compute results for each model/category
                rows = []
                for category, train_indices in train_dict.items():
                    for model_name, model_func in self.model_dict.items():
                        mse, mae = model_func(train_indices, test_indices)
                        rows.append({
                            "subset": subset_name,
                            "category": category,
                            "test_fold": fold + 1,
                            "model": model_name,
                            "mse": mse,
                            "mae": mae
                        })
                        
                # Append small chunk to the list
                results_chunks.append(pl.DataFrame(rows))

        # Concatenate all chunks once at the end
        self.results_df = pl.concat(results_chunks, rechunk=True)
        
    def plot_results(self):
        if self.results_df is None:
            raise ValueError("No results found. Run analyze() first.")

        # Compute log metrics
        df = self.results_df.with_columns([
            pl.col("mse").log10().alias("log_mse"),
            pl.col("mae").log10().alias("log_mae")
        ]).sort("model")

        subsets = df['subset'].unique()
        categories = ['all', 'same', 'other']
        models = sorted(df['model'].unique())

        # Aggregate mean and std
        df_mean = (
            df.group_by(['subset', 'category', 'model'], maintain_order=True)
            .agg([
                pl.col("log_mse").mean().alias("mse_mean"),
                pl.col("log_mse").std().alias("mse_sd"),
                pl.col("log_mae").mean().alias("mae_mean"),
                pl.col("log_mae").std().alias("mae_sd")
            ])
        )

        # Assign colors
        cmap = cm.get_cmap('tab10')
        model_colors = {model: cmap(i % 10) for i, model in enumerate(models)}

        fig_dict = {}

        for subset in subsets:
            df_subset = df.filter(pl.col("subset") == subset)
            df_mean_subset = df_mean.filter(pl.col("subset") == subset)
            n_obs = self.df.filter(pl.col(self.subset_col) == subset).height

            fig, axes = plt.subplots(len(categories), 4, figsize=(16, 1.1 * len(categories)), sharex=False)
            axes = np.array([axes]) if len(categories) == 1 else axes

            mse_min = df_mean_subset['mse_mean'].min() - 2 * df_mean_subset['mse_sd'].max()
            mse_max = df_mean_subset['mse_mean'].max() + 2 * df_mean_subset['mse_sd'].max()
            mae_min = df_mean_subset['mae_mean'].min() - 2 * df_mean_subset['mae_sd'].max()
            mae_max = df_mean_subset['mae_mean'].max() + 2 * df_mean_subset['mae_sd'].max()

            for i, category in enumerate(categories):
                data = df_subset.filter(pl.col("category") == category)
                data_mean = df_mean_subset.filter(pl.col("category") == category)
                model_to_y = {m: y for y, m in enumerate(sorted(data['model'].unique()))}

                for j, metric in enumerate(['mse', 'mse', 'mae', 'mae']):
                    ax = axes[i, j]
                    ax.grid(alpha=0.1)
                    ax.set_ylim(-0.5, len(models) - 0.5)
                    ax.set_yticklabels([])

                    # Set global x-limits for MSE/MAE
                    if metric == 'mse':
                        ax.set_xlim(mse_min, mse_max)
                    else:
                        ax.set_xlim(mae_min, mae_max)

                    if j == 0:
                        ax.set_ylabel(category, rotation=0, labelpad=15, va='center', fontweight='bold')
                    else:
                        ax.set_ylabel('')

                    if i == len(categories) - 1:
                        if j % 2 == 0:
                            ax.set_xlabel(f'log({metric.upper()})')
                        else:
                            ax.set_xlabel(f'log({metric.upper()})')

                    # Plot mean ± sd or individual points
                    if j % 2 == 0:  # mean ± sd
                        for k, row in enumerate(data_mean.to_dicts()):
                            color = model_colors[row["model"]]
                            ax.errorbar(row[f"{metric}_mean"], k, xerr=row[f"{metric}_sd"],
                                        fmt='o', color=color, capsize=2, markersize=5)
                    else:  # individual points
                        for model in model_to_y:
                            vals = data.filter(pl.col("model") == model).select(f"log_{metric}").to_numpy().flatten()
                            ax.scatter(vals, np.ones_like(vals) * model_to_y[model],
                                    facecolor=model_colors[model], edgecolor=model_colors[model], s=20)

            # Legend and title
            handles = [Line2D([], [], marker='o', color=color, markerfacecolor=color,
                            linestyle='', markersize=5, label=m)
                    for m, color in reversed(list(model_colors.items()))]
            fig.legend(handles=handles, loc='upper right', fontsize=8)
            fig.suptitle(f'Subset: {subset} (n={n_obs})', fontsize=11, fontweight='bold')
            plt.tight_layout()
            fig_dict[subset] = fig

        return fig_dict
