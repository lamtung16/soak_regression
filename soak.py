import numpy as np
import polars as pl
import matplotlib.pyplot as plt
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
                        print(rows)
                # Append small chunk to the list
                results_chunks.append(pl.DataFrame(rows))

        # Concatenate all chunks once at the end
        self.results_df = pl.concat(results_chunks, rechunk=True)
        
    def plot_results(self):
        if self.results_df is None:
            raise ValueError("No results found. Run analyze() first.")

        subset_counts = self.df[self.subset_col].value_counts().to_dict()
        df = self.results_df
        df = df.with_columns([
            (pl.col("mse").log10()).alias("log_mse"),
            (pl.col("mae").log10()).alias("log_mae")
        ])

        subsets = df['subset'].unique()
        categories = df['category'].unique()
        models = df['model'].unique()

        # Aggregate by subset, category, model
        df_mean = (
            df.group_by(['subset', 'category', 'model'], maintain_order=True)
            .agg([
                pl.col("log_mse").mean().alias("mse_mean"),
                pl.col("log_mse").std().alias("mse_sd"),
                pl.col("log_mae").mean().alias("mae_mean"),
                pl.col("log_mae").std().alias("mae_sd")
            ])
        )

        # Assign colors for each model dynamically
        model_colors = {model: plt.cm.tab10(i) for i, model in enumerate(models)}

        fig_dict = {}

        for subset in subsets:

            # Determine shared x-axis limits
            df_mean_subset = df_mean.filter(pl.col("subset") == subset)
            mse_min = df_mean_subset['mse_mean'].min() - 1.8*df_mean_subset['mse_sd'].max()
            mse_max = df_mean_subset['mse_mean'].max() + 1.8*df_mean_subset['mse_sd'].max()
            mae_min = df_mean_subset['mae_mean'].min() - 1.8*df_mean_subset['mae_sd'].max()
            mae_max = df_mean_subset['mae_mean'].max() + 1.8*df_mean_subset['mae_sd'].max()

            n_obs = subset_counts.get(subset, 0)
            fig, axes = plt.subplots(len(categories), 4, figsize=(16, 1.1 * len(categories)), sharex=False)

            if len(categories) == 1:
                axes = np.array([axes])

            for i, category in enumerate(categories):
                for j in range(4):
                    ax_err = axes[i, j]
                    ax_err.grid(alpha=0.1)

                    data_mean = df_mean.filter((pl.col("category") == category) & (pl.col("subset") == subset))
                    data = df.filter((pl.col("category") == category) & (pl.col("subset") == subset))

                    model_names = data['model'].unique()
                    model_to_y = {m: yk for yk, m in enumerate(model_names)}

                    ax_err.set_ylim(-0.5, len(models) - 0.5)

                    # Set x-axis label based on column
                    if i == len(categories) - 1:
                        if j in [0, 1]:
                            ax_err.set_xlabel('log(MSE)')
                        elif j in [2, 3]:
                            ax_err.set_xlabel('log(MAE)')

                    # Only show category on first column
                    if j == 0:
                        ax_err.set_ylabel(category, rotation=0, labelpad=15, va='center', fontweight='bold')
                    else:
                        ax_err.set_ylabel('')

                    ax_err.set_yticklabels([])

                    # Set shared x-limits for MSE and MAE
                    if j in [0, 1]:
                        ax_err.set_xlim(mse_min, mse_max)
                    elif j in [2, 3]:
                        ax_err.set_xlim(mae_min, mae_max)

                    # Columns: 0 = log(MSE) mean ± SD, 1 = log(MSE) all points, 2 = log(MAE) mean ± SD, 3 = log(MAE) all points
                    if j == 0:
                        for k, row in enumerate(data_mean.to_dicts()):
                            color = model_colors[row["model"]]
                            ax_err.errorbar(
                                row["mse_mean"], k, xerr=row["mse_sd"], fmt='o',
                                color=color, markerfacecolor=color, markeredgecolor=color,
                                capsize=2, markersize=5
                            )
                    elif j == 1:
                        for model in model_names:
                            vals = data.filter(pl.col("model") == model).select("log_mse").to_numpy().flatten()
                            y_pos = model_to_y[model]
                            color = model_colors[model]
                            ax_err.scatter(vals, np.ones_like(vals) * y_pos, facecolor=color, edgecolor=color, s=20)
                    elif j == 2:
                        for k, row in enumerate(data_mean.to_dicts()):
                            color = model_colors[row["model"]]
                            ax_err.errorbar(
                                row["mae_mean"], k, xerr=row["mae_sd"], fmt='o',
                                color=color, markerfacecolor=color, markeredgecolor=color,
                                capsize=2, markersize=5
                            )
                    elif j == 3:
                        for model in model_names:
                            vals = data.filter(pl.col("model") == model).select("log_mae").to_numpy().flatten()
                            y_pos = model_to_y[model]
                            color = model_colors[model]
                            ax_err.scatter(vals, np.ones_like(vals) * y_pos, facecolor=color, edgecolor=color, s=20)

            # Global legend and title
            handles = [
                plt.Line2D([], [], marker='o', color=color, markerfacecolor=color, linestyle='', markersize=5, label=m)
                for m, color in reversed(list(model_colors.items()))
            ]
            fig.legend(handles=handles, loc='upper right', fontsize=8)
            fig.suptitle(f'test subset: {subset} (n = {n_obs})', fontsize=11, fontweight='bold')
            plt.tight_layout()

            fig_dict[subset] = fig

        return fig_dict