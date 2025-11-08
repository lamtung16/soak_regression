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

class SOAK:
    # --- init ---
    def __init__(self, n_splits=5, seed=123):
        self.n_splits = n_splits
        self.seed = seed
        self.all_models = {
            "featureless": self._featureless_model,
            "linear": self._linear_model,
            "linear_BasExp": self._linear_BasExp_model,
        }

    # --- Model methods ---
    def _featureless_model(self, train_idx, test_idx):
        y = self.df.select(self.target_col).to_numpy().flatten()
        y_pred = np.repeat(y[train_idx].mean(), len(test_idx))
        mse = mean_squared_error(y[test_idx], y_pred)
        mae = mean_absolute_error(y[test_idx], y_pred)
        return mse, mae

    def _linear_model(self, train_idx, test_idx):
        model = LinearRegression().fit(self.df.select(self.feature_cols).to_numpy()[train_idx], 
                                       self.df.select(self.target_col).to_numpy().flatten()[train_idx])
        y_pred = model.predict(self.df.select(self.feature_cols).to_numpy()[test_idx])
        mse = mean_squared_error(self.df.select(self.target_col).to_numpy().flatten()[test_idx], y_pred)
        mae = mean_absolute_error(self.df.select(self.target_col).to_numpy().flatten()[test_idx], y_pred)
        return mse, mae

    def _linear_BasExp_model(self, train_idx, test_idx):
        pipeline = Pipeline([
            ('poly', PolynomialFeatures()),
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-2, 2, 10), cv=4))
        ])
        pipeline.fit(self.df.select(self.feature_cols).to_numpy()[train_idx], 
                     self.df.select(self.target_col).to_numpy().flatten()[train_idx])
        y_pred = pipeline.predict(self.df.select(self.feature_cols).to_numpy()[test_idx])
        mse = mean_squared_error(self.df.select(self.target_col).to_numpy().flatten()[test_idx], y_pred)
        mae = mean_absolute_error(self.df.select(self.target_col).to_numpy().flatten()[test_idx], y_pred)
        return mse, mae
    
    
    # --- down sample if chosen ---
    def _downsample_subsets(self, seed):
        min_size = self.df.group_by(self.subset_col).count().select("count").min().item()
        downsampled_parts = []
        for _, group_df in self.df.group_by(self.subset_col):
            sampled = group_df.sample(n=min_size, shuffle=True, seed=seed)
            downsampled_parts.append(sampled)
        return pl.concat(downsampled_parts)


    # --- subset analysis ---
    def subset_analyze(self, df, subset_col, target_col, subset_values=None, n_folds=5, models=None, downsample_majority=False, seed=123):

        self.df = df
        self.feature_cols = [c for c in df.columns if c.startswith("X_")]
        self.subset_col = subset_col
        self.target_col = target_col

        feature_scaler = StandardScaler()
        scaled_features = feature_scaler.fit_transform(self.df.select(self.feature_cols).to_numpy())
        for i, col in enumerate(self.feature_cols):
            self.df = self.df.with_columns(pl.Series(col, scaled_features[:, i]))

        if models is None:
            models = list(self.all_models.keys())
        self.model_dict = {name: self.all_models[name] for name in models if name in self.all_models}

        if downsample_majority:
            self.df = self._downsample_subsets(seed)

        if subset_values is None:
            subset_values = (self.df.select(self.subset_col).unique().to_series().to_list())

        rows = []
        for subset_val in subset_values:
            subset_idx = np.where(self.df.select(self.subset_col).to_numpy().flatten() == subset_val)[0]
            other_idx  = np.where(self.df.select(self.subset_col).to_numpy().flatten() != subset_val)[0]
            kf = KFold(n_splits=n_folds)

            for fold, (train_idx, test_idx) in enumerate(kf.split(subset_idx)):
                test_indices = subset_idx[test_idx]
                same_indices = subset_idx[train_idx]
                all_indices = np.concatenate([same_indices, other_idx])

                train_sets = {
                    "all": all_indices,
                    "same": same_indices,
                    "other": other_idx,
                }

                for category, train_indices in train_sets.items():
                    for model_name, model_func in self.model_dict.items():
                        mse, mae = model_func(train_indices, test_indices)
                        rows.append({
                            "subset": subset_val,
                            "category": category,
                            "test_fold": fold + 1,
                            "model": model_name,
                            "mse": mse,
                            "mae": mae,
                        })

        self.results_df = pl.DataFrame(rows)

    # --- SOAK split ---
    def split(self, X, y, subset_vec):
        results = []
        for subset_val in np.unique(subset_vec):
            same_idx = np.where(subset_vec == subset_val)[0]
            other_idx = np.where(subset_vec != subset_val)[0]

            for fold_id, (train_idx, test_idx) in enumerate(KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed).split(same_idx)):
                test_fold_idx = same_idx[test_idx]
                X_train_same, y_train_same = X[same_idx[train_idx]], y[same_idx[train_idx]]
                X_train_other, y_train_other = X[other_idx], y[other_idx]

                for category, (X_train, y_train) in {
                    'same': (X_train_same, y_train_same),
                    'other': (X_train_other, y_train_other),
                    'all': (np.vstack([X_train_same, X_train_other]), np.concatenate([y_train_same, y_train_other]))
                }.items():
                    results.append([subset_val, category, fold_id, X_train, X[test_fold_idx], y_train, y[test_fold_idx]])
        return results
    

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

                    if j % 2 == 0:
                        for k, row in enumerate(data_mean.to_dicts()):
                            color = model_colors[row["model"]]
                            ax.errorbar(row[f"{metric}_mean"], k, xerr=row[f"{metric}_sd"], fmt='o', color=color, capsize=2, markersize=5)
                    else:
                        for model in model_to_y:
                            vals = data.filter(pl.col("model") == model).select(f"log_{metric}").to_numpy().flatten()
                            ax.scatter(vals, np.ones_like(vals) * model_to_y[model], facecolor=model_colors[model], edgecolor=model_colors[model], s=20)

            # Legend and title
            handles = [Line2D([], [], marker='o', color=color, markerfacecolor=color,linestyle='', markersize=5, label=m)
                    for m, color in reversed(list(model_colors.items()))]
            fig.legend(handles=handles, loc='upper right', fontsize=7)
            fig.suptitle(f'Subset: {subset} (n={n_obs})', fontsize=11, fontweight='bold')
            plt.tight_layout()
            fig_dict[subset] = fig

        return fig_dict