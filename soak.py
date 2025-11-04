import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


def featureless_model(X_train, X_test, y_train, y_test):
    y_pred = np.repeat(y_train.mean(), len(y_test))
    return mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)


def linear_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)


def gam_model(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=np.logspace(-2, 2, 20), cv=4))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)


# Available models
available_models = {
    "featureless": featureless_model,
    "linear": linear_model,
    "gam": gam_model
}


class soak:
    def __init__(self, df, subset_col, target_col, n_folds=5, models=None):
        self.df = df.copy()
        self.subset_col = subset_col
        self.target_col = target_col
        self.n_folds = n_folds
        
        # If user didn't provide models, use all by default
        if models is None:
            models = list(available_models.keys())
        
        # Keep only valid models
        self.model_dict = {name: available_models[name] for name in models if name in available_models}
        
        self.feature_cols = [c for c in self.df.columns if c.startswith('X_')]
        self.results_df = None
        
        self._scale_features_and_target()

    def _scale_features_and_target(self):
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        self.df[self.feature_cols] = feature_scaler.fit_transform(self.df[self.feature_cols])
        self.df[self.target_col] = target_scaler.fit_transform(self.df[[self.target_col]])

    # --- Analyze ---
    def analyze(self, subset_names):
        results = []

        for subset_name in subset_names:
            subset_idx = np.where(self.df[self.subset_col] == subset_name)[0]
            other_indices = np.where(self.df[self.subset_col] != subset_name)[0]
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

                for category, train_indices in train_dict.items():
                    for model_name, model_func in self.model_dict.items():
                        mse, mae = model_func(
                            X_train=self.df[self.feature_cols].values[train_indices],
                            X_test=self.df[self.feature_cols].values[test_indices],
                            y_train=self.df[self.target_col].values[train_indices],
                            y_test=self.df[self.target_col].values[test_indices]
                        )
                        results.append({
                            "subset": subset_name,
                            "category": category,
                            "test_fold": fold + 1,
                            "model": model_name,
                            "mse": mse,
                            "mae": mae
                        })
        self.results_df = pd.DataFrame(results)
    

    def plot_results(self):
        if self.results_df is None:
            raise ValueError("No results found. Run analyze() first.")

        subset_counts = self.df[self.subset_col].value_counts().to_dict()
        df = self.results_df.copy()
        df['log_mse'] = np.log10(df['mse'])
        df['log_mae'] = np.log10(df['mae'])

        subsets = df['subset'].unique()
        categories = df['category'].unique()
        models = df['model'].unique()

        # Aggregate by subset, category, model
        df_mean = df.groupby(['subset', 'category', 'model'], sort=False).agg(
            mse_mean=('log_mse', 'mean'),
            mse_sd=('log_mse', 'std'),
            mae_mean=('log_mae', 'mean'),
            mae_sd=('log_mae', 'std')
        ).reset_index()

        # Assign colors for each model dynamically
        model_colors = {model: plt.cm.tab10(i) for i, model in enumerate(models)}

        fig_dict = {}

        for subset in subsets:

            # Determine shared x-axis limits
            df_mean_subset = df_mean[df_mean['subset'] == subset]
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

                    data_mean = df_mean[(df_mean['category'] == category) & (df_mean['subset'] == subset)]
                    data = df[(df['category'] == category) & (df['subset'] == subset)]

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
                        for k, row in enumerate(data_mean.itertuples()):
                            color = model_colors[row.model]
                            ax_err.errorbar(
                                row.mse_mean, k, xerr=row.mse_sd, fmt='o',
                                color=color, markerfacecolor=color, markeredgecolor=color,
                                capsize=2, markersize=5
                            )
                    elif j == 1:
                        for model in model_names:
                            vals = data.loc[data['model'] == model, 'log_mse']
                            y_pos = model_to_y[model]
                            color = model_colors[model]
                            ax_err.scatter(vals, np.ones_like(vals) * y_pos, facecolor=color, edgecolor=color, s=20)
                    elif j == 2:
                        for k, row in enumerate(data_mean.itertuples()):
                            color = model_colors[row.model]
                            ax_err.errorbar(
                                row.mae_mean, k, xerr=row.mae_sd, fmt='o',
                                color=color, markerfacecolor=color, markeredgecolor=color,
                                capsize=2, markersize=5
                            )
                    elif j == 3:
                        for model in model_names:
                            vals = data.loc[data['model'] == model, 'log_mae']
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