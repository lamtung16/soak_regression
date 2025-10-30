import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

class soak:
    def __init__(self, df, subset_col="subset", target_col="y", n_folds=5, models=None):
        self.df = df.copy()
        self.subset_col = subset_col
        self.target_col = target_col
        self.n_folds = n_folds
        
        # Available models
        available_models = {
            "featureless": self.featureless_model,
            "linear": self.linear_model,
            "gam": self.gam_model
        }
        
        # If user didn't provide models, use all by default
        if models is None:
            models = list(available_models.keys())
        
        # Keep only valid models
        self.model_dict = {name: available_models[name] for name in models if name in available_models}
        
        self.feature_cols = [c for c in self.df.columns if c not in [self.subset_col, self.target_col]]
        self.results_df = None
        
        self._scale_features_and_target()

    def _scale_features_and_target(self):
        # Scale features
        feature_scaler = StandardScaler()
        self.df[self.feature_cols] = feature_scaler.fit_transform(self.df[self.feature_cols])

        # Scale target
        target_scaler = StandardScaler()
        self.df[self.target_col] = target_scaler.fit_transform(self.df[[self.target_col]])

    # --- Model definitions ---
    @staticmethod
    def featureless_model(X_train, X_test, y_train, y_test):
        y_pred = np.repeat(y_train.mean(), len(y_test))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return mse, mae
    
    @staticmethod
    def linear_model(X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return mse, mae
    
    @staticmethod
    def gam_model(X_train, X_test, y_train, y_test):
        pipeline = Pipeline([
            ('poly', PolynomialFeatures()),
            ('scaler', StandardScaler()),
            ('lasso', LassoCV(cv=4, n_jobs=-1, max_iter=50000, tol=0.01))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return mse, mae

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

        subset_counts = self.df['subset'].value_counts().to_dict()
        df = self.results_df.copy()
        df['log_mse'] = np.log10(df['mse'])

        subsets = df['subset'].unique()
        categories = df['category'].unique()
        models = df['model'].unique()

        # Aggregate by subset, category, model
        df_mean = df.groupby(['subset', 'category', 'model'], sort=False).agg(
            mse_mean=('log_mse', 'mean'),
            mse_sd=('log_mse', 'std')
        ).reset_index()

        # Assign colors for each model dynamically
        model_colors = {model: plt.cm.tab10(i) for i, model in enumerate(models)}

        fig_dict = {}

        for subset in subsets:
            n_obs = subset_counts.get(subset, 0)
            fig, axes = plt.subplots(len(categories), 2, figsize=(10, 1.1 * len(categories)), sharex=True)

            if len(categories) == 1:
                axes = np.array([axes])

            for i, category in enumerate(categories):
                for j in range(2):
                    ax_err = axes[i, j]
                    ax_err.grid(alpha=0.1)

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

                    # Left column: means with error bars
                    if j == 0:
                        for k, row in enumerate(data_mean.itertuples()):
                            color = model_colors[row.model]
                            ax_err.errorbar(
                                row.mse_mean, k, xerr=row.mse_sd, fmt='o',
                                color=color,
                                markerfacecolor=color,
                                markeredgecolor=color,
                                capsize=2, markersize=5
                            )
                    # Right column: all points
                    else:
                        for model in model_names:
                            vals = data.loc[data['model'] == model, 'log_mse']
                            y_pos = model_to_y[model]
                            color = model_colors[model]
                            ax_err.scatter(vals, np.ones_like(vals) * y_pos,
                                        facecolor=color, edgecolor=color, s=20)

            # Global legend and title
            handles = [
                plt.Line2D([], [], marker='o', color=color, markerfacecolor=color, linestyle='', markersize=5, label=m)
                for m, color in model_colors.items()
            ]
            fig.legend(handles=handles, loc='upper right', fontsize=8)
            fig.suptitle(f'test subset: {subset} (n = {n_obs})', fontsize=11, fontweight='bold')
            plt.tight_layout()

            fig_dict[subset] = fig

        return fig_dict