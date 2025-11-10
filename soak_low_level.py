import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


def evaluate(y_pred, y_test):
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    return mse, mae


def featureless_model(X_train, y_train, X_test, y_test):
    mean_target = np.mean(y_train)
    y_pred = np.full_like(y_test, mean_target)
    return evaluate(y_pred, y_test)


def linear_model(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate(y_pred, y_test)


class SOAK:
    def __init__(self, n_splits=5, seed=123):
        self.n_splits = n_splits
        self.seed = seed
        self.all_models = {
            "featureless": featureless_model,
            "linear": linear_model
        }

    def split(self, X, y, subset_vec):
        splits = []
        for subset_value in np.unique(subset_vec):
            same_idx = np.where(subset_vec == subset_value)[0]
            other_idx = np.where(subset_vec != subset_value)[0]

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            for (train_idx, test_idx) in kf.split(same_idx):
                test_fold_idx = same_idx[test_idx]
                X_train_same, y_train_same = X[same_idx[train_idx]], y[same_idx[train_idx]]
                X_train_other, y_train_other = X[other_idx], y[other_idx]

                for category, (X_train, y_train) in {
                    'same': (X_train_same, y_train_same),
                    'other': (X_train_other, y_train_other),
                    'all': (np.vstack([X_train_same, X_train_other]), np.concatenate([y_train_same, y_train_other]))
                }.items():
                    splits.append([subset_value, category, X_train, y_train, X[test_fold_idx], y[test_fold_idx]])
        
        return splits
    
    def model_eval(self, X_train, y_train, X_test, y_test, model='featureless'):
        mse, mae = self.all_models[model](X_train, y_train, X_test, y_test)
        return mse, mae
    
    @staticmethod
    def plot_metrics(df, subset_value, metric='mse', figsize=(5, 3)):
        import pandas as pd
        import matplotlib.pyplot as plt

        df = df[df['subset'] == subset_value].copy()
        df.loc[:, f'log_{metric}'] = np.log10(df[metric])
        df = df.groupby(['subset', 'category', 'model']).agg(
            avg=(f'log_{metric}', 'mean'),
            sd=(f'log_{metric}', 'std'),
        ).reset_index()
        
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)
        
        # Color mapping for models
        models = df['model'].unique()
        color_dict = {model: plt.colormaps['tab10'](i) for i, model in enumerate(models)}
        
        # Loop through each category to create a subplot
        for i, category in enumerate(['all', 'same', 'other']):
            ax = axes[i]
            ax.grid(alpha = 0.1)
            ax.set_ylim(-0.5, len(models) - 0.5)
            ax.set_ylabel(category, rotation=0, labelpad=15, va='center', fontweight='bold')
            ax.set_yticklabels([])
            
            # Filter data for the current category
            category_data = df[df['category'] == category]
            
            # Loop through each model
            for k, model in enumerate(models):
                model_data = category_data[category_data['model'] == model]
                ax.errorbar(model_data['avg'].values[0], k, xerr=model_data['sd'].values[0], fmt='o', label=model, color=color_dict[model])

        ax.set_xlabel(f'log({metric.upper()})')
        
        # Add a single shared legend at the bottom (or top)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', fontsize=9)

        # Adjust layout
        fig.suptitle(f'Subset: {subset_value}', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.show()