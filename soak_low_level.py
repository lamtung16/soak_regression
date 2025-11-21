import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import uniform


def evaluate(y_pred, y_test):
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.median(np.abs(y_test - y_pred))
    return rmse, mae


def featureless_model(X_train, y_train, X_test, y_test):
    mean_target = np.mean(y_train)
    y_pred = np.full_like(y_test, mean_target)
    return evaluate(y_pred, y_test)


def linear_model(X_train, y_train, X_test, y_test):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=np.logspace(-2, 2, 10), cv=4))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate(y_pred, y_test)


def linear_BasExp_model(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=np.logspace(-2, 2, 10), cv=4))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return evaluate(y_pred, y_test)


def treeCV_model(X_train, y_train, X_test, y_test):
    param_grid = {'max_depth': np.arange(2, 21, 2)}
    grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=4, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)
    return evaluate(y_pred, y_test)


def mlpCV_model(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(max_iter=10000, early_stopping=True))
    ])
    param_dist = {
        'mlp__hidden_layer_sizes': [(20,), (20, 20), (80), (80, 80)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__solver': ['adam', 'lbfgs'],
        'mlp__alpha': uniform(0.001, 10),
    }
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=40,
        cv=4,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    return evaluate(y_pred, y_test)


all_models = {
    "featureless": featureless_model,
    "linear": linear_model,
    "linear_BasExp": linear_BasExp_model,
    "tree": treeCV_model,
    "mlp": mlpCV_model
}


class SOAKFold:
    def __init__(self, n_splits=5, seed=123):
        self.n_splits = n_splits
        self.seed = seed

    def split(self, X, y, subset_vec):
        splits = []
        for subset_value in np.unique(subset_vec):
            same_idx = np.where(subset_vec == subset_value)[0]
            other_idx = np.where(subset_vec != subset_value)[0]
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            for (train_idx, test_idx) in kf.split(same_idx):
                X_train_same, y_train_same = X[same_idx[train_idx]], y[same_idx[train_idx]]
                X_train_other, y_train_other = X[other_idx], y[other_idx]
                for category, (X_train, y_train) in {
                    'same': (X_train_same, y_train_same),
                    'other': (X_train_other, y_train_other),
                    'all': (np.vstack([X_train_same, X_train_other]), np.concatenate([y_train_same, y_train_other]))
                }.items():
                    splits.append([subset_value, category, X_train, y_train, X[same_idx[test_idx]], y[same_idx[test_idx]]])
        return splits
    
    # @staticmethod
    # def downsample_majority(X, y, subset_vec, seed=123):
    #     np.random.seed(seed)
    #     unique_subsets, counts = np.unique(subset_vec, return_counts=True)
    #     n_min = counts.min()
    #     indices_to_keep = []
    #     for subset in unique_subsets:
    #         subset_indices = np.where(subset_vec == subset)[0]
    #         chosen_indices = np.random.choice(subset_indices, size=n_min, replace=False)
    #         indices_to_keep.extend(chosen_indices)
    #     indices_to_keep = np.array(indices_to_keep)
    #     return X[indices_to_keep], y[indices_to_keep], subset_vec[indices_to_keep]

    @staticmethod
    def model_eval(X_train, y_train, X_test, y_test, model='featureless'):
        return all_models[model](X_train, y_train, X_test, y_test)
    
    @staticmethod
    def plot_metrics(df, subset_value, subset_vec=None, metric='rmse', figsize=(5, 3)):
        import matplotlib.pyplot as plt

        observation_info = "" if subset_vec is None else f"(n = {np.sum(subset_vec == subset_value)})"

        df = df[df['subset'] == subset_value].copy()
        df = df.groupby(['subset', 'category', 'model']).agg(
            avg=(f'{metric}', 'mean'),
            sd=(f'{metric}', 'std'),
        ).reset_index()
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)
        models = df['model'].unique()
        for i, category in enumerate(['all', 'same', 'other']):
            ax = axes[i]
            ax.grid(alpha = 0.2)
            ax.set_ylim(-0.5, len(models) - 0.5)
            ax.set_ylabel(category, rotation=0, labelpad=15, va='center', fontweight='bold')
            ax.set_yticklabels([])
            category_data = df[df['category'] == category]
            for k, model in enumerate(models):
                row = category_data[category_data['model'] == model]
                _x = row['avg'].values[0]
                _xerr = row['sd'].values[0]
                ax.errorbar(_x, k, xerr=_xerr, fmt='o', label=model)
                ax.text(_x, k, f"{_x:.3f}±{_xerr:.3f}", ha='center', va='bottom')
        ax.set_xlabel(f'{metric.upper()}')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc='upper right', fontsize=9)
        fig.suptitle(f'Subset: {subset_value} {observation_info}', fontsize=10, fontweight='bold')
        plt.tight_layout()
        return fig