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
    return y_pred, (rmse, mae)


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
            for fold_id, (train_idx, test_idx) in enumerate(kf.split(same_idx)):
                X_train_same, y_train_same = X[same_idx[train_idx]], y[same_idx[train_idx]]
                X_train_other, y_train_other = X[other_idx], y[other_idx]
                for category, (X_train, y_train) in {
                    'same': (X_train_same, y_train_same),
                    'other': (X_train_other, y_train_other),
                    'all': (np.vstack([X_train_same, X_train_other]), np.concatenate([y_train_same, y_train_other]))
                }.items():
                    splits.append([subset_value, category, fold_id + 1, X_train, y_train, X[same_idx[test_idx]], y[same_idx[test_idx]]])
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
    def plot_metrics(results_df, subset_value, model, metric, figsize=(6, 2.5)):
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        
        _, counts = np.unique(results_df['downsample'], return_counts=True)

        df = results_df[
            (results_df['subset'] == subset_value) &
            (results_df['model'] == model)
        ].copy()

        df = df.groupby(
            ['subset', 'category', 'model', 'downsample', 'train_size']
        ).agg(
            avg=(metric, 'mean'),
            sd=(metric, lambda x: x.std(ddof=0)),
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

        final_df = pd.concat(
            [agg_df, df[df["downsample"] == True].copy()],
            ignore_index=True
        ).sort_values(['category', 'train_size'])

        # ----- Create and return the figure -----
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylim(-0.5, final_df.shape[0] - 0.5)

        for _, row in final_df.iterrows():
            _x = row['avg']
            _y = f"{row['category']}.{row['train_size']}"
            _xerr = row['sd']
            _color = 'red' if row['downsample'] else 'black'

            ax.errorbar(x=_x, y=_y, xerr=_xerr, fmt='o', color=_color)
            ax.text(_x, _y, f"{_x:.3f}±{_xerr:.3f}",
                    ha='center', va='bottom', color=_color)

        ax.plot([], [], 'o', color='red',   label='reduced')
        ax.plot([], [], 'o', color='black', label='full')

        ax.set_yticks(final_df.index)
        ax.set_yticklabels(final_df.apply(lambda r: f"{r['category']}.{r['train_size']}", axis=1), fontsize=9)
        ax.set_xlabel(metric.upper())

        ax.set_title(
            f"subset: {final_df['subset'].iloc[0]} || "
            f"model: {final_df['model'].iloc[0]} || "
            f"{results_df[results_df['downsample'] == False].groupby(['subset', 'category', 'model']).size().min()} folds || "
            f"{int(counts.max()/counts.min())} random seeds",
            fontsize=10,
            x=0.35
        )

        ax.legend(bbox_to_anchor=(1, 1.2))
        ax.grid(alpha=0.5)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        return fig