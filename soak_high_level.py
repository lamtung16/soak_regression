import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import ttest_ind

def split(subset_vec, n_splits=5, n_random_seeds=5, seed=123):
    """
    Perform SOAK splitting.

    Parameters
    ----------
    subset_vec : array-like
        Vector indicating subset/group membership.
    n_splits : int, default=5
        Number of folds.
    n_random_seeds: int, default=5
        Number of random seeds for downsampling
    seed : int, default=123
        Random seed for reproducibility.

    Returns
    -------
    list
        List of [test_subset, category, fold_id, random_seed, train_idx, test_idx]
    """
    
    subset_vec = np.asarray(subset_vec)

    if subset_vec.ndim != 1:
        raise ValueError("subset_vec must be a 1D array-like")

    n = subset_vec.shape[0]

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if n_splits > n:
        raise ValueError("n_splits cannot exceed number of samples")
    
    splits = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_idx = np.arange(n)
    for fold_id, (fold_train_idx, fold_test_idx) in enumerate(kf.split(range(n), subset_vec)):
        for test_subset in np.unique(subset_vec):
            test_subset_idx = np.where(subset_vec == test_subset)[0]
            test_idx = np.intersect1d(test_subset_idx, fold_test_idx)
            other_subset_idx = np.setdiff1d(all_idx, test_subset_idx)
            
            bigger_set = ""
            downsample_size = int(min(len(test_subset_idx), len(other_subset_idx))*(n_splits-1)/n_splits)
            downsample_subset_idx = []
            if abs(len(test_subset_idx) - len(other_subset_idx)) >= n_splits:
                bigger_set = "same" if len(test_subset_idx) > len(other_subset_idx) else "other"
                downsample_subset_idx = max(test_subset_idx, other_subset_idx, key=len)

            train_idx_dict = {
                "same":              test_subset_idx,
                "other":             other_subset_idx,
                f"{bigger_set}-ds":  downsample_subset_idx,
                "all":               all_idx,
                "all-ds":            all_idx,
            }

            for category, train_idx in train_idx_dict.items():
                if len(train_idx) > 0:
                    if "ds" in category:
                        for random_seed in range(n_random_seeds):
                            splits.append((test_subset, category, fold_id + 1, random_seed + 1, sorted(np.random.choice(np.intersect1d(fold_train_idx, train_idx), size=downsample_size, replace=False)), test_idx))
                    else:
                        splits.append((test_subset, category, fold_id + 1, 0, np.intersect1d(fold_train_idx, train_idx), test_idx))
                    
    return splits



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
        ('ridge', RidgeCV(alphas=np.logspace(-2, 2, 20), cv=4))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate(y_pred, y_test)


def treeCV_model(X_train, y_train, X_test, y_test):
    param_grid = {'max_depth': np.arange(2, 41, 2)}
    grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=4, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)
    return evaluate(y_pred, y_test)


all_models = {
    "featureless": featureless_model,
    "linear": linear_model,
    "tree": treeCV_model
}

class SOAK:
    def __init__(self, df, subset_col, target_col):
        """
        SOAK class.

        Parameters
        ----------
        df : pandas dataframe
            Tabular dataset
        subset_col: str
            Name of subset column for SOAK spliting
        target_col: str
            Name of target column

        """
        self.df = df
        self.subset_col = subset_col
        self.target_col = target_col
        self.results_df = None
    
    def analyze(self, model_list = ["tree"], n_splits=10, n_random_seeds=10, log_target=False, seed=123):
        """
        SOAK analyze.

        Parameters
        ----------
        model_list : list, default = ["featureless", "tree"]
            List of train models, subset of ["featureless", "linear", "tree"]
        n_splits : int, default=10
            Number of folds for each subset.
        n_random_seeds: int, default=10
            Number of random seeds for downsampling
        log_target: boolean, default=False
            Transform target using log or not
        seed : int, default=123
            Random seed for reproducibility.

        """
        X = np.array(self.df.drop(columns=[self.subset_col, self.target_col]).select_dtypes(include=[np.number]))
        y = self.df[self.target_col]
        if log_target:
            y = np.log(y)
        y = (y - np.mean(y)) / np.std(y)
        subset_vec = self.df[self.subset_col]
        results = []
        for test_subset, category, fold_id, random_seed, train_idx, test_idx in split(subset_vec, n_splits, n_random_seeds, seed):
            for model in model_list:
                rmse, mae = self.model_eval(X[train_idx], y[train_idx], X[test_idx], y[test_idx], model)
                results.append({
                                "subset": test_subset,
                                "category": category,
                                "fold_id": fold_id,
                                "model": model,
                                "seed_id": random_seed,
                                "train_size": len(train_idx),
                                "test_size": len(test_idx),
                                "rmse": rmse,
                                "mae": mae,
                            })
        self.results_df = pd.DataFrame(results)

    @staticmethod
    def model_eval(X_train, y_train, X_test, y_test, model):
        return all_models[model](X_train, y_train, X_test, y_test)


    def visualize(self, subset_value=None, model=None, metric="rmse", figsize=(15, 3)):
        """
        SOAK visualize.

        Parameters
        ----------
        subset_value : str, default is the last seen subset value
            Value of test subset
        model : str, default is the last seen training model
            Trained model
        metric: str, default='rmse'
            Metric, it can be either 'rmse' or 'mae'
        figsize: tuple, default=(15, 3)
            Size of figure

        """
        if subset_value == None:
            subset_value = np.unique(self.df[self.subset_col])[-1]
        if model == None:
            model = np.unique(self.results_df['model'])[-1]
        
        def pval(cat1, cat2):
            x = df.loc[df["category"] == cat1, metric]
            y = df.loc[df["category"] == cat2, metric]
            _, p = ttest_ind(x, y, equal_var=False)
            return p

        df = self.results_df[(self.results_df['subset'] == subset_value) &(self.results_df["model"] == model)].copy()

        cats = set(df["category"].unique())
        sorted_cats_full = ["other", "same", "all"]
        sorted_cats_ds = ["other", "same", "all-ds"]
        if "same-ds" in cats:
            sorted_cats_ds = ["other", "same-ds", "all-ds"]
        if "other-ds" in cats:
            sorted_cats_ds = ["other-ds", "same", "all-ds"]
        
        dfs = [None, None]
        for i, sorted_cats in enumerate([sorted_cats_full, sorted_cats_ds]):   
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
            final['category'] = final['category'].str.replace('-ds', '', regex=False)
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
                text = f"{mean:.5f} ± {sd:.5f}" if i % 2 == 0 else f"P = {row['p_value']:.4f}" if row['p_value'] > 0.0001 else "P < 0.0001"
                marker_size = 4 if i % 2 == 0 else 0
                ax.errorbar(mean, y, xerr=sd, fmt="o", color=color, markersize=marker_size)
                ax.text(mean, y + 0.15, text, ha="center", va="bottom", fontsize=8)

            # y-axis formatting
            ax.set_yticks([y_pos[c] for c in category_order])
            ax.set_yticklabels(category_order, fontsize=9)
            ax.set_ylim(-0.5, len(category_order) - 0.2)

            # labels & title
            ax.set_title("sample size: " + ("full" if idx==0 else f"{int(dfs[1]['train_size'].max())}"), fontsize=10)
            ax.grid(alpha=0.5)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ax.tick_params(axis='x', labelsize=9)

        fig.supxlabel(f"{metric.upper()} (mean ± 2sd) | test subset: {subset_value} | model: {model} | {set(self.results_df['fold_id']).__len__()} test folds | {len(set(self.results_df['seed_id']))-1} downsample random seeds", fontsize=11)
        fig.tight_layout()
        plt.close(fig)
        return fig