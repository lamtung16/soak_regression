import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

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


all_models = {
    "featureless": featureless_model,
    "linear": linear_model,
    "linear_BasExp": linear_BasExp_model,
    "tree": treeCV_model
}


class SOAKFold:
    def __init__(self, n_splits=5, seed=123):
        self.n_splits = n_splits
        self.seed = seed
    
    def split_idx(self, subset_vec):
        splits = []
        n = len(subset_vec)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        all_idx = np.arange(n)
        for fold_id, (fold_train_idx, fold_test_idx) in enumerate(kf.split(range(n))):
            for test_subset in np.unique(subset_vec):
                test_subset_idx = np.where(subset_vec == test_subset)[0]
                test_idx = np.intersect1d(test_subset_idx, fold_test_idx)
                train_idx_dict = {
                    "same":  test_subset_idx,
                    "other": np.setdiff1d(all_idx, test_subset_idx),
                    "all":   all_idx,
                }
                for category, train_idx in train_idx_dict.items():
                    splits.append((test_subset, category, fold_id + 1, np.intersect1d(fold_train_idx, train_idx), test_idx))
        return splits

    @staticmethod
    def model_eval(X_train, y_train, X_test, y_test, model='featureless'):
        return all_models[model](X_train, y_train, X_test, y_test)