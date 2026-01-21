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
    

    def split_idx(self, subset_vec):
        splits = []
        for subset_value in np.unique(subset_vec):
            same_idx = np.where(subset_vec == subset_value)[0]
            other_idx = np.where(subset_vec != subset_value)[0]
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            for fold_id, (train_idx, test_idx) in enumerate(kf.split(same_idx)):
                train_same_idx = same_idx[train_idx]
                test_same_idx = same_idx[test_idx]
                for category, train_idx_final in {
                    'same': train_same_idx,
                    'other': other_idx,
                    'all': np.concatenate([train_same_idx, other_idx])
                }.items():
                    splits.append([subset_value, category, fold_id + 1, train_idx_final, test_same_idx])
        return splits


    @staticmethod
    def model_eval(X_train, y_train, X_test, y_test, model='featureless'):
        return all_models[model](X_train, y_train, X_test, y_test)