import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from scipy import stats
import numpy as np


def forward_selection(X, y, n_iter=20, stop=None):
    X = X.values if isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series)) else X
    y = y.values if isinstance(y, (pd.core.frame.DataFrame, pd.core.series.Series)) else y
    n_iter = X.shape[1] if n_iter > X.shape[1] else n_iter
    columns = list(range(X.shape[1]))
    steps = {}
    var_selected = []

    for i in range(n_iter):
        coeffs = []
        p_values = []
        t_stats = []

        # Do the algo for each column without the previous one with the smallest p-value
        for index in columns:
            if index in var_selected:
                p_values.append(np.NaN)
                t_stats.append(np.NaN)
            else:
                # we compute the OLS
                X_tilde = X[:, index].reshape(-1, 1)
                n = X_tilde.shape[0]
                rang_x = X_tilde.shape[1] + 1
                linear_model = LinearRegression(fit_intercept=True).fit(X_tilde, y)
                gram_matrix = X_tilde.T.dot(X_tilde)
                var = np.sum(np.square(y - linear_model.predict(X_tilde))) / (n - rang_x)
                # Apply t_test
                t_stat = abs(linear_model.coef_[0]) / (np.sqrt(var) * np.sqrt(np.linalg.inv(gram_matrix)))
                p_value = 1 - stats.t.cdf(t_stat[0][0], df=n - rang_x)
                p_values.append(p_value)
                t_stats.append(t_stat[0][0])

        # Search var X with the smallest p_value
        index_var = np.array(p_values).argsort()[0]
        if stop is not None:
            p_value = np.array(p_values)[index_var]
            if p_value > 0.1:
                return steps, var_selected

        X_tilde = X[:, index_var].reshape(-1, 1)
        linear_model = LinearRegression(fit_intercept=True).fit(X_tilde, y)
        y = y - linear_model.predict(X_tilde)

        # save the step
        steps[i] = {'p_values': np.array(p_values), 't_stat': np.array(t_stats)}
        var_selected.append(index_var)

    return steps, var_selected


class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median',
                 index=None, verbose=True):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        self.index = index
        self.verbose = verbose

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = SimpleImputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        X_reduce = ReduceVIF.calculate_vif(X, self.thresh, verbose=self.verbose)
        return X_reduce if self.index is None else X_reduce.set_index(self.index)

    @staticmethod
    def calculate_vif(X, thresh=5.0, verbose=True):
        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                if verbose:
                    print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X
