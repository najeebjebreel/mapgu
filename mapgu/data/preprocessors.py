"""Sklearn-compatible column selectors and encoders for tabular preprocessing."""

from __future__ import annotations

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """
    Select columns from a pandas DataFrame by dtype group.
    baseline_experiments.py uses ColumnsSelector(type="int") and type="object".
    """
    def __init__(self, type: str):
        self.type = str(type)

    def fit(self, X, y=None):
        # no-op
        return self

    def __sklearn_is_fitted__(self):
        # Stateless transformer.
        return True

    def transform(self, X):
        if not hasattr(X, "select_dtypes"):
            raise TypeError("ColumnsSelector expects a pandas DataFrame.")
        t = self.type.lower()
        if t in ["int", "int64", "integer", "number", "numeric", "float"]:
            # select numeric types
            return X.select_dtypes(include=["number"])
        if t in ["object", "category", "cat", "string", "str"]:
            return X.select_dtypes(include=["object", "category", "string"])
        # fallback: try direct dtype include
        return X.select_dtypes(include=[self.type])


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values in specified categorical columns.
    Default strategy: most_frequent.
    """
    def __init__(self, columns=None, strategy: str = "most_frequent"):
        self.columns = list(columns) if columns is not None else None
        self.strategy = strategy
        self.imputer_ = None

    def fit(self, X, y=None):
        if not hasattr(X, "__getitem__"):
            raise TypeError("CategoricalImputer expects a pandas DataFrame-like input.")
        cols = self.columns if self.columns is not None else list(X.columns)
        self.columns_ = cols
        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(X[self.columns_])
        return self

    def transform(self, X):
        cols = getattr(self, "columns_", None)
        if cols is None:
            cols = self.columns if self.columns is not None else list(X.columns)
        Xc = X.copy()
        Xc[cols] = self.imputer_.transform(Xc[cols])
        return Xc

    def __sklearn_is_fitted__(self):
        return getattr(self, "imputer_", None) is not None and hasattr(self, "columns_")


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encode categorical columns.
    baseline_experiments.py calls: CategoricalEncoder(train_data, test_data, dropFirst=True)
    We'll fit on union of train+test categories to avoid unseen-category issues.

    Output is a dense numpy array (for FeatureUnion compatibility).
    """
    def __init__(self, train_df=None, test_df=None, dropFirst: bool = True):
        self.train_df = train_df
        self.test_df = test_df
        self.dropFirst = bool(dropFirst)
        self.encoder_ = None
        self.cols_ = None

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            raise TypeError("CategoricalEncoder expects a pandas DataFrame.")
        self.cols_ = list(X.columns)

        drop = "first" if self.dropFirst else None
        self.encoder_ = OneHotEncoder(
            drop=drop,
            handle_unknown="ignore",
            sparse_output=False,
        )

        # Fit on union (train + test) if provided, else just X
        if self.train_df is not None and self.test_df is not None:
            # Make sure we only use these columns
            Xt = self.train_df[self.cols_].copy()
            Xv = self.test_df[self.cols_].copy()
            Xall = pd.concat([Xt, Xv], axis=0, ignore_index=True)
            self.encoder_.fit(Xall)
        else:
            self.encoder_.fit(X[self.cols_])

        return self

    def transform(self, X):
        if self.encoder_ is None:
            raise RuntimeError("CategoricalEncoder.transform called before fit.")
        return self.encoder_.transform(X[self.cols_])

    def __sklearn_is_fitted__(self):
        return getattr(self, "encoder_", None) is not None and getattr(self, "cols_", None) is not None
