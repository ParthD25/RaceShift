from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Train-only preprocessing shared by RaceShift FFR and every learned baseline.

    Fit this on the training split only. Validation and test rows are transformed with
    the statistics learned from training rows, never refitted.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    # A missing category is information (no compound recorded for a legacy-tier lap), so it
    # becomes its own one-hot column instead of borrowing the most frequent training value.
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop", sparse_threshold=0.0)
