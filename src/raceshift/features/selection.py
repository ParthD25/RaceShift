from __future__ import annotations

from typing import Iterable

from .full_context import feature_contract, feature_taxonomy

ABLATION_GROUPS = ("static_categorical", "dynamic_numeric", "temporal_numeric", "historical_numeric")


def select_features(columns: Iterable[str], history: int = 5, drop_groups: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    """Numeric and categorical model inputs present in `columns`, minus any ablated groups."""
    taxonomy = feature_taxonomy(history)
    dropped: set[str] = set()
    for group in drop_groups:
        if group not in ABLATION_GROUPS:
            raise ValueError(f"Unknown feature group {group!r}. Choose from {ABLATION_GROUPS}")
        dropped.update(taxonomy[group])
    available = set(columns)
    contract = feature_contract(history)
    numeric = [c for c in contract.numeric if c in available and c not in dropped]
    categorical = [c for c in contract.categorical if c in available and c not in dropped]
    return numeric, categorical


def drop_sparse_features(train, numeric: list[str], min_coverage: float = 0.05) -> tuple[list[str], list[str]]:
    """Remove numeric features that are almost never observed in the training split.

    A feature that is missing in 95%+ of training rows is imputed to a near-constant, so
    its scaler variance is tiny and any real value seen later becomes an enormous
    standardized outlier. Returns (kept, dropped).
    """
    kept, dropped = [], []
    for col in numeric:
        coverage = float(train[col].notna().mean()) if len(train) else 0.0
        (kept if coverage >= min_coverage else dropped).append(col)
    return kept, dropped
