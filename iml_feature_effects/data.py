"""Dataset loading.

Every dataset is returned as a :class:`Dataset` with numeric, imputed features so that all
models in :mod:`iml_feature_effects.models` can be fit on it directly. Add further loaders
(e.g. other OpenML or Kaggle datasets) following the pattern of :func:`load_bike_sharing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

# OpenML downloads are cached here, relative to the current working directory.
DATA_HOME = Path("data")


@dataclass(frozen=True)
class Dataset:
    """A preprocessed tabular regression dataset.

    :param name: Human-readable dataset name, used in result tables and plots.
    :param X: Numeric feature matrix without missing values.
    :param y: Target vector.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def load_bike_sharing(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the hourly Bike Sharing Demand dataset from OpenML (data id 42712).

    The target is the number of rented bikes per hour. The download is cached in ``data_home``, so
    network access is only needed on the first call.

    :param data_home: Cache directory for the download, relative paths are resolved against the
        current working directory.
    :return: The preprocessed Bike Sharing dataset.
    :rtype: Dataset
    """
    bunch = fetch_openml(data_id=42712, as_frame=True, parser="liac-arff", data_home=data_home)
    # This OpenML version already excludes ``casual`` and ``registered``, which sum to the target.
    X = pd.get_dummies(bunch.data, drop_first=True, dtype=float)
    y = bunch.target.astype(float).rename("count")
    return Dataset(name="bike_sharing", X=X.astype(float), y=y)
