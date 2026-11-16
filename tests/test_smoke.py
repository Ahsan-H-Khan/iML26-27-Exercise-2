"""Smoke tests that run offline on a synthetic dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from iml_feature_effects import (
    Dataset,
    EffectProperty,
    PartialDependence,
    Roughness,
    SubsampledRegressor,
    evaluate,
)


class _ConstantProperty(EffectProperty):
    name = "constant"

    def score(self, curve):
        return 1.0


@pytest.fixture
def dataset() -> Dataset:
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=0)
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


def test_evaluate_returns_one_row_per_dataset_model_and_feature(dataset):
    models = {"linear": LinearRegression(), "subsampled": SubsampledRegressor(LinearRegression(), n_samples=20)}
    effect = PartialDependence(resolution=10)
    scores, curves = evaluate([(dataset, ["x0", "x1"])], models, effect, [_ConstantProperty()])
    assert len(scores) == len(models) * 2
    assert set(scores.columns) == {"dataset", "model", "feature", "train_r2", "test_r2", "constant"}
    assert set(curves.columns) == {"dataset", "model", "feature", "value", "effect"}


def test_partial_dependence_of_linear_model_is_linear_and_centered(dataset):
    model = LinearRegression().fit(dataset.X, dataset.y)
    curve = PartialDependence(resolution=10).explain(model, dataset.X, "x0")
    assert len(curve) == 10
    assert curve["effect"].mean() == pytest.approx(0, abs=1e-8)
    slope = np.polyfit(curve["value"], curve["effect"], 1)[0]
    assert slope == pytest.approx(model.coef_[0])
    assert Roughness().score(curve) == pytest.approx(0, abs=1e-8)


def test_subsampled_regressor_uses_only_n_samples(dataset):
    model = SubsampledRegressor(LinearRegression(), n_samples=20, random_state=0).fit(dataset.X, dataset.y)
    assert model.estimator_.n_features_in_ == dataset.X.shape[1]
    # A linear model on 20 noisy points of a 4-feature linear problem still fits, but differs from the full fit.
    full = LinearRegression().fit(dataset.X, dataset.y)
    assert not np.allclose(model.estimator_.coef_, full.coef_)
