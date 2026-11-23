"""Known-function and physical controls for the study's Co-12 diagnostics."""

import numpy as np
import pytest
import torch

from iml_feature_effects.co12 import (
    curves_at,
    direction_disagreement,
    interaction_residual,
    local_shape_gain,
    permute_channels,
    physical_probes,
)
from iml_feature_effects.operators import Operator
from iml_feature_effects.physics import BOUNDS, encode, qoi, sample, solve


@pytest.mark.parametrize("system,linear", [("burgers", False), ("oscillator", False), ("oscillator", True)])
def test_hidden_channel_reparameterization_preserves_function(system, linear):
    torch.manual_seed(71)
    model = Operator(system, 4, 3, 2, linear=linear).eval()
    equivalent = permute_channels(model, seed=34)
    data = torch.from_numpy(encode(system, sample(system, 8, 9)))
    with torch.no_grad():
        torch.testing.assert_close(model(data), equivalent(data), rtol=1e-5, atol=2e-5)


def test_interaction_residual_separates_additive_and_multiplicative_effects():
    z = np.linspace(-1, 1, 11)
    w = np.array([1.0, 2.0, 4.0])
    additive = z[None, :] + w[:, None]
    np.testing.assert_allclose(interaction_residual(additive), 0, atol=1e-15)
    multiplicative = z[None, :] * w[:, None]
    expected = (w - w.mean())[:, None] * (z - z.mean())[None, :]
    np.testing.assert_allclose(interaction_residual(multiplicative), expected, atol=1e-15)


def test_local_shape_gain_and_direction_use_reference_not_roughness():
    curves = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    assert local_shape_gain(curves, curves, np.array([0.01, 0.02]), 2.0) == 0.0
    axis = np.linspace(0, 1, 3)
    assert direction_disagreement(-curves, curves, axis, 2.0) == 1.0
    assert direction_disagreement(curves, curves, axis, 2.0) == 0.0


@pytest.mark.parametrize("system", ["burgers", "oscillator"])
def test_physical_coherence_probes_have_oracle_support(system):
    base, changed, factor, _ = physical_probes(system, n=8)
    limits = BOUNDS[system]
    assert np.all(base >= limits[:, 0]) and np.all(base <= limits[:, 1])
    assert np.all(changed >= limits[:, 0]) and np.all(changed <= limits[:, 1])
    grid = np.linspace(*limits[0], 9)

    def oracle(p):
        return qoi(system, solve(system, p))

    a = curves_at(system, oracle, base, grid)
    b = curves_at(system, oracle, changed, grid)
    # Translation aligns with the spatial grid; joint oscillator scaling is exact.
    tolerance = 1e-8 if system == "burgers" else 1e-12
    np.testing.assert_allclose(b, factor * a, rtol=tolerance, atol=tolerance)
