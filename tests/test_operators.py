"""Independent physics and operator/effect checks, run without network access."""

import numpy as np
import torch
from scipy.integrate import solve_ivp

from iml_feature_effects.operators import LaplaceLayer, Operator
from iml_feature_effects.physics import coordinates, encode, qoi, sample, solve
from iml_feature_effects.study import effects


def test_oscillator_matches_independent_ode_solver():
    p = sample("oscillator", 3, 123)
    t = coordinates("oscillator")
    actual = solve("oscillator", p)
    for i, (w, a, phase, x0) in enumerate(p):
        sol = solve_ivp(
            lambda t, z, a=a, w=w, phase=phase: [z[1], a * np.cos(w * t + phase) - 0.4 * z[1] - 4 * z[0]],
            [0, 12],
            [x0, 0],
            t_eval=t,
            rtol=1e-10,
            atol=1e-11,
        )
        np.testing.assert_allclose(actual[i], sol.y[0], atol=2e-8)


def test_burgers_resolution_and_dissipation():
    p = sample("burgers", 12, 123)
    y = solve("burgers", p, internal=512)
    np.testing.assert_allclose(y, solve("burgers", p, internal=1024), atol=1e-7)
    u0 = encode("burgers", p)[:, 0]
    assert np.all(qoi("burgers", y) < qoi("burgers", u0))
    np.testing.assert_allclose(y.mean(1), 0, atol=1e-6)


def test_burgers_matches_independent_spectral_time_integrator():
    p = np.array([[0.15, 1.0, 0.2, 0.3]])
    n = 128
    x = coordinates("burgers", n)
    k = np.fft.fftfreq(n, 1 / n)
    initial = p[0, 1] * np.sin(x + p[0, 3]) + p[0, 2] * np.cos(2 * x - p[0, 3])

    def rhs(t, u):
        ux = np.fft.ifft(1j * k * np.fft.fft(u)).real
        uxx = np.fft.ifft(-k * k * np.fft.fft(u)).real
        return -u * ux + p[0, 0] * uxx

    result = solve_ivp(rhs, [0, 1], initial, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(solve("burgers", p, n=n)[0], result.y[:, -1], atol=2e-7)


def test_laplace_layer_is_causal():
    torch.manual_seed(0)
    layer = LaplaceLayer(3, 4)
    x = torch.randn(2, 3, 64)
    changed = x.clone()
    changed[:, :, 32:] += 3
    torch.testing.assert_close(layer(x)[:, :, :32], layer(changed)[:, :, :32], atol=2e-5, rtol=1e-5)


def test_spectral_and_laplace_parameters_receive_gradients():
    for system in ["burgers", "oscillator"]:
        model = Operator(system, 4, 3, 2)
        x = torch.from_numpy(encode(system, sample(system, 3, 9)))
        model(x).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_effects_recover_linear_parameter_effect():
    bg = sample("oscillator", 128, 9)
    grid = np.linspace(0.5, 3.5, 41)
    edges = np.quantile(bg[:, 0], np.linspace(0, 1, 9))
    e = effects("oscillator", lambda p: 2 * p[:, 0] + p[:, 1], bg, grid, edges)
    np.testing.assert_allclose(e["pdp"], 2 * (grid - grid.mean()), atol=1e-12)
    np.testing.assert_allclose(e["ale"], 2 * (edges - bg[:, 0].mean()), atol=1e-12)
    assert np.min(e["counts"]) == 16
