"""Parameterized function-to-function datasets and independent analytic solvers."""

from __future__ import annotations

import numpy as np

BOUNDS = {
    "burgers": np.array([[0.06, 0.25], [0.5, 1.5], [-0.5, 0.5], [-np.pi, np.pi]]),
    "oscillator": np.array([[0.5, 3.5], [0.5, 1.5], [-np.pi, np.pi], [-0.5, 0.5]]),
}
NAMES = {
    "burgers": ["viscosity", "amplitude", "second_mode", "phase"],
    "oscillator": ["frequency", "amplitude", "phase", "initial_displacement"],
}


def sample(system, n, seed):
    rng = np.random.default_rng(seed)
    b = BOUNDS[system]
    return rng.uniform(b[:, 0], b[:, 1], (n, len(b)))


def coordinates(system, n=64):
    return np.linspace(0, 2 * np.pi, n, endpoint=False) if system == "burgers" else np.linspace(0, 12, n)


def encode(system, p, n=64):
    """Rebuild all signal and conditioning channels after a parameter intervention."""
    x = coordinates(system, n)
    if system == "burgers":
        signal = p[:, 1, None] * np.sin(x + p[:, 3, None]) + p[:, 2, None] * np.cos(2 * x - p[:, 3, None])
    else:
        signal = p[:, 1, None] * np.cos(p[:, 0, None] * x + p[:, 2, None])
    b = BOUNDS[system]
    pn = 2 * (p - b[:, 0]) / (b[:, 1] - b[:, 0]) - 1
    channels = [
        signal[:, None, :],
        np.broadcast_to(pn[:, :, None], (len(p), 4, n)),
        np.broadcast_to((x / x[-1])[None, None, :], (len(p), 1, n)),
    ]
    return np.concatenate(channels, axis=1).astype("float32")


def solve(system, p, n=64, internal=512):
    if system == "oscillator":
        t = coordinates(system, n)
        w, a, phase, x0 = (p[:, j, None] for j in range(4))
        damping, stiffness = 0.4, 4.0
        d = damping / 2
        wd = np.sqrt(stiffness - d * d)
        c = a * np.exp(1j * phase) / (stiffness - w * w + 1j * damping * w)
        b = x0 - c.real
        h = (-(1j * w * c).real + d * b) / wd
        return (c * np.exp(1j * w * t)).real + np.exp(-d * t) * (b * np.cos(wd * t) + h * np.sin(wd * t))
    # Cole-Hopf: u=-2 nu d_x log(phi), phi_t=nu phi_xx, periodic domain [0,2pi].
    x = coordinates(system, internal)
    nu, a, b, phase = (p[:, j, None] for j in range(4))
    potential = -a * np.cos(x + phase) + b / 2 * np.sin(2 * x - phase)
    logphi = -potential / (2 * nu)
    phi = np.exp(logphi - logphi.max(axis=1, keepdims=True))
    k = np.fft.rfftfreq(internal, 1 / internal)[None, :]
    spectrum = np.fft.rfft(phi) * np.exp(-nu * k * k * 1.0)
    heat = np.fft.irfft(spectrum, n=internal)
    derivative = np.fft.irfft(1j * k * spectrum, n=internal)
    if np.min(heat) <= 0:
        raise FloatingPointError("Insufficient Cole-Hopf precision")
    u = -2 * nu * derivative / heat
    if internal % n:
        raise ValueError("internal resolution must be a multiple of output resolution")
    return u[:, :: internal // n]


def qoi(system, y):
    if system == "oscillator":
        return np.sqrt(np.trapezoid(y * y, x=coordinates(system, y.shape[-1]), axis=-1) / 12)
    return np.sqrt(np.mean(y * y, axis=-1))
