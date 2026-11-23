"""Compact FNO and causal pole/residue Laplace neural operator.

The Laplace layer learns stable conjugate pole pairs and residues. Its inverse
Laplace kernel is applied by zero-padded causal convolution, not periodic FFT
convolution. This is a custom separable LNO, not a reproduction of a paper model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FourierLayer(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)
        self.local = nn.Conv1d(width, width, 1)

    def forward(self, x):
        xf = torch.fft.rfft(x)
        out = torch.zeros_like(xf)
        m = min(self.modes, xf.shape[-1])
        out[:, :, :m] = torch.einsum("bik,iok->bok", xf[:, :, :m], self.weight[:, :, :m])
        return self.local(x) + torch.fft.irfft(out, n=x.shape[-1])


class LaplaceLayer(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.decay = nn.Parameter(torch.full((width, modes), -1.5))
        self.frequency = nn.Parameter(torch.linspace(0.3, 4, modes).repeat(width, 1))
        self.residue = nn.Parameter(torch.randn(width, modes, 2) / (modes**0.5))
        self.mix = nn.Conv1d(width, width, 1)
        self.local = nn.Conv1d(width, width, 1)

    def forward(self, x):
        n = x.shape[-1]
        t = torch.linspace(0, 12, n, device=x.device)
        decay = F.softplus(self.decay)[:, :, None]
        angle = self.frequency[:, :, None] * t
        kernel = (
            torch.exp(-decay * t)
            * (self.residue[:, :, 0, None] * torch.cos(angle) + self.residue[:, :, 1, None] * torch.sin(angle))
        ).sum(1)
        # Trapezoidal convolution: half-weight the two integration endpoints.
        z = torch.fft.irfft(torch.fft.rfft(x, n=2 * n) * torch.fft.rfft(kernel, n=2 * n)[None], n=2 * n)[:, :, :n]
        z = (z - 0.5 * kernel[:, 0][None, :, None] * x - 0.5 * kernel[None] * x[:, :, :1]) * (12 / (n - 1))
        return self.local(x) + self.mix(z)


class Operator(nn.Module):
    def __init__(self, system, width, modes, depth, linear=False):
        super().__init__()
        self.linear = linear
        self.lift = nn.Conv1d(6, width, 1)
        layer = FourierLayer if system == "burgers" else LaplaceLayer
        self.layers = nn.ModuleList([layer(width, modes) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.Identity() if linear else nn.GELU(), nn.Conv1d(width, 1, 1)
        )

    def forward(self, x):
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x) if self.linear else F.gelu(layer(x))
        return self.project(x)[:, 0]
