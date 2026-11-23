"""Grid and ALE-bin sensitivity on the first paired refit; no retraining."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .operators import Operator
from .physics import BOUNDS, encode, qoi, sample, solve
from .study import CONFIGS, effects, predict


def run(root=Path("results/study")):
    torch.set_num_threads(2)
    rows = []
    for system in BOUNDS:
        bg = sample(system, 128, 30)
        sigma = np.std(qoi(system, solve(system, bg)))
        for size, bins in [(41, 8), (81, 8), (41, 16)]:
            grid = np.linspace(*BOUNDS[system][0], size)
            edges = np.quantile(bg[:, 0], np.linspace(0, 1, bins + 1))
            ref = effects(system, lambda p, system=system: qoi(system, solve(system, p)), bg, grid, edges)
            for regime in CONFIGS:
                payload = torch.load(root / f"{system}_{regime}_0.pt", weights_only=True)
                c = payload["config"]
                model = Operator(system, c["width"], c["modes"], c["depth"], linear=c.get("linear", False))
                model.load_state_dict(payload["state"])
                model.eval()

                def predictor(p, model=model, scale=payload["scale"], system=system):
                    return qoi(system, predict(model, encode(system, p)) * scale)

                e = effects(system, predictor, bg, grid, edges)
                for method in ["ice", "pdp", "ale"]:
                    axis = edges if method == "ale" else grid
                    axis = (axis - axis[0]) / (axis[-1] - axis[0])
                    for source, value in [("model", e[method]), ("simulator", ref[method])]:
                        slopes = np.diff(value / sigma, axis=-1) / np.diff(axis)
                        curvature = np.diff(slopes, axis=-1) / ((axis[2:] - axis[:-2]) / 2)
                        rows.append(
                            dict(
                                system=system,
                                regime=regime,
                                grid=size,
                                bins=bins,
                                method=method,
                                source=source,
                                min_bin_count=int(e["counts"].min()),
                                effect_nrmse=float(np.sqrt(np.mean((value - ref[method]) ** 2)) / sigma),
                                slope=float(abs(slopes).mean()),
                                curvature=float(abs(curvature).mean()),
                            )
                        )
    pd.DataFrame(rows).to_csv(root / "sensitivity.csv", index=False)


if __name__ == "__main__":
    run()
