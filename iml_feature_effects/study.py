"""Reproducible paired training study and simulator-referenced effects."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .operators import Operator
from .physics import BOUNDS, encode, qoi, sample, solve

CONFIGS = {
    "underfit": dict(width=2, modes=1, depth=1, steps=650, wd=0.1),
    "well_fit": dict(width=16, modes=8, depth=2, steps=1400, wd=0.002),
    "overfit": dict(width=48, modes=16, depth=3, steps=2200, wd=0.0),
}


def predict(model, x):
    with torch.no_grad():
        return np.concatenate([model(z).numpy() for z in torch.from_numpy(x).split(256)])


def errors(system, pred, target):
    rel = np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-8)
    return dict(
        rel_l2=float(rel.mean()),
        mse=float(np.mean((pred - target) ** 2)),
        qoi_rmse=float(np.sqrt(np.mean((qoi(system, pred) - qoi(system, target)) ** 2))),
    )


def train(system, repeat, root, steps_scale=1.0):
    # One common training subset per repeat; fixed disjoint validation/eval/test pools.
    pool = sample(system, 512, 10)
    ix = np.random.default_rng(100 + repeat).choice(len(pool), 48, replace=False)
    parts = {
        "train": pool[ix],
        "val": sample(system, 128, 20),
        "eval": sample(system, 128, 30),
        "test": sample(system, 128, 50),
    }
    np.savez(root / f"{system}_data_{repeat}.npz", train_indices=ix, **parts)
    xx = {k: encode(system, p) for k, p in parts.items()}
    yy = {k: solve(system, p).astype("float32") for k, p in parts.items()}
    x, y = torch.from_numpy(xx["train"]), torch.from_numpy(yy["train"])
    # Fixed scale learned on training targets, shared by every regime in this repeat.
    scale = float(np.std(yy["train"]))
    rows = []
    for regime, base_cfg in CONFIGS.items():
        cfg = dict(base_cfg)
        if system == "oscillator" and regime == "well_fit":
            cfg = dict(width=8, modes=8, depth=1, steps=3500, wd=0.0, linear=True)
        if system == "oscillator" and regime == "overfit":
            cfg = dict(width=48, modes=16, depth=3, steps=3500, wd=0.0, cosine=True)
        path = root / f"{system}_{regime}_{repeat}.pt"
        if path.exists():
            payload = torch.load(path, weights_only=True)
            if payload["config"] != cfg:
                raise ValueError(f"Checkpoint configuration mismatch: {path}; use a fresh output directory")
            if payload.get("steps_scale", 1.0) != steps_scale:
                raise ValueError("Checkpoint training budget mismatch; use a fresh output directory")
            restored = Operator(system, cfg["width"], cfg["modes"], cfg["depth"], linear=cfg.get("linear", False))
            restored.load_state_dict(payload["state"])
            restored.eval()
            for score in payload["scores"]:
                if score["split"] == "test":
                    score.update(errors(system, predict(restored, xx["test"]) * payload["scale"], yy["test"]))
            torch.save(payload, path)
            rows.extend(payload["scores"])
            continue
        torch.manual_seed(1000 + repeat)
        model = Operator(system, cfg["width"], cfg["modes"], cfg["depth"], linear=cfg.get("linear", False))
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=cfg["wd"])
        history = []
        best = float("inf")
        state = None
        best_step = 0
        t0 = time.time()
        steps = int(cfg["steps"] * steps_scale)
        for step in range(steps):
            if cfg.get("cosine", False):
                for group in opt.param_groups:
                    group["lr"] = 0.002 * (0.01 + 0.99 * (1 + np.cos(np.pi * step / max(steps - 1, 1))) / 2)
            model.train()
            opt.zero_grad()
            loss = torch.mean((model(x) - y / scale) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            if step % 25 == 0 or step == steps - 1:
                model.eval()
                v = np.mean((predict(model, xx["val"]) - yy["val"] / scale) ** 2)
                history.append(dict(step=step, train_mse=float(loss.detach()), val_mse=float(v)))
                if v < best:
                    best = v
                    state = copy.deepcopy(model.state_dict())
                    best_step = step
        selected = best_step if regime == "well_fit" else steps - 1
        if regime == "well_fit":
            model.load_state_dict(state)
        model.eval()
        scores = []
        for split in ["train", "val", "test"]:
            scores.append(
                dict(
                    system=system,
                    regime=regime,
                    repeat=repeat,
                    split=split,
                    selected_step=selected,
                    parameters=sum(p.numel() for p in model.parameters()),
                    seconds=time.time() - t0,
                    **errors(system, predict(model, xx[split]) * scale, yy[split]),
                )
            )
        torch.save(
            dict(state=model.state_dict(), config=cfg, scale=scale, scores=scores, steps_scale=steps_scale), path
        )
        pd.DataFrame(history).to_csv(root / f"{system}_{regime}_{repeat}_history.csv", index=False)
        rows.extend(scores)
        print(system, repeat, regime, [(r["split"], round(r["rel_l2"], 4)) for r in scores], flush=True)
    return rows


def effects(system, predictor, background, grid, edges):
    n, m = len(background), len(grid)
    intervention = np.repeat(background, m, axis=0)
    intervention[:, 0] = np.tile(grid, n)
    ice = predictor(intervention).reshape(n, m)
    # All effects are shape-centered with the same grid-average convention.
    ice = ice - ice.mean(axis=1, keepdims=True)
    local = []
    counts = []
    bins = np.clip(np.searchsorted(edges, background[:, 0], side="right") - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        p = background[bins == b].copy()
        counts.append(len(p))
        p[:, 0] = edges[b]
        lo = predictor(p)
        p[:, 0] = edges[b + 1]
        hi = predictor(p)
        local.append(float(np.mean(hi - lo)))
    ale = np.r_[0, np.cumsum(local)]
    # Empirical centering at each background observation, identical for model and oracle.
    ale -= np.mean(np.interp(background[:, 0], edges, ale))
    return dict(ice=ice, pdp=ice.mean(0), ale=ale, counts=np.array(counts))


def explain(system, repeats, root):
    bg = sample(system, 128, 30)
    lo, hi = BOUNDS[system][0]
    grid = np.linspace(lo, hi, 41)
    edges = np.quantile(bg[:, 0], np.linspace(0, 1, 9))
    sigma = float(np.std(qoi(system, solve(system, bg))))

    def oracle(p):
        return qoi(system, solve(system, p))

    ref = effects(system, oracle, bg, grid, edges)
    np.savez(root / f"{system}_oracle.npz", grid=grid, edges=edges, scale=sigma, **ref)
    rows = []
    for regime in CONFIGS:
        curves = {k: [] for k in ["ice", "pdp", "ale"]}
        for r in range(repeats):
            payload = torch.load(root / f"{system}_{regime}_{r}.pt", weights_only=True)
            c = payload["config"]
            model = Operator(system, c["width"], c["modes"], c["depth"], linear=c.get("linear", False))
            model.load_state_dict(payload["state"])
            model.eval()

            def pred(p, model=model, scale=payload["scale"]):
                return qoi(system, predict(model, encode(system, p)) * scale)

            e = effects(system, pred, bg, grid, edges)
            for method in curves:
                v = e[method]
                truth = ref[method]
                curves[method].append(v)
                axis = (edges - edges[0]) / (edges[-1] - edges[0]) if method == "ale" else (grid - lo) / (hi - lo)
                slopes = np.diff(v / sigma, axis=-1) / np.diff(axis)
                # Native, possibly nonuniform grid. ALE curvature is descriptive bin-dependent.
                curvature = np.diff(slopes, axis=-1) / ((axis[2:] - axis[:-2]) / 2)
                rows.append(
                    dict(
                        system=system,
                        regime=regime,
                        repeat=r,
                        method=method,
                        effect_nrmse=float(np.sqrt(np.mean((v - truth) ** 2)) / sigma),
                        slope=float(np.mean(abs(slopes))),
                        curvature=float(np.mean(abs(curvature))),
                    )
                )
        np.savez(
            root / f"{system}_{regime}_effects.npz",
            grid=grid,
            edges=edges,
            **{k: np.stack(v) for k, v in curves.items()},
        )
        for method, v in curves.items():
            dispersion = float(np.std(v, axis=0, ddof=1).mean() / sigma) if repeats > 1 else float("nan")
            for row in rows:
                if row["regime"] == regime and row["method"] == method:
                    row["refit_dispersion"] = dispersion
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/study"))
    args = parser.parse_args()
    if args.repeats < 1 or args.steps_scale <= 0:
        parser.error("repeats and steps-scale must be positive")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    scores = []
    metrics = []
    for system in BOUNDS:
        for r in range(args.repeats):
            scores.extend(train(system, r, root, args.steps_scale))
            pd.DataFrame(scores).to_csv(root / "scores.csv", index=False)
        metrics.extend(explain(system, args.repeats, root))
        pd.DataFrame(metrics).to_csv(root / "effect_metrics.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                repeats=args.repeats,
                steps_scale=args.steps_scale,
                configurations=CONFIGS,
                source_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
                },
                torch=torch.__version__,
                numpy=np.__version__,
                oscillator_overfit=dict(width=48, modes=16, depth=3, steps=3500, wd=0.0, cosine=True),
                oscillator_well_fit=dict(width=8, modes=8, depth=1, steps=3500, wd=0.0, linear=True),
                final_test_seed=50,
                pilot_test_seed=40,
                note=(
                    "Paired training resamples; one matched initialization seed per resample. "
                    "No independent seed factorial."
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
