# Neural operator feature effects from underfit to overfit

This study examines how model fit changes ICE, PDP and ALE explanations of two physical systems: a Fourier neural operator (FNO) for periodic viscous Burgers flow, and a custom Laplace neural operator for a forced, damped oscillator. Models predict complete functions; explanations describe the RMS of the predicted solution. Independent simulators provide reference explanations.

## Reproduce

Run commands from the repository root. Python 3.10 or newer is required; the checked run uses Python 3.12 and CPU PyTorch. `uv.lock` pins dependencies; PyTorch uses its official CPU wheel index.

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run python scripts/run_experiment.py --repeats 3
uv run python -m iml_feature_effects.sensitivity
uv run python -m iml_feature_effects.report
```

The full experiment trains 18 compact models on CPU. Checkpoints resume completed configurations; use `--output results/new_run` for a fresh experiment. Do not mix checkpoints from different code revisions. A shorter smoke experiment can use `--repeats 1 --steps-scale 0.05 --output results/smoke`; it does not establish fit regimes. Sensitivity and reporting modules default to `results/study`.

In a restricted container, set writable cache locations, for example `UV_CACHE_DIR=/tmp/iml-uv` and `MPLCONFIGDIR=/tmp/iml-mpl`. No dataset download is needed.

## Datasets and operator inputs

Parameters are sampled independently and uniformly over the ranges below. These are two synthetic datasets restricted to parameterized input-function families, rather than benchmarks of generalization to arbitrary functions.

### Periodic viscous Burgers

We solve `u_t + u u_x = nu u_xx` on `[0, 2 pi)`, with periodic boundary conditions and

`u(x,0) = a sin(x + phase) + b cos(2x - phase)`.

The input is the initial function, four normalized parameter channels, and the spatial coordinate. The output is the solution at `T=1` on 64 spatial points. The Cole-Hopf transformation evolves the associated heat equation in Fourier space on 512 points before subsampling. Tests compare with 1024 points and an independent spectral method-of-lines solver.

| Parameter | Range |
|---|---|
| Viscosity nu, primary explanatory feature | 0.06 to 0.25 |
| First-mode amplitude a | 0.5 to 1.5 |
| Second-mode coefficient b | -0.5 to 0.5 |
| Phase | -pi to pi |

The scalar quantity is spatial RMS at `T=1`. The model is an FNO with learned complex Fourier multipliers, local channel mixing, GELU nonlinearities and pointwise lifting/projection.

### Forced damped oscillator

We solve `x'' + 0.4 x' + 4 x = A cos(omega t + phase)`, with `x(0)=x0` and `x'(0)=0`, on `[0,12]`. The input contains the forcing function, normalized parameter channels and time. The output has 64 time points. An analytic particular solution plus the homogeneous transient provides targets; tests compare it with a high-accuracy independent ODE integrator.

| Parameter | Range |
|---|---|
| Forcing frequency omega, primary explanatory feature | 0.5 to 3.5 |
| Forcing amplitude A | 0.5 to 1.5 |
| Forcing phase | -pi to pi |
| Initial displacement x0 | -0.5 to 0.5 |

The scalar quantity is time RMS over `[0,12]`, using trapezoidal quadrature. The finite window includes transients; this is not a steady-state resonance-amplitude experiment.

The custom separable Laplace operator learns stable conjugate poles `-softplus(decay) +/- i frequency` and residues. Inverse-Laplace kernels are applied using zero-padded causal convolution with trapezoidal quadrature, with local and channel-mixing layers. This is a compact implementation of the pole/residue operator idea, not a reproduction of the published LNO architecture. The well-fit oscillator uses a linear stack, reflecting the linear dynamics; the other regimes have nonlinear stacks. Therefore this comparison changes inductive bias as well as capacity.

## Controlled training protocol

Each refit samples 48 functions without replacement from the same 512-function training pool. All three regimes receive exactly the same training subset. There are separate fixed validation, explanation-background and final-test sets of 128 functions each. Every function is a sample; spatial/time points are never split between sets. Training targets have no added noise.

Training-pool seed: 10; subset seeds: 100 through 102; initialization seeds: 1000 through 1002, paired across regimes. Validation seed: 20; explanation seed: 30. Pilot test seed 40 was inspected during development; the final report uses a fresh test seed 50, after selecting configurations on validation.

| Configuration | Width | Modes/pole pairs per channel | Layers | Steps | Weight decay |
|---|---:|---:|---:|---:|---:|
| Underfit, either system | 2 | 1 | 1 | 650 | 0.1 |
| Well-fit FNO | 16 | 8 | 2 | 1400 | 0.002 |
| Well-fit linear Laplace operator | 8 | 8 | 1 | 3500 | 0 |
| Overfit FNO | 48 | 16 | 3 | 2200 | 0 |
| Overfit nonlinear Laplace operator | 48 | 16 | 3 | 3500 | 0 |

The nonlinear Laplace model uses cosine learning-rate decay from 0.002 to 0.00002 to avoid the late optimization instability seen in the pilot. Other configurations use a constant learning rate. Training uses full-batch AdamW with initial learning rate 0.002, gradient clipping at 5, and a fixed training-target standard-deviation scale shared across regimes in each refit. Well-fit checkpoints minimize validation MSE, checked every 25 steps; the other regimes use their final checkpoint. Measured errors determine whether these intended regimes were achieved. `regime_checks.json` records aggregate comparisons; histories retain all checked training and validation losses.

Relative field L2 is averaged over functions. We also report field MSE and scalar-QoI RMSE. Three paired refits change both training subset and initialization; their dispersion combines those sources of variability. It is not an independent resample-by-seed factorial experiment, and standard-deviation bands are descriptive rather than confidence intervals.

## Feature effects and metrics

All interventions rebuild the forcing/initial function and every conditioning channel. PDP and ICE use the same 128 background instances and 41 uniformly spaced primary-feature values. ICE is centered by each curve's grid mean, and PDP is its average. We retain raw prediction metrics separately so shape centering cannot conceal output offsets.

First-order ALE uses eight quantile bins from the fixed background, with 16 instances per bin. Each local effect averages the prediction difference at the bin endpoints for instances in that bin. The accumulated endpoint curve is centered by its linearly interpolated mean over background observations. Model and simulator use exactly the same finite-bin convention.

For every explanation method, normalized effect RMSE compares the model curve against the corresponding simulator curve; ICE averages over both grid points and instances. The denominator is the standard deviation of simulator QoI values on the fixed background. Between-refit dispersion is the mean pointwise sample standard deviation divided by that same scale. Smaller values indicate greater stability.

Slope magnitude and curvature use input-range and simulator-QoI normalization, including nonuniform spacing for ALE. They describe response shape, not mathematical continuity or explanation correctness. A physical resonance can have substantial curvature; ALE binning introduces additional curvature. The sensitivity run doubles the ICE/PDP grid and separately doubles ALE bins for the first refit, including simulator shape metrics and bin occupancy.

## Artifacts

- `results/study/scores.csv`, `score_summary.csv`: prediction errors and aggregate results.
- `results/study/effect_metrics.csv`, `effect_summary.csv`: effect error, shape metrics and refit dispersion.
- `results/study/sensitivity.csv`: grid and bin sensitivity.
- `results/study/*_history.csv`: optimization histories.
- `results/study/*.pt`: trained CPU model states, configurations and target scales.
- `results/study/*_data_*.npz`: parameter datasets and training-pool indices; solutions regenerate analytically.
- `results/study/*_effects.npz`, `*_oracle.npz`: individual and aggregate effect curves.
- `results/study/figures/`: fit diagnostics, effects, ICE and training histories.
- `presentation/presentation.pdf`: seven slides intended for five minutes.

`physics.py` implements data generation, `operators.py` implements the architectures, `study.py` trains and computes explanations, and `report.py` builds figures and slides. The original template utilities remain available for its regression smoke tests.

## Measured results

Values below average three paired refits. Relative field errors are percentages on the final independent test pool; effect errors are normalized by simulator-QoI standard deviation.

| System | Regime | Train field error | Test field error | PDP effect error | ICE effect error |
|---|---|---:|---:|---:|---:|
| burgers | underfit | 37.618% | 37.197% | 0.16132 | 0.17292 |
| burgers | well_fit | 0.693% | 3.081% | 0.01816 | 0.02807 |
| burgers | overfit | 0.083% | 3.869% | 0.03264 | 0.03891 |
| oscillator | underfit | 45.114% | 46.897% | 0.36197 | 0.40505 |
| oscillator | well_fit | 0.403% | 0.446% | 0.00064 | 0.00188 |
| oscillator | overfit | 0.358% | 1.441% | 0.00404 | 0.00901 |

All aggregate regime checks pass. The overfit configurations have training–validation gaps of approximately 45 times for Burgers and 4.3 times for the oscillator, and worse validation performance than the well-fit configurations. Absolute errors remain small, especially for the oscillator: the results demonstrate modest overfitting rather than catastrophic prediction failure. One oscillator refit has slightly lower training error in the well-fit model; the aggregate comparison and the overfit model's own generalization gap support the intended regime interpretation.

Underfitting has the largest effect errors in both systems. It understates how much solution RMS decreases with increasing viscosity and compresses the oscillator's resonance contrast near forcing frequency 2. Someone relying on those curves could underestimate sensitivity to viscosity or proximity to resonance.

For Burgers PDP, the overfit model has larger effect error (0.03264 versus 0.01816) but smaller between-refit dispersion (0.00735 versus 0.01049) than the well-fit model. Its stability is therefore not evidence of greater explanation fidelity. The oscillator's well-fit and overfit average curves nearly overlap the oracle visually, but quantitative effect errors and individual ICE reveal their differences.

The initial hypothesis that overfitting necessarily causes jagged or unstable effects is not supported uniformly. In the first Burgers refit, the overfit PDP is smoother by the measured curvature statistic than the well-fit PDP despite worse effect fidelity. Underfit explanations also vary substantially across initializations and training subsets.

Doubling the PDP grid from 41 to 81 points preserves the first-refit effect-error ordering on both systems. Doubling ALE bins from 8 to 16 also preserves that ordering, but changes numerical effect errors and reduces occupancy from 16 to 8 instances per bin. Curvature and ALE discretization should therefore remain descriptive evidence. The sensitivity check covers the first refit, not all resamples.

## Interpretation limits

The comparison concerns fit within each physical system. Different architectures on different datasets do not identify a causal architecture advantage. Smooth or stable effects can remain inaccurate; simulator-referenced fidelity must accompany roughness and variability. Uniform, independent parameter sampling makes interventions supported by the parameter distribution, although the finite training sample can still be sparse. Agreement on the restricted forcing/initial-condition families does not establish general operator accuracy outside those families.

## AI statement

OpenAI Codex was used to implement the simulators, neural operators, training and explanation pipeline, automated tests, figures, README and presentation. Automated checks compare simulator outputs against independent numerical solvers, verify causal convolution and gradient propagation, and check explanation formulas on a known linear function. Training and explanation results are computed from actual runs. The repository owner should review the code, findings and slides and adapt this statement to reflect their own checks before submitting.

## Submission

The exercise requires a pull request from this fork to `automl-edu/iML26-27-Exercise-2` before November 29, 2026, 23:59, including code, this README and `presentation/presentation.pdf`. Review the generated findings and AI statement before submitting.
