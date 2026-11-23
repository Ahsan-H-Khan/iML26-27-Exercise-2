# GUnit five minute presentation

`presentation.pdf` is compiled with native LaTeX Beamer and Latin Modern mathematical fonts. Equations, fractions, subscripts, superscripts, integrals and diagram labels are typeset by LaTeX. `presentation.tex` is the editable slide source; Python generates measured numeric tables and vector plot assets from the saved experiment.

GUnit is the presenter, visible on the opening slide and all slide footers and recorded as the PDF author. There are twelve main slides paced for five minutes and four backup slides for questions. The main slides explain each operator's prediction goal, input channels, output function, learned parameters and chosen capacity settings before presenting the experiment and results.

| Slide | Topic | Suggested time |
|---|---|---:|
| 1 | Learn a solution operator, then explain it | 15 s |
| 2 | FNO task: initial flow to later flow; exact inputs and output | 20 s |
| 3 | FNO layer, learned Fourier weights and capacity choices | 20 s |
| 4 | LNO task: forcing history to displacement; exact inputs and output | 20 s |
| 5 | Custom LNO layer, learned poles/residues and capacity choices | 20 s |
| 6 | Actual hyperparameters for all six dataset–regime configurations | 35 s |
| 7 | Matched splits, training, validation selection and repeated refits | 30 s |
| 8 | Output RMS and the ICE/PDP/ALE explanation methods | 30 s |
| 9 | Measured training and final-test errors | 25 s |
| 10 | Physical effects compared with the simulator | 35 s |
| 11 | Four-property Co-12 comparison: observed Burgers FNO results | 25 s |
| 12 | Four-property Co-12 comparison: observed oscillator LNO results | 25 s |
| 13–16 | Backup: metric definitions, parameter roles/counts, full PDP/ALE, individual ICE | Q&A only |

## What the parameters mean

**Physical inputs** define the simulated problem. Burgers varies viscosity, initial-condition amplitudes and phase. The oscillator varies forcing frequency/amplitude/phase and initial displacement; mass, damping and stiffness remain fixed. Viscosity and forcing frequency are the primary features varied for explanations.

**Learned parameters** are optimized from training examples. The FNO learns complex Fourier multipliers, local channel mixing and input/output projections. The custom LNO learns stable pole decay/frequency values, residues and channel mixing/projection weights. Its learned pole frequencies are different quantities from the physical forcing frequency supplied as an input.

**Hyperparameters** are selected before training: hidden-channel width, retained Fourier modes or Laplace pole pairs per channel, layer count, weight decay and training steps. These are the controls used to create the intended fit regimes. The slides show their actual checkpoint configurations, rather than a schematic small/medium/large description. Observed errors establish whether those configurations underfit or overfit.

The well-fit oscillator model also uses a linear stack appropriate for the linear system; the underfit and overfit stacks are nonlinear. This changes inductive bias as well as capacity. The nonlinear overfit LNO uses a cosine learning-rate schedule to avoid late optimizer instability. All three regimes use the same 48 training functions within each refit.

## Speaker notes

### 1 — Function to function

A neural operator learns a mapping between functions. In this experiment it predicts a complete spatial or temporal solution. After training, we explain a scalar quantity derived from that output. Prediction quality and explanation quality are separate questions.

### 2–3 — FNO goal, interface and learned weights

For Burgers, the six input channels are the initial spatial profile, four physical parameters and the spatial coordinate, sampled at 64 points. The target is the entire 64-point solution at final time 1. The illustrated curves are simulator input/target examples, not model predictions. The initial family is specified on the slide so the amplitude and phase inputs have clear meanings.

The FNO mixes information globally by transforming hidden feature curves into Fourier coefficients, applying learned complex matrices and transforming back. The result is added to a local channel-mixing path before the activation. Input lifting and output projection surround the repeated layers. Distinguish learned Fourier weights from the chosen number of retained modes, hidden channels and layers. The layer equation summarizes the diagram; introduce its terms verbally rather than reading it symbol by symbol.

### 4–5 — LNO goal, interface and learned memory

For the oscillator, the six channels are the forcing history, four physical parameters and time, sampled at 64 points. The output is the entire displacement trajectory on the same time grid. Initial velocity is zero. Mass, damping and stiffness remain fixed, so forcing frequency is an input to a fixed physical system.

A stable pole has a negative real part that determines decay and an imaginary part that determines oscillation frequency. The custom Laplace layer learns these values and the coefficients that mix the damped responses. Causal convolution sums contributions from the input history up to the current time. Local and channel-mixing paths are also present. This is a custom separable pole/residue LNO, not a reproduction of a published architecture. The learned pole frequency is not the physical forcing frequency. The number of pole pairs, width and depth are capacity settings, not quantities learned by gradient descent.

### 6 — How we control fit

Read the settings table across each regime: width, modes/pole pairs, layers, weight decay and steps. The table is generated from the actual saved checkpoint configurations. Smaller, regularized models are restricted; larger unregularized models can fit training details. Labels describe intended regimes until train/validation/test errors confirm them. Avoid saying that viscosity itself is being overfitted: a model overfits its training data.

### 7 — Controlled protocol

One entire function is one sample. Match training subsets across regimes. Use separate validation, explanation-background and test data. Train against simulator field/trajectory targets with MSE, select the well-fit checkpoint using validation, and report final-test errors. Repeat with three training subsets and initialization seeds. Keep the explanation background and grid fixed. The spread combines those two sources of variation and is descriptive rather than a confidence interval.

### 8 — Explain a scalar derived from the full output

The neural operators predict full functions; RMS is computed afterwards. For Burgers it is spatial RMS at the final time. For the oscillator it is RMS over the finite time window, including transients, using trapezoidal quadrature. ICE varies a parameter for one instance, PDP averages those individual responses, and ALE accumulates local bin-wise changes. The examples illustrate simulator effects. Every model curve is compared against its method-matched simulator reference on the same instances, grid and centering convention.

### 9–10 — Evidence and interpretation

The measured errors support the three regimes in aggregate. The large models have small absolute errors but larger generalization gaps and worse validation performance than the selected well-fit models. One oscillator refit has slightly lower training error in the well-fit configuration; the displayed values average three refits.

Underfitting understates viscosity sensitivity and compresses the oscillator's contrast near resonance. The overfit Burgers PDP is more repeatable across refits yet less accurate against the simulator. Thus stability alone is insufficient. The evidence covers three refits and restricted function families, not a causal FNO-versus-LNO comparison. The first-refit grid/bin sensitivity check preserves the effect-error ordering while ALE values remain bin-dependent.

### 11–12 — Compare the Co-12 hypotheses with measurements

The original table is a useful hypothesis, but the observed outcomes are mixed. All 18 models pass deterministic repeat and equivalent-network checks; the consistency row in the comparison tables reports the separate refit-stability proxy. Underfit models have the largest refit variation on both datasets. The overfit FNO is more stable than the well-fit FNO, while the overfit LNO is less stable than its well-fit counterpart.

Explain continuity numbers as ratios to the simulator. The underfit curves have reduced curvature and reduced response to small input changes; neither automatically makes the explanation good. The overfit curves show no marked curvature inflation on the common grid. Interaction strengths are suppressed by underfitting and largely retained by the oscillator overfit model. Compare interaction residual error as well as its magnitude; larger individual variation can be real physical structure. This does not measure whether people understand the explanation.

Coherence checks physical equalities as well as slope direction. The Burgers check uses phase translation of a pure sine, which must preserve spatial RMS. The oscillator check jointly scales forcing and initial displacement, which must scale the response RMS. The well-fit configurations have the lowest mean law residuals; the underfit Burgers slope direction is nevertheless still correct. Avoid claiming that every overfit model produces visibly unphysical or jagged curves. These are three-refit observations under specified grids and perturbations, not a universal fit-to-quality law.

## Regenerate or edit

Run from the repository root:

```bash
uv run python -m iml_feature_effects.report
```

The four-property audit can be recomputed independently with `uv run python -m iml_feature_effects.co12`. Reporting checks its input/source hashes and runs it if needed.

This needs `pdflatex` and the Beamer, TikZ, amsmath, booktabs and Latin Modern packages. They are provided by the current cloud environment. The Python command rebuilds plot assets and numeric tables, compiles the editable LaTeX source twice and updates `presentation.pdf` and `preview.png`. It does not retrain models or change experimental measurements.

To edit only slide wording or layout after the generated assets exist:

```bash
cd presentation
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
```

The plot PDFs and `generated/values.tex` are committed, so LaTeX compilation is also possible without running Python. `preview.png` shows the FNO input/output slide; `co12-preview.png` shows the measured Burgers Co-12 comparison. Rehearse the timing and adapt the speaker notes to your own phrasing before submission.
