# Five minute presentation

`presentation.pdf` has nine main slides paced for a five-minute talk and three backup slides for questions. Navy text, blue, coral and slate are used consistently across the diagrams and result plots. Line styles distinguish the fit regimes as well as colors. The main slides introduce the physics and neural operators before showing the experimental protocol and findings.

| Slide | Topic | Suggested time |
|---|---|---:|
| 1 | The function-to-function prediction task and explanation question | 20 s |
| 2 | Burgers, the oscillator and the RMS quantity being explained | 25 s |
| 3 | FNO as learned mixing of spatial Fourier modes | 30 s |
| 4 | LNO as learned damped oscillations and causal memory | 30 s |
| 5 | Fixed data splits, three fit configurations and three matched refits | 40 s |
| 6 | ICE, PDP and ALE with visual examples | 40 s |
| 7 | Training and final-test errors establishing the regimes | 35 s |
| 8 | Physical effects and simulator comparison | 45 s |
| 9 | Stability versus accuracy and the practical conclusion | 35 s |
| 10–12 | Backup: architecture settings, full PDP/ALE comparison, individual ICE | Q&A only |

## Speaker notes

### 1 — From curves to curves

A neural operator learns a mapping between functions. Here it sees a complete initial flow profile or forcing history and predicts another complete function. We then ask how changing one physical parameter changes the typical magnitude of that predicted function. The initial and later flow curves illustrate the simulator task; the later curve is the prediction target, not a measured neural-network output.

### 2 — Two physical examples

In Burgers flow, viscosity controls how strongly spatial variation is smoothed. The oscillator combines a restoring force, damping and periodic forcing. Its response is largest near resonance. Both systems have known solutions, so we can independently assess the model and its explanation. RMS, the root mean square, describes typical solution magnitude. We use a spatial average for Burgers and a time average, numerically integrated, for the oscillator.

### 3 — What an FNO does

Fourier analysis represents a curve as a collection of spatial waves. One FNO layer transforms hidden feature curves into Fourier modes, learns how to mix them, and transforms back. A local mixing path and nonlinear activation complete the layer. The formula summarizes the diagram; the audience need not follow each symbol. Several such layers act between pointwise input lifting and output projection. The curves and Fourier bars on this slide are conceptual illustrations.

### 4 — What this LNO does

A damped oscillation has a frequency and a decay rate. In a Laplace-transform representation, these correspond to poles; learned residues control their contributions. The custom LNO learns a mixture of these responses as a temporal kernel. Causal convolution means that only the current and past inputs contribute at time t. The formula shows one temporal layer; local/channel mixing is also present. This is a custom separable pole/residue operator, not a reproduction of a published LNO architecture. The diagrams illustrate the concept rather than measured model predictions.

### 5 — Why the comparison is controlled

For each system, each refit uses the same 48 training functions across all three regimes. Separate validation data select the well-fit checkpoint; a fixed explanation background defines the explanations. The final test set measures generalization after the pilot choices. We repeat with three training subsets and initialization seeds, pairing them across regimes. The resulting 18 runs vary both the subset and initialization; their spread does not separate these sources. Architecture capacity, regularization and, for the oscillator baseline, linear structure establish different intended fit regimes. The observed errors confirm their aggregate interpretation.

### 6 — What the explanations show

ICE asks what happens for one instance when a chosen parameter varies. PDP averages those individual effects. ALE averages changes within local parameter bins and accumulates them. The examples show simulator effects to illustrate the methods. Every intervention reconstructs the forcing or initial signal and its metadata consistently. Model explanations use exactly the same instances and feature grids as the corresponding simulator reference. The curves are centered, so they describe changes rather than absolute output offsets.

### 7 — Verify fit before explaining

The underfit configurations have large training and test errors. The well-fit configurations have low validation and test errors. The large configurations fit training data closely but have larger generalization gaps and worse validation performance than the selected well-fit models. Their absolute errors are still small, especially for the oscillator. One oscillator refit has slightly smaller training error in the well-fit configuration; the displayed comparisons are means over three refits. The well-fit LNO uses a linear stack appropriate for the linear dynamics; this also changes its inductive bias.

### 8 — Explain the physics, not just the curves

The underfit FNO understates the reduction of flow RMS with increasing viscosity. The underfit oscillator model compresses the contrast near forcing frequency 2. Someone relying on those explanations could underestimate physical sensitivity or resonance risk. The other models are closer to the simulator, although numerical effect errors still distinguish them. The shaded bands show one standard deviation across three refits, not confidence intervals. Full ALE and ICE plots are in the backup slides.

### 9 — Stability does not imply correctness

For Burgers PDP, the overfit model has smaller between-refit dispersion but larger simulator-referenced error than the well-fit model. A repeatable curve can therefore still be biased. Judge explanations using prediction error, reference effect error and stability together. The evidence covers three refits and restricted function families; using a different architecture for each dataset does not establish that FNO or LNO is superior. The grid/bin sensitivity check preserves the first-refit effect-error ordering, while ALE values remain sensitive to discretization.

## Regenerate

From the repository root, run:

```bash
uv run python -m iml_feature_effects.report
```

This rebuilds the presentation and study figures using saved results; no retraining is needed. The deck source is `iml_feature_effects/presentation.py`, with shared chart styling in `iml_feature_effects/visual_style.py`. `preview.png` shows the FNO explanation slide. Review the slides and rehearse the five-minute pacing before submission.
