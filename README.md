# iML 26/27 – Exercise 2: Feature Effects from Underfit to Overfit

Template repository for the bonus task. **Fork this repository**, replace the example with your own
work, add the PDF of your presentation, and submit by opening a merge request to this repository
(see [Submission](#submission)). **You are expected to rewrite this README as well**: by
the deadline it should describe your approach, how to reproduce your results, and contain your own
[AI statement](#ai-statement).

> **Deadline: Sunday, November 29th, 2026, 23:59.** Your code and presentation must be submitted by then.

## Task

1. **Datasets.** Select 2–3 datasets that you find interesting to understand (e.g. from
   [OpenML](https://www.openml.org/), [Kaggle](https://www.kaggle.com/datasets), …).
2. **Models from underfit to overfit.** On each dataset, fit models (of the same or of different
   classes) that range from underfit to overfit. You need to be able to argue *why* each model is
   underfit or overfit, e.g. with train/test performance, learning curves, or its capacity relative
   to the amount of data.
3. **Feature effects.** Analyze how feature effect explanations (e.g. PDP, ICE, ALE, …) change with
   respect to the fit of the model. You may focus on consistency, continuity, or something else.
4. **What does that mean in reality?** Interpret your findings: what would someone who relies on
   these explanations conclude about the data, and when would they be misled?

### Presentation

Prepare a **5-minute presentation** of your datasets, models, and findings. Commit it as
[`presentation/presentation.pdf`](presentation/).

## Setup

The repository is an installable Python package. We recommend [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py
```

With plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The first run downloads the Bike Sharing dataset from OpenML and caches it in `data/` (relative to
the directory you run from). It is only there to show the pipeline: pick datasets **you** find
interesting.

## Repository structure

```
iml_feature_effects/
├── data.py        # Dataset container + load_bike_sharing(); add your own datasets here
├── models.py      # get_models(): models from underfit to overfit (example: three MLPs)
├── effects.py     # FeatureEffect + PartialDependence, EffectProperty + Roughness (examples)
└── evaluate.py    # evaluate(): fits all models, reports train/test R², computes effects
scripts/
└── run_experiment.py   # writes results/scores.csv, results/curves.csv, results/effects_<dataset>.png
tests/                  # pytest tests
presentation/           # put presentation.pdf here
```

The repository runs end to end on a minimal example: three MLPs on the hourly Bike Sharing data,
explained with centered partial dependence plots for `hour`, `temp` and `humidity`:

| Model          | Setup                                                              | Why                                         |
|----------------|--------------------------------------------------------------------|---------------------------------------------|
| `mlp_underfit` | one hidden neuron                                                  | too little capacity: low train *and* test R² |
| `mlp_fit`      | two hidden layers of 64, early stopping                            | high train and test R²                      |
| `mlp_overfit`  | two hidden layers of 256, no regularization, 10 000 epochs on only 20 training points | memorizes the 20 points: train R² ≫ test R² |

These three models only illustrate the pipeline. How they are made to underfit and overfit is
**not necessarily optimal** (e.g. a single neuron or 20 training points are rather crude). Think about
which ways of controlling the fit are meaningful for your datasets and models, and argue for your choice.

`Roughness` in [`iml_feature_effects/effects.py`](iml_feature_effects/effects.py) is a toy example of
a property of an effect curve. Replace it with your own analysis. You are free to restructure
everything, as long as the package stays installable and your results stay reproducible from the
scripts in this repository.

## Code style

Please keep your code clean and consistent. We recommend [ruff](https://docs.astral.sh/ruff/) as
linter and formatter. A configuration is already included in `pyproject.toml`.

## Submission

- Submit your code and presentation by **Sunday, November 29th, 2026, 23:59**.
- **To submit, open a merge request (pull request) from your fork to this repository.** Only work in a
  merge request opened before the deadline counts as submitted.
- Your fork contains your code, the scripts that reproduce your results, and
  `presentation/presentation.pdf`.
- **Rewrite this README.** Replace the template text with a short description of your approach and
  how to reproduce your results.
- **Mandatory:** replace the [AI statement](#ai-statement) below with your own, stating how you
  used AI tools (or that you did not use any).

## AI statement

> **Mandatory.** Replace this example with your own: which AI tools you used, for what, and how you
> checked their output (or that you used none).

*Example:* I created this repository with Claude Code (Claude Opus 5.5). After clarifying the scope
with it, it generated the code, tests and README based on the previous exercise as a style reference.
I reviewed and edited the result and checked that it runs. I take full responsibility for and
ownership of the content of this repository.
