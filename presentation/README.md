# Five minute presentation

`presentation.pdf` contains seven slides generated from the saved experiment results:

1. Research question and physical systems, about 30 seconds.
2. Paired training and simulator-referenced explanations, about 45 seconds.
3. Burgers fit diagnostics, about 35 seconds.
4. Burgers effects and uncertainty bands, about 45 seconds.
5. Oscillator fit diagnostics, about 35 seconds.
6. Oscillator effects and resonance response, about 45 seconds.
7. Interpretation and limits, about 65 seconds.

Focus on measured generalization gaps, effect accuracy against the simulator, and the distinction between stability and accuracy. Bands show one standard deviation across three paired refits, not confidence intervals. The Laplace model is a custom separable pole/residue implementation; its well-fit configuration uses a linear stack appropriate for this linear system.

Regenerate from the repository root with `uv run python -m iml_feature_effects.report` after running the experiment. Review and rehearse the slides before submission.
