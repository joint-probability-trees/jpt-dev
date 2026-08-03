# Ensemble JPT paper

`ensemble_jpt.tex` — *Ensemble Joint Probability Trees: A Likelihood-Geometric
Account of Bagging and Boosting for Hybrid Density Trees*.

The paper derives bagging (`JPTForest`), generative likelihood boosting
(`JPTLikelihoodBoost`), and discriminative gradient boosting (`JPTBoost`)
of JPTs, proves both generative ensembles are valid normalized joints,
translates the full JPT inference calculus to the ensemble, and reports a
measured empirical evaluation.

## Experiments

All numbers in the Experiments section are produced by
`evaluation.py` from the public library API:

```sh
python evaluation.py                  # everything -> results.json
python evaluation.py E1 wine          # one experiment / dataset
EVAL_OUT=results_wine.json python evaluation.py E1 wine   # parallel runs
```

The measured raw results ship as `results_*.json` (deep-merged by
`mktables.py`, which formats them into the LaTeX table bodies and
pgfplots coordinates embedded in `ensemble_jpt.tex`).

## Build

Any LaTeX engine works; the document depends only on standard CTAN packages
(`amsmath`, `amsthm`, `mathtools`, `natbib`, `hyperref`, …). The bibliography
is inline (`thebibliography`), so no `bibtex`/`biber` pass is needed.

```sh
# self-contained, fetches packages on demand
tectonic ensemble_jpt.tex
# or
pdflatex ensemble_jpt.tex && pdflatex ensemble_jpt.tex
```
