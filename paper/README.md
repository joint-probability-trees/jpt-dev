# Ensemble JPT paper

`ensemble_jpt.tex` — *Ensemble Joint Probability Trees: A Likelihood-Geometric
Account of Bagging and Boosting for Hybrid Density Trees*.

This is the **Phase 0 (theory)** draft: it derives bagging, generative
likelihood boosting, and discriminative gradient boosting of JPTs from first
principles and settles whether one framework subsumes the other (it does not —
they share a meta-principle but live in two different geometries). Experiments
and final polish land in later phases.

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
