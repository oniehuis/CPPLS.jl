# CPPLS.jl

CPPLS provides a pure-Julia implementation of Canonical Powered Partial Least Squares 
Discriminant Analysis (CPPLS-DA; Indahl et al. 2019, Liland & Indahl 2009) that handles 
collinear predictors and exports interpretable loadings and scores.. The goal is to enable 
reproducible preprocessing, fitting, and validation pipelines, so common chemometric 
analyses stay transparent and auditable.

## Installation

The package is not registered, so install it directly from GitHub:

```
julia> ]
pkg> add https://github.com/oniehuis/CPPLS.jl
```

After the installation finishes you can load it in the Julia REPL with:

```
julia> using CPPLS
```

## Current capabilities

- A pure-Julia implementation of Canonical Powered Partial Least Squares Discriminant
  Analysis (CPPLS-DA; Indahl et al. 2019, Liland & Indahl 2009) that handles collinear 
  predictors and exports interpretable loadings and scores.
- Cross-validation utilities (`nested_cv`, `nested_cv_permutation`) for selecting the
  number of latent components and estimating classification performance or permutation
  baselines (Smit et al. 2007; currently only validated for discriminant/classification
  models).
- Permutation-based significance testing via `calculate_p_value` to quantify whether
  observed accuracies exceed what would be expected by chance.
- Optional preprocessing utilities so that scaling, centering, or other chemometric
  transformations can be folded into the modeling workflow.

## Quick taste

```@example 1
using CPPLS
using Random
using Statistics

rng = MersenneTwister(1)
X = randn(rng, 60, 30)                                     # predictors (e.g., spectra)
labels = repeat(["classA", "classB"], inner=30)
Y, _ = labels_to_one_hot(labels)

accuracies, components = nested_cv(
    X, Y;
    max_components=2,
    num_outer_folds=3,
    num_inner_folds=2,
    rng=rng,
    verbose=false)

best_components = floor(Int, median(components))           # consensus components across folds
model = fit_cppls_light(X, Y, best_components; gamma=0.5)
ŷ = predictonehot(model, predict(model, X))                # fitted class indicators

permutation_scores = nested_cv_permutation(
    X, Y;
    max_components=2,
    num_outer_folds=3,
    num_inner_folds=2,
    num_permutations=25,
    rng=rng,
    verbose=false)

p_value = calculate_p_value(permutation_scores, mean(accuracies))
```

The calculated `p_value` reports the empirical probability of obtaining mean accuracies
this high when class labels are randomly permuted, so smaller values suggest structure
unlikely to arise from chance alone.

## Usage

- Learn how to fit models (options, preprocessing, γ-search) in
  [`CPPLS/fit.md`](CPPLS/fit.md) and how to generate projections or class predictions in
  [`CPPLS/predict.md`](CPPLS/predict.md).
- Dive into the cross-validation and permutation tooling described in
  [`CPPLS/crossvalidation.md`](CPPLS/crossvalidation.md).
- Inspect the underlying data structures (`CPPLS`, `CPPLSLight`, preprocessing helpers)
  once you need finer control in [`CPPLS/types.md`](CPPLS/types.md) and
  [`CPPLS/internal.md`](CPPLS/internal.md).

## Disclaimer

CPPLS is research software provided “as is.” You remain responsible for validating every 
discriminant analysis and any downstream decision or deployment based on these models; 
the authors cannot be held liable if the algorithms produce misleading or incorrect results.

## References

- Indahl UG, Liland KH, Naes T (2009) *Canonical partial least squares — a unified PLS 
  approach to classification and regression problems.* Journal of Chemometrics 23: 495-504. 
  https://doi.org/10.1002/cem.1243.
- Liland KH, Indahl UG (2009): *Powered partial least squares discriminant analysis.* 
  Journal of Chemometrics 23: 7-18. https://doi.org/10.1002/cem.1186.
- Smit S, van Breemen MJ, Hoefsloot HCJ, Smilde AK, Aerts JMFG, de Koster CG (2007): 
  *Assessing the statistical validity of proteomics based biomarkers.* Analytica Chimica 
  Acta 592: 210-217. https://doi.org/10.1016/j.aca.2007.04.043.
