# CPPLS.jl

CPPLS provides a pure-Julia implementation of Canonical Powered Partial Least Squares
(CPPLS) for both regression and discriminant analysis. The package is intended for
high-dimensional and collinear predictor settings in which latent-variable models are used
for supervised dimension reduction, interpretation, and prediction.

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

## Scope

The package supports supervised latent-variable modelling with either matrix-valued
responses or class labels. Fitting is controlled through `CPPLSSpec`, which separates
model configuration from the data passed to `fit`. The resulting fitted models provide
prediction, class assignment, latent projections, regression coefficients, fitted values,
residuals, and labels or metadata retained from model fitting.

CPPLS also includes helper functionality for common preprocessing and encoding tasks,
including one-hot encoding of class labels and utility functions used in chemometric and
multivariate workflows.

## Validation and Inference

The cross-validation interface is designed around explicit evaluation callbacks, so the
same machinery can be used for either discriminant analysis or regression. Standard
callback bundles are provided by `cv_classification()` and `cv_regression()`, and can be
supplied directly to `nested_cv` and `nested_cv_permutation`. For classification
workflows, `cv_outlier_scan` performs repeated outer-fold evaluation to quantify how often
individual samples are misclassified.

Permutation-based significance assessment is available through
`nested_cv_permutation` together with `calculate_p_value`, allowing empirical evaluation
of whether observed predictive performance exceeds what would be expected under response
permutation.

## Usage

For fitting and model specification, see [`CPPLS/fit.md`](CPPLS/fit.md). Prediction,
projection, and class-assignment methods are described in
[`CPPLS/predict.md`](CPPLS/predict.md). Cross-validation, permutation testing, and
outlier scanning are documented in [`CPPLS/crossvalidation.md`](CPPLS/crossvalidation.md).
The main data structures and fitted-model accessors are summarized in
[`CPPLS/types.md`](CPPLS/types.md), while utility functionality is described under the
`Utils` section of the manual.

## Disclaimer

CPPLS is provided "as is," without warranty of any kind. Users are responsible for
independently validating all outputs and interpretations and for determining suitability
for their specific applications. The authors and contributors disclaim any liability for
errors, omissions, or any consequences arising from use of the software, including use
in regulated, clinical, or safety-critical contexts.

## References

- Indahl UG, Liland KH, Naes T (2009) Canonical partial least squares — a unified PLS 
  approach to classification and regression problems. *Journal of Chemometrics* 23: 495-504. 
  https://doi.org/10.1002/cem.1243.
- Liland KH, Indahl UG (2009): Powered partial least squares discriminant analysis. 
  *Journal of Chemometrics* 23: 7-18. https://doi.org/10.1002/cem.1186.
- Smit S, van Breemen MJ, Hoefsloot HCJ, Smilde AK, Aerts JMFG, de Koster CG (2007): 
  Assessing the statistical validity of proteomics based biomarkers. *Analytica Chimica 
  Acta* 592: 210-217. https://doi.org/10.1016/j.aca.2007.04.043.
