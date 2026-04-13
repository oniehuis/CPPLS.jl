# CPPLS.jl

The CPPLS.jl package implements Canonical Partial Least Squares (CPLS) and its 
"powered" extension (CPPLS), providing a unified framework for regression and 
classification. The package is designed for high-dimensional and collinear predictor 
settings where latent-variable models are used for supervised dimension reduction, 
interpretation, and prediction. Operating on the predictor matrix X, these methods 
utilize Canonical Correlation Analysis (CCA) to ensure latent variables are invariant 
to response scaling and more aggressively focused on predictive information than 
traditional PLS2 or PLS-DA. This allows the models to handle continuous and categorical 
responses simultaneously while often requiring significantly fewer components than these 
standard PLS approaches to achieve superior predictive accuracy. A hallmark of this 
approach is the ability to include "additional" information in the Y matrix during 
training—such as experimental design or auxiliary data—to further stabilize and focus 
the model. Note that a multilinear CPLS approach that operates on tensors is implemented 
in a separate Julia package named NCPLS. Finally, the "powered" mechanism in CPPLS enables 
automated variable selection within X by sharpening the focus on the most important 
predictors.


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
responses or class labels. Fitting is controlled through `CPPLSModel`, which separates
model configuration from the data passed to `fit`. The resulting fitted models provide
prediction, class assignment, latent projections, regression coefficients, fitted values,
residuals, and labels or metadata retained from model fitting.

CPPLS also includes helper functionality for common preprocessing and encoding tasks,
including one-hot encoding of class labels and utility functions used in chemometric and
multivariate workflows.

## Validation and Inference

The cross-validation interface is designed around explicit evaluation callbacks, so the
same machinery can be used for either discriminant analysis or regression. For most
users, the preferred entry points are the higher-level wrappers `cvda`, `permda`,
`cvreg`, and `permreg`, which apply standard defaults for common workflows. Lower-level
callback bundles are also available through `CPPLS.cv_classification()` and
`CPPLS.cv_regression()` for direct use with `nestedcv` and `nestedcvperm`. For
classification workflows, `outlierscan` performs repeated outer-fold evaluation to
quantify how often individual samples are misclassified.

Permutation-based significance assessment is available through
`nestedcvperm` together with `pvalue`, allowing empirical evaluation
of whether observed predictive performance exceeds what would be expected under response
permutation.

## Usage

For fitting and model specification, see [`CPPLS/fit.md`](CPPLS/fit.md). Longer worked
workflows for discriminant analysis and regression are collected in
[`CPPLS/fit_examples.md`](CPPLS/fit_examples.md). Prediction, projection, and
class-assignment methods are described in [`CPPLS/predict.md`](CPPLS/predict.md).
Cross-validation, permutation testing, and outlier scanning are documented in
[`CPPLS/crossvalidation.md`](CPPLS/crossvalidation.md). The main data structures and
fitted-model accessors are summarized in [`CPPLS/types.md`](CPPLS/types.md), while utility
functionality is described under the `Utils` section of the manual.

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
