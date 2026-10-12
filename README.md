# CPPLS.jl

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][GHA-img]][GHA-url] [![][codecov-img]][codecov-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://oniehuis.github.io/CPPLS.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://oniehuis.github.io/CPPLS.jl/dev

[GHA-img]: https://github.com/oniehuis/CPPLS.jl/actions/workflows/CI.yml/badge.svg?branch=main
[GHA-url]: https://github.com/oniehuis/CPPLS.jl/actions/workflows/CI.yml?query=branch%3Amain

[codecov-img]: https://codecov.io/gh/oniehuis/CPPLS.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/oniehuis/CPPLS.jl

A Julia package for Canonical Partial Least Squares (CPLS) and its powered
extension (CPPLS), supporting regression and discriminant analysis. Model
sample-by-variable predictor matrices, including high-dimensional or collinear
data, and inspect latent scores, predictions, and regression coefficients.

## Features

- Fit regression, classification, or hybrid models that combine class indicators
  with continuous responses.
- Control predictor centering and scaling, supply observation weights, and include
  auxiliary responses (`Yadd`) that guide component extraction.
- Set the power parameter `gamma` directly or select it per component using a grid
  or search intervals, with diagnostics for the candidates considered.
- Select component counts and assess predictions with nested cross-validation and
  permutation tests for regression and discriminant analysis; use `outlierscan`
  to inspect repeated misclassification of individual samples.
- Plot latent scores with Makie or PlotlyJS, summarize class centers, and identify
  representative samples in score space.

## Installation

Julia 1.9 or later is required. Install the development version from GitHub in
your Julia environment:

```julia
using Pkg
Pkg.add(url="https://github.com/oniehuis/CPPLS.jl", rev="main")
```

Then load the package with `using CPPLS`.

## Quickstart

Run this in the Julia REPL. The example uses a synthetic dataset distributed with
CPPLS, so no measurement files of your own are needed. Install JLD2 in the same
environment to read it:

```julia
using Pkg
Pkg.add("JLD2")
```

```julia
using CPPLS
using JLD2

# Load 100 samples with 14 predictor variables and two imbalanced classes.
data = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"))
X = data["X"]
classes = categorical(data["classes"])

# Fit a two-component discriminant model with class balancing and extra supervision.
model = CPPLSModel(ncomponents=2, gamma=0.5, analysis_mode=:discriminant, scale_X=true)
mf = fit(
    model,
    X,
    classes;
    Yadd=data["Y_add"],
    obs_weights=invfreqweights(classes),
    samplelabels=data["sample_labels"],
)

# Inspect latent scores and classify the example samples using two components.
scores = xscores(mf)
predicted_classes = predictclasses(mf, X, 2)
```

`X` has size `(100, 14)`: samples occupy rows and predictor variables occupy
columns. `scores` has one row per sample and one column per component. `Yadd`
and `obs_weights` are optional fitting inputs; only predictor data are needed
for prediction. These predictions use the training samples; see the
[cross-validation guide](docs/src/CPPLS/crossvalidation.md) for component selection
and evaluation on held-out samples.

### Fit continuous responses

Use `analysis_mode=:regression` and pass a numeric response vector or matrix
instead of categorical labels. Continue with the data from the quickstart, using
its continuous auxiliary responses as the prediction targets:

```julia
reg_model = CPPLSModel(ncomponents=2, gamma=0.5, analysis_mode=:regression, scale_X=true)
mf_reg = fit(reg_model, X, data["Y_add"])
predictions = predict(mf_reg, X, 2)
predicted_responses = predictions[:, :, 2]
```

`predict` returns an array with dimensions `(samples, responses, components)`.
The slice `[:, :, 2]` contains cumulative predictions using the first two
components. Use `project(mf, Xnew)` to obtain latent scores for new samples, or
`predictclasses(mf, Xnew, 2)` to classify them with the discriminant model above.

### Plot latent scores

Install a Makie backend in the same environment to use CPPLS's Makie score plots:

```julia
using Pkg
Pkg.add("CairoMakie")
```

Continue with the discriminant model from the quickstart:

```julia
using CairoMakie

fig = scoreplot(mf; backend=:makie, figure_kwargs=(; size=(900, 600)))
save("cppls_scores.png", fig)
```

The figure is saved in the current working directory. Score plots also support
PlotlyJS; see the [visualization guide](docs/src/CPPLS/visualization.md) for both
backends, projected samples, class centers, and representative samples.

### Use your own data

Replace `X` with a numeric matrix shaped `(samples, variables)` and supply
responses in the same sample order. Use `categorical(labels)` with
`analysis_mode=:discriminant` for classification, or a numeric vector or matrix
with `analysis_mode=:regression` for one or more continuous responses. CPPLS
centers predictors by default; the examples above also enable predictor scaling
with `scale_X=true`. New samples must use the same predictor columns and ordering
as the training data.

See the [fitting guide](docs/src/CPPLS/fit.md) for hybrid response blocks,
observation weights, auxiliary responses, and power-parameter selection. For
tensor-valued predictors, see the related
[NCPLS.jl](https://github.com/oniehuis/NCPLS.jl) package.

## Documentation

See the [documentation](https://oniehuis.github.io/CPPLS.jl/dev/) for the full API
and worked examples, including [CPPLS theory](docs/src/CPPLS/theory.md),
[model fitting](docs/src/CPPLS/fit.md),
[fitting examples](docs/src/CPPLS/fit_examples.md),
[projection and prediction](docs/src/CPPLS/predict.md),
[cross-validation and permutation testing](docs/src/CPPLS/crossvalidation.md), and
[visualization](docs/src/CPPLS/visualization.md).
