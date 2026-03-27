# Fit

Model fitting in `CPPLS` is performed using [`StatsAPI.fit`](@ref) together with a
[`CPPLSModel`](@ref). This unified interface supports both regression and discriminant
analysis, providing a consistent workflow for a wide range of supervised modeling tasks.

!!! info
    The distinction between regression and discriminant analysis in CPPLS, as specified by
    the `mode` keyword in [`CPPLSModel`](@ref), determines which convenience functions are
    available for downstream analysis. For model fitting itself, the essential difference
    is that discriminant analysis (DA) uses a one-hot encoded $Y$ matrix as the response,
    whereas regression typically uses a $Y$ vector or matrix with continuously varying
    values.

    CPPLS is flexible, however: the response matrix $Y$ may contain both one-hot encoded
    columns (for classification or DA) and continuous columns (for regression) at the same
    time. This allows hybrid models in which predictor variables are aligned with multiple
    response variables of different types. In such cases, users must encode the $Y$ matrix
    appropriately and extract the relevant outputs from [`project`](@ref) and
    [`predict`](@ref).

    When primary response columns differ strongly in variance or unit, it may be sensible
    to set `scale_Yprim=true` in [`CPPLSModel`](@ref) so that they are on a comparable
    footing, and then, if needed, to control their relative influence explicitly through
    the `response_weights` and `target_weights` keyword arguments of [`StatsAPI.fit`](@ref).

You can optionally provide observation weights (keyword argument `obs_weights`), response
weights (keyword argument `response_weights`), target weights (keyword argument
`target_weights`), and auxiliary response information (keyword argument `Yaux`) to
[`StatsAPI.fit`](@ref). Observation weights control the influence of each sample on the
model and are especially useful in discriminant analysis when classes are imbalanced.
Response weights control the influence of each response column, both primary and auxiliary,
on the supervised compression step. Target weights control the influence of each primary
response column in the canonical-correlation step. Auxiliary responses guide the supervised
projection without becoming prediction targets themselves. Together with the `gamma`
parameter, which balances predictor scale and predictor-response association, these options
allow the user to tailor a CPPLS model to the structure of the dataset. Other choices, such
as the number of components, may also be important depending on the application.

If you plan to use observation weights, response weights, target weights, or auxiliary
responses, these choices should be made before selecting `gamma`, because all of them can
affect the supervised objective and therefore the most appropriate value of `gamma`.

## Quick Start

The same `fit` entry point is used for both discriminant analysis and regression. The
main difference is the structure of `Yprim`: in DA it is a one-hot representation of the
classes, whereas in regression it is a continuous vector or matrix.

For a plain discriminant-analysis fit:

```julia
using CPPLS
using StatsAPI

m = CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant)
mf = fit(m, X, classes)
```

For a plain regression fit:

```julia
using CPPLS
using StatsAPI

m = CPPLSModel(ncomponents=2, gamma=0.5, mode=:regression)
mf = fit(m, X, Y)
```

To add class balancing and auxiliary supervision in DA:

```julia
using CPPLS
using StatsAPI

m = CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant)
mf = fit(
    m,
    X,
    classes;
    obs_weights=invfreqweights(classes),
    Yaux=Y_aux
)
```

For complete worked examples, including score plots, gamma selection, and a regression
workflow with auxiliary responses, see [Fit Examples](fit_examples.md).

## Centering, Scaling, and Response Weighting

CPPLS provides convenience options for centering and scaling, but these options are
intentionally asymmetric across $X$, `Yprim`, and `Yaux`, because these matrices do not
enter the algorithm in the same way.

For the predictor matrix $X$, centering is usually the most important preprocessing step.
CPPLS extracts latent components from $X$, and without centering, the model can partly
reflect absolute measurement levels rather than variation among samples. Centering
therefore makes the score space represent deviations from the average sample rather than
deviations from an arbitrary zero point. For this reason, centering of $X$ is the default
in [`CPPLSModel`](@ref). Scaling $X$ is a separate modeling choice. If predictor variables
differ strongly in variance or physical scale, scaling gives them a more equal opportunity
to contribute. If large predictor variance is scientifically meaningful, leaving $X$
unscaled allows that information to remain part of the model. In CPPLS this matters
directly, because the power parameter `gamma` mixes predictor scale and predictor-response
association when constructing the supervised projection. The default setting of `scale_X`
is therefore `false`.

For the primary response block `Yprim`, scaling can be useful, especially in multivariate
regression. When primary response columns differ strongly in variance or unit, scaling
helps prevent high-variance responses from dominating the target block purely because of
magnitude. In discriminant analysis, where `Yprim` is typically a one-hot matrix, scaling
is usually less important, since class imbalance is more naturally addressed through
`obs_weights`. The default setting of `scale_Yprim` is therefore also `false`. CPPLS does
not expose centering of `Yprim` as a user option, because the response-guided step of the
algorithm works with predictor-response correlations, and those correlations already center
response columns internally. A separate centering option for `Yprim` would therefore be
redundant and would give the impression of additional control without materially changing
the supervised projection.

The auxiliary response block `Yaux` is treated differently again. Auxiliary variables guide
the construction of the supervised space, but they are not prediction targets. In the CPPLS
implementation used here, they enter through predictor-response correlations rather than
raw covariances. Because correlation is invariant to affine rescaling apart from sign,
ordinary centering and scaling of `Yaux` do not provide meaningful control over how
strongly auxiliary responses influence the model. What matters is the pattern of an
auxiliary variable across samples, not its numerical unit. For this reason, centering and
scaling options for `Yaux` are not exposed. If a primary or auxiliary response should have
more or less influence on the model, this should be controlled through `response_weights`
and `target_weights` rather than through preprocessing scale changes.

## API

The reference below documents the `fit` interface itself.

```@docs
StatsAPI.fit
```
