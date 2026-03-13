# Types

`CPPLS` relies on two main custom types in the `StatsAPI.fit` and `StatsAPI.predict` 
workflow. [`CPPLSSpec`](@ref) stores the user-controlled model specification, including 
the number of components, the `gamma` configuration, centering, the analysis mode, and 
numerical tolerances. [`CPPLSFit`](@ref) is the full fitted model returned by `fit`; 
it contains regression coefficients, scores, labels, and other quantities needed for 
prediction, diagnostics, and plotting.

A third container, [`CPPLSFitLight`](@ref), stores only the subset of fitted-model
information needed for cross-validation. It is mainly an internal optimization and is
typically not accessed directly by users.

The package also provides getters for fields that are commonly useful in downstream
work. For [`CPPLSSpec`](@ref), the getters are [`analysis_mode`](@ref), [`gamma`](@ref), 
and [`n_components`](@ref). For [`CPPLSFit`](@ref), the getters are [`coef`](@ref), 
[`analysis_mode`](@ref), [`fitted`](@ref), [`gamma`](@ref), [`predictor_labels`](@ref), 
[`response_labels`](@ref), [`residuals`](@ref), [`sample_classes`](@ref), 
[`sample_labels`](@ref), and [`X_scores`](@ref).

Both container types expose additional named fields. You can inspect them with
`names(spec)` or `names(model)` and access them via dot notation, for example
`spec.X_loading_weight_tolerance`.


## CPPLSSpec

```@docs
CPPLS.CPPLSSpec
CPPLS.analysis_mode(::CPPLSSpec)
CPPLS.gamma(::CPPLSSpec)
CPPLS.n_components(::CPPLSSpec)
```

## AbstractCPPLSFit

```@docs
CPPLS.AbstractCPPLSFit
StatsAPI.coef(model::AbstractCPPLSFit) 
```

## CPPLSFit

```@docs
CPPLS.CPPLSFit
CPPLS.analysis_mode(::CPPLSFit)
StatsAPI.fitted(::CPPLSFit)
CPPLS.gamma(::CPPLSFit)
CPPLS.predictor_labels(::CPPLSFit)
CPPLS.response_labels(::CPPLSFit)
StatsAPI.residuals(::CPPLSFit)
CPPLS.sample_classes(::CPPLSFit)
CPPLS.sample_labels(::CPPLSFit)
CPPLS.X_scores(::CPPLSFit)
```

## CPPLSFitLight

```@docs
CPPLS.CPPLSFitLight
```
