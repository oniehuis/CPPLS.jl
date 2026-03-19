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
work. For [`CPPLSSpec`](@ref), the getters are [`mode`](@ref), [`gamma`](@ref), 
and [`ncomponents`](@ref). For [`CPPLSFit`](@ref), the getters are [`coef`](@ref), 
[`mode`](@ref), [`fitted`](@ref), [`gamma`](@ref),
[`gammas`](@ref), [`rhos`](@ref),
[`predictorlabels`](@ref), [`responselabels`](@ref), [`residuals`](@ref),
[`sampleclasses`](@ref), [`samplelabels`](@ref), and [`xscores`](@ref).

Both container types expose additional named fields. You can inspect them with
`names(spec)` or `names(model)` and access them via dot notation, for example
`spec.X_loading_weight_tolerance`.


## CPPLSSpec

```@docs
CPPLS.CPPLSSpec
CPPLS.mode(::CPPLSSpec)
CPPLS.gamma(::CPPLSSpec)
CPPLS.ncomponents(::CPPLSSpec)
```

## AbstractCPPLSFit

```@docs
CPPLS.AbstractCPPLSFit
StatsAPI.coef(model::AbstractCPPLSFit) 
CPPLS.regression_coefficients(::AbstractCPPLSFit)
CPPLS.xbar(::AbstractCPPLSFit)
CPPLS.ybar(::AbstractCPPLSFit)
```

## CPPLSFit

```@docs
CPPLS.CPPLSFit
CPPLS.mode(::CPPLSFit)
StatsAPI.fitted(::CPPLSFit)
CPPLS.gamma(::CPPLSFit)
CPPLS.gammas
CPPLS.rhos
CPPLS.predictorlabels(::CPPLSFit)
CPPLS.projectionmatrix(::CPPLSFit)
CPPLS.responselabels(::CPPLSFit)
StatsAPI.residuals(::CPPLSFit)
CPPLS.sampleclasses(::CPPLSFit)
CPPLS.samplelabels(::CPPLSFit)
CPPLS.xscores(::CPPLSFit)
```

## CPPLSFitLight

```@docs
CPPLS.CPPLSFitLight
```
