# Types

`CPPLS` relies on two main custom types in the [`StatsAPI.fit`](@ref) and 
[`StatsAPI.predict`](@ref) workflow. [`CPPLSModel`](@ref) stores the user-controlled model 
specification, including the number of components, the gamma configuration, centering, the 
analysis mode, and numerical tolerances. [`CPPLSFit`](@ref) is the full fitted model 
returned by [`fit`](@ref); it contains regression coefficients, scores, labels, and other 
quantities needed for prediction, diagnostics, and plotting. A third container, 
[`CPPLSFitLight`](@ref), stores only the subset of fitted-model information needed for 
cross-validation. It is an internal optimization and is typically not accessed directly by 
users.

The package provides getters for fields that are commonly useful in downstream
work. For [`CPPLSModel`](@ref), the getters are [`gamma`](@ref), [`analysis_mode`](@ref), 
and [`ncomponents`](@ref). For [`CPPLSFit`](@ref), the getters are [`coef`](@ref), 
[`analysis_mode`](@ref), [`fitted`](@ref), [`gamma`](@ref), [`gammas`](@ref), [`rhos`](@ref),
[`predictorlabels`](@ref), [`responselabels`](@ref), [`residuals`](@ref),
[`sampleclasses`](@ref), [`samplelabels`](@ref), and [`xscores`](@ref).

Both container types contain additional named fields of possible interest. You can inspect
them with `propertynames(spec)` or `propertynames(model)` and access them directly via dot
notation, for example `spec.X_loading_weight_tolerance`.

## CPPLSModel

```@docs
CPPLS.CPPLSModel
CPPLS.gamma(::CPPLSModel)
CPPLS.analysis_mode(::CPPLSModel)
CPPLS.ncomponents(::CPPLSModel)
```

## AbstractCPPLSFit

```@docs
CPPLS.AbstractCPPLSFit
StatsAPI.coef(::AbstractCPPLSFit) 
CPPLS.coefall(::AbstractCPPLSFit)
```

## CPPLSFit

```@docs
CPPLS.CPPLSFit
StatsAPI.fitted(::CPPLSFit)
CPPLS.gamma(::CPPLSFit)
CPPLS.gammas(::CPPLSFit)
CPPLS.analysis_mode(::AbstractCPPLSFit)
CPPLS.predictorlabels(::CPPLSFit)
CPPLS.projectionmatrix(::CPPLSFit)
StatsAPI.residuals(::CPPLSFit)
CPPLS.responselabels(::CPPLSFit)
CPPLS.rhos(::CPPLSFit)
CPPLS.sampleclasses(::CPPLSFit)
CPPLS.samplelabels(::CPPLSFit)
CPPLS.xscores(::CPPLSFit)
```

## CPPLSFitLight

```@docs
CPPLS.CPPLSFitLight
```
