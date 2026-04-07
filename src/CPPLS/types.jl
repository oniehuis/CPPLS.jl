"""
    CPPLSModel{T}

Model specification passed to `fit`. A `CPPLSModel` stores the user-controlled settings
for CPPLS fitting, most importantly `ncomponents`, `gamma`, centering and scaling of predictor 
and response variables, and `mode`.
"""
struct CPPLSModel{T}
    ncomponents::Int
    gamma::T
    center_X::Bool
    scale_X::Bool
    scale_Yprim::Bool
    X_tolerance::Float64
    X_loading_weight_tolerance::Float64
    t_squared_norm_tolerance::Float64
    gamma_rel_tol::Float64
    gamma_abs_tol::Float64
    mode::Symbol
end

"""
    CPPLSModel(; 
        ncomponents::Integer=2, 
        gamma=0.5,
        center_X::Bool=true,
        scale_X::Bool=false,
        scale_Yprim::Bool=false,
        X_tolerance::Real=1e-12, 
        X_loading_weight_tolerance::Real=eps(Float64), 
        t_squared_norm_tolerance::Real=1e-10, 
        gamma_rel_tol::Real=1e-6, 
        gamma_abs_tol::Real=1e-12, 
        mode::Symbol=:regression
    )

Construct a model specification for `fit`. The most commonly adjusted settings are
`ncomponents`, `gamma`, and `mode`. `gamma` may be a fixed value, a
`(lo, hi)` interval, or a collection of such candidates used during fitting.
"""
function CPPLSModel(;
    ncomponents::T1=2,
    gamma=0.5,
    center_X::Bool=true,
    scale_X::Bool=false,
    scale_Yprim::Bool=false,
    X_tolerance::T2=1e-12,
    X_loading_weight_tolerance::T3=eps(Float64),
    t_squared_norm_tolerance::T4=1e-10,
    gamma_rel_tol::T5=1e-6,
    gamma_abs_tol::T6=1e-12,
    mode::Symbol=:regression
) where {
        T1<:Integer, 
        T2<:Real, 
        T3<:Real, 
        T4<:Real, 
        T5<:Real, 
        T6<:Real
    }

    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))
    mode in (:regression, :discriminant) || throw(ArgumentError(
            "mode must be :regression or :discriminant, got $mode"))

    CPPLSModel(
        Int(ncomponents),
        gamma,
        center_X,
        scale_X,
        scale_Yprim,
        Float64(X_tolerance),
        Float64(X_loading_weight_tolerance),
        Float64(t_squared_norm_tolerance),
        Float64(gamma_rel_tol),
        Float64(gamma_abs_tol),
        mode
    )
end

function Base.show(io::IO, spec::CPPLSModel)
    print(io, "CPPLSModel(",
        "ncomponents=", spec.ncomponents,
        ", gamma=", repr(spec.gamma),
        ", center_X=", spec.center_X,
        ", scale_X=", spec.scale_X,
        ", scale_Yprim=", spec.scale_Yprim,
        ", mode=", spec.mode,
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", spec::CPPLSModel)
    println(io, "CPPLSModel")
    println(io, "  ncomponents: ", spec.ncomponents)
    println(io, "  gamma: ", repr(spec.gamma))
    println(io, "  center_X: ", spec.center_X)
    println(io, "  scale_X: ", spec.scale_X)
    println(io, "  scale_Yprim: ", spec.scale_Yprim)
    print(io, "  mode: ", spec.mode)
end

"""
    gamma(model::CPPLSModel)

Return the gamma requested in the model.
"""
gamma(model::CPPLSModel) = model.gamma

"""
    ncomponents(model::CPPLSModel)

Return the number of components requested in the model.
"""
ncomponents(model::CPPLSModel) = model.ncomponents

"""
    mode(model::CPPLSModel)

Return the analysis mode requested in the model.
"""
mode(model::CPPLSModel) = model.mode

"""
    AbstractCPPLSFit

Common supertype for fitted CPPLS models that share the fields B, X_bar, Y_bar,
and mode. 
"""
abstract type AbstractCPPLSFit end

"""
    coef(fit::AbstractCPPLSFit)
    coef(fit::AbstractCPPLSFit, ncomponents::Integer)

Return the regression coefficient matrix for the final (or requested) number of components.
"""
coef(fit::AbstractCPPLSFit) = @views fit.B[:, :, end]
coef(fit::AbstractCPPLSFit, ncomponents::Integer) = @views fit.B[:, :, ncomponents]

"""
    coefall(fit::AbstractCPPLSFit)

Return the regression coefficients for the fitted model.
"""
coefall(fit::AbstractCPPLSFit) = fit.B

"""
    xmean(mf::AbstractCPPLSFit)

Return the predictor mean vector for the fitted model.
"""
xmean(mf::AbstractCPPLSFit) = mf.X_mean

"""
    xstd(mf::AbstractCPPLSFit)

Return the predictor standard deviation vector for the fitted model.
"""
xstd(mf::AbstractCPPLSFit) = mf.X_std

"""
    ystd(mf::AbstractCPPLSFit)

Return the response standard deviation vector for the fitted model.
"""
ystd(mf::AbstractCPPLSFit) = mf.Yprim_std

"""
    CPPLSFit{T1, T2}

Full fitted CPPLS model returned by `fit`. This type stores regression coefficients
together with the intermediate quantities needed for diagnostics, projections, and
plotting.

Most users will work with a `CPPLSFit` through `predict`, `project`, `coef`, `fitted`,
`residuals`, `xscores`, and the various label getters rather than by accessing fields
directly.
"""
struct CPPLSFit{
    T1<:Real,
    T2<:Integer,
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:AbstractVector,
    T6<:Union{AbstractVector, Nothing}
} <: AbstractCPPLSFit

    B::Array{T1,3}
    T::Matrix{T1}
    P::Matrix{T1}
    W_comp::Matrix{T1}
    U::Matrix{T1}
    C::Matrix{T1}
    R::Matrix{T1}
    Y_hat::Array{T1,3}
    F::Array{T1,3}
    X_var::Vector{T1}
    X_var_total::T1
    gamma::Vector{T1}
    rho::Vector{T1}
    gammas::Matrix{T1}
    rhos::Matrix{T1}
    zero_mask::Matrix{T2}
    a::Matrix{T1}
    b::Matrix{T1}
    W0::Array{T1,3}
    X_mean::Vector{T1}
    X_std::Vector{T1}
    Yprim_std::Vector{T1}
    samplelabels::T3
    predictorlabels::T4
    responselabels::T5
    mode::Symbol
    sampleclasses::T6
end

function CPPLSFit(
    B::Array{T1,3},
    T::Matrix{T1},
    P::Matrix{T1},
    W_comp::Matrix{T1},
    U::Matrix{T1},
    C::Matrix{T1},
    R::Matrix{T1},
    Y_hat::Array{T1,3},
    F::Array{T1,3},
    X_var::Vector{T1},
    X_var_total::T1,
    gamma::Vector{T1},
    rho::Vector{T1},
    gammas::Matrix{T1},
    rhos::Matrix{T1},
    zero_mask::Matrix{T2},
    a::Matrix{T1},
    b::Matrix{T1},
    W0::Array{T1,3},
    X_mean::Vector{T1},
    X_std::Vector{T1},
    Yprim_std::Vector{T1};
    samplelabels::T3=String[],
    predictorlabels::T4=String[],
    responselabels::T5=String[],
    mode::Symbol=:regression,
    sampleclasses::T6=nothing
) where {
        T1<:Real,
        T2<:Integer,
        T3<:AbstractVector,
        T4<:AbstractVector,
        T5<:AbstractVector,
        T6<:Union{AbstractVector, Nothing}
    }

    mode in (:regression, :discriminant) || throw(ArgumentError(
            "mode must be :regression or :discriminant, got $mode"))

    mode ≡ :discriminant || isnothing(sampleclasses) || throw(ArgumentError(
        "sampleclasses are only stored for discriminant analysis models"))

    CPPLSFit{T1, T2, T3, T4, T5, T6}(B, T, P, W_comp, U, C, R, Y_hat, 
        F, X_var, X_var_total, gamma, rho, gammas, rhos, zero_mask, a, b, W0, 
        X_mean, X_std, Yprim_std,
        samplelabels, predictorlabels, responselabels, mode, sampleclasses)
end

function Base.show(io::IO, mf::CPPLSFit)
    print(io, "CPPLSFit(",
        "mode=", mf.mode,
        ", samples=", size(mf.T, 1),
        ", predictors=", size(mf.B, 1),
        ", responses=", size(mf.B, 2),
        ", components=", size(mf.B, 3),
        ", gamma=", repr(mf.gamma),
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", mf::CPPLSFit)
    println(io, "CPPLSFit")
    println(io, "  mode: ", mf.mode)
    println(io, "  samples: ", size(mf.T, 1))
    println(io, "  predictors: ", size(mf.B, 1))
    println(io, "  responses: ", size(mf.B, 2))
    println(io, "  components: ", size(mf.B, 3))
    println(io, "  gamma: ", repr(mf.gamma))
end

"""
    mode(fit::CPPLSFit)

Return the analysis mode for the fitted model.
"""
mode(fit::CPPLSFit) = fit.mode

"""
    fitted(fit::CPPLSFit)
    fitted(fit::CPPLSFit, ncomponents::Integer)

Return the fitted response matrix for the final (or requested) number of components.
"""
fitted(fit::CPPLSFit) = @views fit.Y_hat[:, :, end]
fitted(fit::CPPLSFit, ncomponents::Integer) = @views fit.Y_hat[:, :, ncomponents]

"""
    gamma(fit::CPPLSFit)

Return the power-parameter values selected during fitting.
"""
gamma(fit::CPPLSFit) = fit.gamma

"""
    gammas(fit::CPPLSFit)
    gammas(fit::CPPLSFit, latent_variable::Integer)

Return the matrix of per-candidate gamma values considered during fitting, or the column
for a specific latent variable.
"""
gammas(fit::CPPLSFit) = fit.gammas
gammas(fit::CPPLSFit, latent_variable::Integer) = @views fit.gammas[:, latent_variable]

"""
    rhos(fit::CPPLSFit)
    rhos(fit::CPPLSFit, latent_variable::Integer)

Return the matrix of per-candidate squared canonical correlations considered during
fitting, or the column for a specific latent variable.
"""
rhos(fit::CPPLSFit) = fit.rhos
rhos(fit::CPPLSFit, latent_variable::Integer) = @views fit.rhos[:, latent_variable]

"""
    predictorlabels(fit::CPPLSFit)

Return the stored predictor labels for the fitted model.
"""
predictorlabels(fit::CPPLSFit) = fit.predictorlabels

"""
    projectionmatrix(fit::CPPLSFit)

Return the projection matrix `R` for the fitted model.
"""
projectionmatrix(fit::CPPLSFit) = fit.R

"""
    residuals(fit::CPPLSFit)
    residuals(fit::CPPLSFit, ncomponents::Integer)

Return the response residual matrix for the final (or requested) number of components.
"""
residuals(fit::CPPLSFit) = @views fit.F[:, :, end]
residuals(fit::CPPLSFit, ncomponents::Integer) = @views fit.F[:, :, ncomponents]

"""
    responselabels(fit::CPPLSFit)

Return the response labels (response names or class names) for the fitted model.
"""
responselabels(fit::CPPLSFit) = fit.responselabels

"""
    sampleclasses(fit::CPPLSFit)

Return the per-sample class labels stored for discriminant analysis models, or `nothing`
for regression fits.
"""
sampleclasses(fit::CPPLSFit) = fit.sampleclasses

"""
    samplelabels(fit::CPPLSFit)

Return the stored sample labels for the fitted model.
"""
samplelabels(fit::CPPLSFit) = fit.samplelabels

"""
    xscores(fit::CPPLSFit)
    xscores(fit::CPPLSFit, comp::Integer)
    xscores(fit::CPPLSFit, comps::AbstractUnitRange{<:Integer})
    xscores(fit::CPPLSFit, comps::AbstractVector{<:Integer})

Return the predictor score matrix `T` for the fitted model, or a subset of its columns:

- `xscores(fit)` returns the full score matrix (all components).
- `xscores(fit, comp)` returns the column for a single component as a vector.
- `xscores(fit, comps::AbstractUnitRange)` returns a matrix with the selected range of components.
- `xscores(fit, comps::AbstractVector)` returns a matrix with the specified components (in given order).

Throws an error if any requested component index is out of bounds.
"""
xscores(fit::CPPLSFit) = fit.T

function xscores(fit::CPPLSFit, comp::Integer)
    ncomp = size(fit.T, 2)
    1 ≤ comp ≤ ncomp || throw(
        ArgumentError("Component index $comp out of bounds (1:$ncomp)"))
    view(fit.T, :, comp)
end

function xscores(fit::CPPLSFit, comps::AbstractUnitRange{<:Integer})
    ncomp = size(fit.T, 2)
    (1 ≤ first(comps) ≤ ncomp && 1 ≤ last(comps) ≤ ncomp) || throw(
        ArgumentError("Component range $(comps) out of bounds (1:$ncomp)"))
    view(fit.T, :, comps)
end

function xscores(fit::CPPLSFit, comps::AbstractVector{<:Integer})
    ncomp = size(fit.T, 2)
    all(1 .≤ comps .≤ ncomp) || throw(
        ArgumentError("Component indices $(comps) out of bounds (1:$ncomp)"))
    view(fit.T, :, comps)
end

"""
    CPPLSFitLight{T}

Reduced fitted CPPLS model that retains only the information needed for prediction. This
type is used mainly for efficient internal prediction during cross-validation.

Most users will work with `CPPLSFit` instead.
"""
struct CPPLSFitLight{T<:Real} <: AbstractCPPLSFit
    B::Array{T, 3}   
    X_mean::Vector{T}
    X_std::Vector{T}
    Yprim_std::Vector{T}
    mode::Symbol
end

function Base.show(io::IO, mf::CPPLSFitLight)
    print(io, "CPPLSFitLight(",
        "mode=", mf.mode,
        ", predictors=", size(mf.B, 1),
        ", responses=", size(mf.B, 2),
        ", components=", size(mf.B, 3),
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", mf::CPPLSFitLight)
    println(io, "CPPLSFitLight")
    println(io, "  mode: ", mf.mode)
    println(io, "  predictors: ", size(mf.B, 1))
    println(io, "  responses: ", size(mf.B, 2))
    print(io, "  components: ", size(mf.B, 3))
end
