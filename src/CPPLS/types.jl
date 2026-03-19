"""
    CPPLSSpec{T}

Model specification passed to `fit`. A `CPPLSSpec` stores the user-controlled settings
for CPPLS fitting, most importantly `n_components`, `gamma`, `center`, and
`analysis_mode`.
"""
struct CPPLSSpec{T}
    n_components::Int
    gamma::T
    center::Bool
    X_tolerance::Float64
    X_loading_weight_tolerance::Float64
    t_squared_norm_tolerance::Float64
    gamma_rel_tol::Float64
    gamma_abs_tol::Float64
    analysis_mode::Symbol
end

"""
    CPPLSSpec(; 
        n_components::Integer=2, 
        gamma=0.5, 
        center::Bool=true, 
        X_tolerance::Real=1e-12, 
        X_loading_weight_tolerance::Real=eps(Float64), 
        t_squared_norm_tolerance::Real=1e-10, 
        gamma_rel_tol::Real=1e-6, 
        gamma_abs_tol::Real=1e-12, 
        analysis_mode::Symbol=:regression
    )

Construct a model specification for `fit`. The most commonly adjusted settings are
`n_components`, `gamma`, `center`, and `analysis_mode`. `gamma` may be a fixed value, a
`(lo, hi)` interval, or a collection of such candidates used during fitting.
"""
function CPPLSSpec(;
    n_components::T1=2,
    gamma=0.5,
    center::Bool=true,
    X_tolerance::T2=1e-12,
    X_loading_weight_tolerance::T3=eps(Float64),
    t_squared_norm_tolerance::T4=1e-10,
    gamma_rel_tol::T5=1e-6,
    gamma_abs_tol::T6=1e-12,
    analysis_mode::Symbol=:regression
) where {
        T1<:Integer, 
        T2<:Real, 
        T3<:Real, 
        T4<:Real, 
        T5<:Real, 
        T6<:Real
    }

    n_components > 0 || throw(ArgumentError("n_components must be greater than zero"))
    analysis_mode in (:regression, :discriminant) || throw(ArgumentError(
            "analysis_mode must be :regression or :discriminant, got $analysis_mode"))

    CPPLSSpec(
        Int(n_components),
        gamma,
        center,
        Float64(X_tolerance),
        Float64(X_loading_weight_tolerance),
        Float64(t_squared_norm_tolerance),
        Float64(gamma_rel_tol),
        Float64(gamma_abs_tol),
        analysis_mode
    )
end

function Base.show(io::IO, spec::CPPLSSpec)
    print(io, "CPPLSSpec(",
        "n_components=", spec.n_components,
        ", gamma=", repr(spec.gamma),
        ", center=", spec.center,
        ", analysis_mode=", spec.analysis_mode,
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", spec::CPPLSSpec)
    println(io, "CPPLSSpec")
    println(io, "  n_components: ", spec.n_components)
    println(io, "  gamma: ", repr(spec.gamma))
    println(io, "  center: ", spec.center)
    print(io, "  analysis_mode: ", spec.analysis_mode)
end

"""
    gamma(spec::CPPLSSpec)

Return the gamma requested in the specification.
"""
gamma(spec::CPPLSSpec) = spec.gamma

"""
    n_components(spec::CPPLSSpec)

Return the number of components requested in the specification.
"""
n_components(spec::CPPLSSpec) = spec.n_components

"""
    analysis_mode(spec::CPPLSSpec)

Return the analysis mode requested in the specification.
"""
analysis_mode(spec::CPPLSSpec) = spec.analysis_mode

"""
    AbstractCPPLSFit

Common supertype for fitted CPPLS models that share the fields B, X_bar, Y_bar,
and analysis_mode.
"""
abstract type AbstractCPPLSFit end

"""
    coef(model::AbstractCPPLSFit)
    coef(model::AbstractCPPLSFit, n_components::Integer)

Return the regression coefficient matrix for the final (or requested) number of components.
"""
coef(model::AbstractCPPLSFit) = @views model.B[:, :, end]
coef(model::AbstractCPPLSFit, n_components::Integer) = @views model.B[:, :, n_components]

"""
    regression_coefficients(model::AbstractCPPLSFit)

Return the regression coefficients for the fitted model.
"""
regression_coefficients(model::AbstractCPPLSFit) = model.B

"""
    X_bar(cpplsfit::AbstractCPPLSFit)

Return the predictor mean vector for the fitted model.
"""
X_bar(cpplsfit::AbstractCPPLSFit) = cpplsfit.X_bar

"""
    Y_bar(cpplsfit::AbstractCPPLSFit)

Return the response mean vector for the fitted model.
"""
Y_bar(cpplsfit::AbstractCPPLSFit) = cpplsfit.Y_bar

"""
    CPPLSFit{T1, T2}

Full fitted CPPLS model returned by `fit`. This type stores regression coefficients
together with the intermediate quantities needed for diagnostics, projections, and
plotting.

Most users will work with a `CPPLSFit` through `predict`, `project`, `coef`, `fitted`,
`residuals`, `X_scores`, and the various label getters rather than by accessing fields
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
    X_bar::Matrix{T1}
    Y_bar::Matrix{T1}
    Y_hat::Array{T1,3}
    F::Array{T1,3}
    X_var::Vector{T1}
    X_var_total::T1
    gamma::Vector{T1}
    rho::Vector{T1}
    gamma_search_gammas::Matrix{T1}
    gamma_search_rhos::Matrix{T1}
    zero_mask::Matrix{T2}
    a::Matrix{T1}
    b::Matrix{T1}
    W0::Array{T1,3}
    Z::Array{T1,3}
    samplelabels::T3
    predictorlabels::T4
    responselabels::T5
    analysis_mode::Symbol
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
    X_bar::Matrix{T1},
    Y_bar::Matrix{T1},
    Y_hat::Array{T1,3},
    F::Array{T1,3},
    X_var::Vector{T1},
    X_var_total::T1,
    gamma::Vector{T1},
    rho::Vector{T1},
    gamma_search_gammas::Matrix{T1},
    gamma_search_rhos::Matrix{T1},
    zero_mask::Matrix{T2},
    a::Matrix{T1},
    b::Matrix{T1},
    W0::Array{T1,3},
    Z::Array{T1,3};
    samplelabels::T3=String[],
    predictorlabels::T4=String[],
    responselabels::T5=String[],
    analysis_mode::Symbol=:regression,
    sampleclasses::T6=nothing
) where {
        T1<:Real,
        T2<:Integer,
        T3<:AbstractVector,
        T4<:AbstractVector,
        T5<:AbstractVector,
        T6<:Union{AbstractVector, Nothing}
    }

    analysis_mode in (:regression, :discriminant) || throw(ArgumentError(
            "analysis_mode must be :regression or :discriminant, got $analysis_mode"))

    analysis_mode ≡ :discriminant || isnothing(sampleclasses) || throw(ArgumentError(
        "sampleclasses are only stored for discriminant analysis models"))

    CPPLSFit{T1, T2, T3, T4, T5, T6}(B, T, P, W_comp, U, C, R, X_bar, Y_bar, Y_hat, F, 
        X_var, X_var_total, gamma, rho, gamma_search_gammas, gamma_search_rhos,
        zero_mask, a, b, W0, Z, samplelabels, predictorlabels, responselabels,
        analysis_mode, sampleclasses)
end

function Base.show(io::IO, model::CPPLSFit)
    print(io, "CPPLSFit(",
        "mode=", model.analysis_mode,
        ", samples=", size(model.T, 1),
        ", predictors=", size(model.B, 1),
        ", responses=", size(model.B, 2),
        ", components=", size(model.B, 3),
        ", gamma=", repr(model.gamma),
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", model::CPPLSFit)
    println(io, "CPPLSFit")
    println(io, "  mode: ", model.analysis_mode)
    println(io, "  samples: ", size(model.T, 1))
    println(io, "  predictors: ", size(model.B, 1))
    println(io, "  responses: ", size(model.B, 2))
    println(io, "  components: ", size(model.B, 3))
    println(io, "  gamma: ", repr(model.gamma))
end

"""
    analysis_mode(cpplsfit::CPPLSFit)

Return the analysis mode for the fitted model.
"""
analysis_mode(cpplsfit::CPPLSFit) = cpplsfit.analysis_mode

"""
    fitted(model::CPPLSFit)
    fitted(model::CPPLSFit, n_components::Integer)

Return the fitted response matrix for the final (or requested) number of components.
"""
fitted(model::CPPLSFit) = @views model.Y_hat[:, :, end]
fitted(model::CPPLSFit, n_components::Integer) = @views model.Y_hat[:, :, n_components]

"""
    gamma(cpplsfit::CPPLSFit)

Return the power-parameter values selected during fitting.
"""
gamma(cpplsfit::CPPLSFit) = cpplsfit.gamma

"""
    gamma_search_gammas(cpplsfit::CPPLSFit)
    gamma_search_gammas(cpplsfit::CPPLSFit, latent_variable::Integer)

Return the matrix of per-candidate gamma values considered during fitting, or the column
for a specific latent variable.
"""
gamma_search_gammas(cpplsfit::CPPLSFit) = cpplsfit.gamma_search_gammas
gamma_search_gammas(cpplsfit::CPPLSFit, latent_variable::Integer) =
    @views cpplsfit.gamma_search_gammas[:, latent_variable]

"""
    gamma_search_rhos(cpplsfit::CPPLSFit)
    gamma_search_rhos(cpplsfit::CPPLSFit, latent_variable::Integer)

Return the matrix of per-candidate squared canonical correlations considered during
fitting, or the column for a specific latent variable.
"""
gamma_search_rhos(cpplsfit::CPPLSFit) = cpplsfit.gamma_search_rhos
gamma_search_rhos(cpplsfit::CPPLSFit, latent_variable::Integer) =
    @views cpplsfit.gamma_search_rhos[:, latent_variable]

"""
    predictorlabels(cpplsfit::CPPLSFit)

Return the stored predictor labels for the fitted model.
"""
predictorlabels(cpplsfit::CPPLSFit) = cpplsfit.predictorlabels

"""
    projectionmatrix(cpplsfit::CPPLSFit)

Return the projection matrix `R` for the fitted model.
"""
projectionmatrix(cpplsfit::CPPLSFit) = cpplsfit.R

"""
    residuals(model::CPPLSFit)
    residuals(model::CPPLSFit, n_components::Integer)

Return the response residual matrix for the final (or requested) number of components.
"""
residuals(model::CPPLSFit) = @views model.F[:, :, end]
residuals(model::CPPLSFit, n_components::Integer) = @views model.F[:, :, n_components]

"""
    responselabels(cpplsfit::CPPLSFit)

Return the response labels (response names or class names) for the fitted model.
"""
responselabels(cpplsfit::CPPLSFit) = cpplsfit.responselabels

"""
    sampleclasses(cpplsfit::CPPLSFit)

Return the per-sample class labels stored for discriminant analysis models, or `nothing`
for regression fits.
"""
sampleclasses(cpplsfit::CPPLSFit) = cpplsfit.sampleclasses

"""
    samplelabels(cpplsfit::CPPLSFit)

Return the stored sample labels for the fitted model.
"""
samplelabels(cpplsfit::CPPLSFit) = cpplsfit.samplelabels

"""
    X_scores(cpplsfit::CPPLSFit)

Return the predictor score matrix `T` for the fitted model.
"""
X_scores(cpplsfit::CPPLSFit) = cpplsfit.T

"""
    CPPLSFitLight{T}

Reduced fitted CPPLS model that retains only the information needed for prediction. This
type is used mainly for efficient internal prediction during cross-validation.

Most users will work with `CPPLSFit` instead.
"""
struct CPPLSFitLight{T<:Real} <: AbstractCPPLSFit
    B::Array{T,3}
    X_bar::Matrix{T}
    Y_bar::Matrix{T}
    analysis_mode::Symbol
end

function Base.show(io::IO, model::CPPLSFitLight)
    print(io, "CPPLSFitLight(",
        "mode=", model.analysis_mode,
        ", predictors=", size(model.B, 1),
        ", responses=", size(model.B, 2),
        ", components=", size(model.B, 3),
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", model::CPPLSFitLight)
    println(io, "CPPLSFitLight")
    println(io, "  mode: ", model.analysis_mode)
    println(io, "  predictors: ", size(model.B, 1))
    println(io, "  responses: ", size(model.B, 2))
    print(io, "  components: ", size(model.B, 3))
end
