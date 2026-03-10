"""
    AbstractCPPLSFit

Common supertype for fitted Canonical Powered Partial Least Squares models. Any subtype
must expose at least the following fields so shared functions can operate generically:
`B::Array{<:Real,3}`, `X_bar::Matrix{<:Real}`, and `Y_bar::Matrix{<:Real}`. Additionally,
subtypes are expected to work with the exported generic helpers `predict`, `predictonehot`,
and `project`.
"""
abstract type AbstractCPPLSFit end


"""
    CPPLSSpec{T}

Model specification for CPPLS fits. Stores hyperparameters and numerical tolerances but no 
data-dependent quantities. `T` is the element type of `gamma`, which can be a fixed value, 
a `(lo, hi)` bounds tuple, a range (e.g., 0.01:0.01:1.0), or a vector mixing these forms. 
The fields are the number of components (`n_components`), power settings (`gamma`), 
centering flag (`center`), tolerances for deflation and weights (`X_tolerance`, 
`X_loading_weight_tolerance`), stabilization for score norms (`t_squared_norm_tolerance`), o
ptimizer tolerances (`gamma_rel_tol`, `gamma_abs_tol`), and the analysis mode 
(`analysis_mode`), which is either `:regression` or `:discriminant`.
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

n_components(spec::CPPLSSpec) = spec.n_components
gamma(spec::CPPLSSpec) = spec.gamma
center(spec::CPPLSSpec) = spec.center
X_tolerance(spec::CPPLSSpec) = spec.X_tolerance
X_loading_weight_tolerance(spec::CPPLSSpec) = spec.X_loading_weight_tolerance
t_squared_norm_tolerance(spec::CPPLSSpec) = spec.t_squared_norm_tolerance
gamma_rel_tol(spec::CPPLSSpec) = spec.gamma_rel_tol
gamma_abs_tol(spec::CPPLSSpec) = spec.gamma_abs_tol
analysis_mode(spec::CPPLSSpec) = spec.analysis_mode

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

Model specification for CPPLS fits. Stores hyperparameters and numerical tolerances but no 
data-dependent quantities. Use with `fit`.
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


"""
    CPPLSFit{T1, T2}

Full CPPLS model storing all intermediate quantities required for diagnostics and
visualisation. `T1` is the floating-point element type used for continuous arrays,
`T2` is the integer type used for boolean-like masks.

# Fields
- `B::Array{T1,3}` — cumulative regression matrices for the
   first `k = 1 … n_components` latent variables.
- `T::Matrix{T1}` — predictor scores per component.
- `P::Matrix{T1}` — predictor loadings per component.
- `W_comp::Matrix{T1}` — predictor weight vectors per component.
- `U::Matrix{T1}` — response scores derived from the fitted model.
- `C::Matrix{T1}` — response loadings per component.
- `R::Matrix{T1}` — mapping from centred predictors to component scores.
- `X_bar::Matrix{T1}` — row vector of predictor means used for centering.
- `Y_bar::Matrix{T1}` — row vector of response means used for centering.
- `Y_hat::Array{T1,3}` — fitted responses for the first `k` components.
- `F::Array{T1,3}` — residual cubes matching `Y_hat`.
- `X_var::Vector{T1}` — variance explained in `X` per component.
- `X_var_total::T1` — total variance present in the centred predictors.
- `gamma::Vector{T1}` — power-parameter selections per component.
- `rho::Vector{T1}` — squared canonical correlations per component.
- `zero_mask::Matrix{T2}` — boolean mask of columns deflated to zero.
- `a::Matrix{T1}` — canonical coefficient matrix from CCA.
- `b::Matrix{T1}` — canonical coefficient matrix in Y-space from CCA.
- `W0::Array{T1,3}` — initial CPPLS weight matrices per component.
- `Z::Array{T1,3}` — supervised predictor projections per component (`X_def * W0`).
- `sample_labels::AbstractVector` — optional labels describing each observation.
- `predictor_labels::AbstractVector` — optional labels for predictor columns.
- `response_labels::AbstractVector` — labels for regression responses or DA classes.
- `analysis_mode::Symbol` — either `:regression` or `:discriminant`.
- `sample_classes` — original categorical responses for DA models (`nothing` otherwise).
"""
struct CPPLSFit{
    T1<:Real,
    T2<:Integer,
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:AbstractVector,
    T6<:Union{AbstractVector,Nothing}
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
    zero_mask::Matrix{T2}
    a::Matrix{T1}
    b::Matrix{T1}
    W0::Array{T1,3}
    Z::Array{T1,3}
    sample_labels::T3
    predictor_labels::T4
    response_labels::T5
    analysis_mode::Symbol
    sample_classes::T6
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
    zero_mask::Matrix{T2},
    a::Matrix{T1},
    b::Matrix{T1},
    W0::Array{T1,3},
    Z::Array{T1,3};
    sample_labels::AbstractVector=String[],
    predictor_labels::AbstractVector=String[],
    response_labels::AbstractVector=String[],
    analysis_mode::Symbol=:regression,
    sample_classes=nothing
    ) where {
        T1<:Real,
        T2<:Integer
    }

    analysis_mode in (:regression, :discriminant) || throw(ArgumentError(
            "analysis_mode must be :regression or :discriminant, got $analysis_mode"))

    analysis_mode === :discriminant || sample_classes === nothing || throw(ArgumentError(
        "sample_classes are only stored for discriminant analysis models"))

    CPPLSFit{
        T1,
        T2,
        typeof(sample_labels),
        typeof(predictor_labels),
        typeof(response_labels),
        typeof(sample_classes),
    }(
        B,
        T,
        P,
        W_comp,
        U,
        C,
        R,
        X_bar,
        Y_bar,
        Y_hat,
        F,
        X_var,
        X_var_total,
        gamma,
        rho,
        zero_mask,
        a,
        b,
        W0,
        Z,
        sample_labels,
        predictor_labels,
        response_labels,
        analysis_mode,
        sample_classes,
    )
end

"""
    gamma(cpplsfit::CPPLSFit)

Return the power-parameter values selected during fitting.
"""
gamma(cpplsfit::CPPLSFit) = cpplsfit.gamma

"""
    sample_labels(cpplsfit::CPPLSFit)

Return the stored sample labels for the fitted model.
"""
sample_labels(cpplsfit::CPPLSFit) = cpplsfit.sample_labels

"""
    predictor_labels(cpplsfit::CPPLSFit)

Return the stored predictor labels for the fitted model.
"""
predictor_labels(cpplsfit::CPPLSFit) = cpplsfit.predictor_labels

"""
    response_labels(cpplsfit::CPPLSFit)

Return the response labels (response names or class names) for the fitted model.
"""
response_labels(cpplsfit::CPPLSFit) = cpplsfit.response_labels

"""
    analysis_mode(cpplsfit::CPPLSFit)

Return the analysis mode (`:regression` or `:discriminant`) for the fitted model.
"""
analysis_mode(cpplsfit::CPPLSFit) = cpplsfit.analysis_mode

"""
    sample_classes(cpplsfit::CPPLSFit)

Return the per-sample class labels stored for discriminant analysis models, or `nothing`
for regression fits.
"""
sample_classes(cpplsfit::CPPLSFit) = cpplsfit.sample_classes

"""
    X_scores(cpplsfit::CPPLSFit)

Return the predictor score matrix `T` for the fitted model.
"""
X_scores(cpplsfit::CPPLSFit) = cpplsfit.T


"""
    CPPLSFitLight{T}

Memory-lean CPPLS variant retaining only the pieces needed for prediction. `T`
is the floating-point element type shared by all stored matrices.

# Fields
- `B::Array{T,3}` — stacked regression matrices.
- `X_bar::Matrix{T}` — predictor means copied from the training data.
- `Y_bar::Matrix{T}` — response means copied from the training data.
- `analysis_mode::Symbol` — either `:regression` or `:discriminant`.
"""
struct CPPLSFitLight{T<:Real} <: AbstractCPPLSFit
    B::Array{T,3}
    X_bar::Matrix{T}
    Y_bar::Matrix{T}
    analysis_mode::Symbol
end
