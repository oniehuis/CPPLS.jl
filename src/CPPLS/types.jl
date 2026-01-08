"""
    AbstractCPPLS

Common supertype for Canonical Powered Partial Least Squares models. Any subtype must
expose at least the following fields so shared functions can operate generically:

- `regression_coefficients::Array{<:Real, 3}`
- `X_means::Matrix{<:Real}`
- `Y_means::Matrix{<:Real}`

Additionally, subtypes are expected to work with the exported generic helpers
`predict`, `predictonehot`, and `project`.
"""
abstract type AbstractCPPLS end


"""
    CPPLS{T1, T2}

Full CPPLS model storing all intermediate quantities required for diagnostics and
visualisation. `T1` is the floating-point element type used for continuous arrays,
`T2` is the integer type used for boolean-like masks.

# Fields
- `regression_coefficients::Array{T1, 3}` — cumulative regression matrices for the
   first `k = 1 … n_components` latent variables.
- `X_scores::Matrix{T1}` — predictor scores per component.
- `X_loadings::Matrix{T1}` — predictor loadings per component.
- `X_loading_weights::Matrix{T1}` — predictor weight vectors per component.
- `Y_scores::Matrix{T1}` — response scores derived from the fitted model.
- `Y_loadings::Matrix{T1}` — response loadings per component.
- `projection::Matrix{T1}` — mapping from centred predictors to component scores.
- `X_means::Matrix{T1}` — row vector of predictor means used for centering.
- `Y_means::Matrix{T1}` — row vector of response means used for centering.
- `fitted_values::Array{T1,3}` — fitted responses for the first `k` components.
- `residuals::Array{T1,3}` — residual cubes matching `fitted_values`.
- `X_variance::Vector{T1}` — variance explained in `X` per component.
- `X_total_variance::T1` — total variance present in the centred predictors.
- `gammas::Vector{T1}` — power-parameter selections per component.
- `canonical_correlations::Vector{T1}` — squared canonical correlations per component.
- `small_norm_indices::Matrix{T2}` — boolean mask of columns deflated to zero.
- `canonical_coefficients::Matrix{T1}` — canonical coefficient matrix from CCA.
- `canonical_coefficients_y::Matrix{T1}` — canonical coefficient matrix in Y-space from CCA.
- `W0_weights::Array{T1,3}` — initial CPPLS weight matrices per component.
- `Z::Array{T1,3}` — supervised predictor projections per component (`X_deflated * W0_weights`).
- `sample_labels::AbstractVector` — optional labels describing each observation.
- `predictor_labels::AbstractVector` — optional labels for predictor columns.
- `response_labels::AbstractVector` — labels for regression responses or DA classes.
- `analysis_mode::Symbol` — either `:regression` or `:discriminant`.
- `da_categories` — original categorical responses for DA models (`nothing` otherwise).
"""
struct CPPLS{
    T1<:Real,
    T2<:Integer,
    SL<:AbstractVector,
    PL<:AbstractVector,
    RL<:AbstractVector,
    DAC,
} <: AbstractCPPLS
    regression_coefficients::Array{T1,3}
    X_scores::Matrix{T1}
    X_loadings::Matrix{T1}
    X_loading_weights::Matrix{T1}
    Y_scores::Matrix{T1}
    Y_loadings::Matrix{T1}
    projection::Matrix{T1}
    X_means::Matrix{T1}
    Y_means::Matrix{T1}
    fitted_values::Array{T1,3}
    residuals::Array{T1,3}
    X_variance::Vector{T1}
    X_total_variance::T1
    gammas::Vector{T1}
    canonical_correlations::Vector{T1}
    small_norm_indices::Matrix{T2}
    canonical_coefficients::Matrix{T1}
    canonical_coefficients_y::Matrix{T1}
    W0_weights::Array{T1,3}
    Z::Array{T1,3}
    sample_labels::SL
    predictor_labels::PL
    response_labels::RL
    analysis_mode::Symbol
    da_categories::DAC
end

function CPPLS(
    regression_coefficients::Array{T1,3},
    X_scores::Matrix{T1},
    X_loadings::Matrix{T1},
    X_loading_weights::Matrix{T1},
    Y_scores::Matrix{T1},
    Y_loadings::Matrix{T1},
    projection::Matrix{T1},
    X_means::Matrix{T1},
    Y_means::Matrix{T1},
    fitted_values::Array{T1,3},
    residuals::Array{T1,3},
    X_variance::Vector{T1},
    X_total_variance::T1,
    gammas::Vector{T1},
    canonical_correlations::Vector{T1},
    small_norm_indices::Matrix{T2},
    canonical_coefficients::Matrix{T1},
    canonical_coefficients_y::Matrix{T1},
    W0_weights::Array{T1,3},
    Z::Array{T1,3};
    sample_labels::AbstractVector = String[],
    predictor_labels::AbstractVector = String[],
    response_labels::AbstractVector = String[],
    analysis_mode::Symbol = :regression,
    da_categories = nothing,
) where {T1<:Real,T2<:Integer}
    analysis_mode in (:regression, :discriminant) || throw(
        ArgumentError(
            "analysis_mode must be :regression or :discriminant, got $analysis_mode",
        ),
    )
    analysis_mode === :discriminant ||
        da_categories === nothing ||
        throw(
            ArgumentError("da_categories are only stored for discriminant analysis models"),
        )
    return CPPLS{
        T1,
        T2,
        typeof(sample_labels),
        typeof(predictor_labels),
        typeof(response_labels),
        typeof(da_categories),
    }(
        regression_coefficients,
        X_scores,
        X_loadings,
        X_loading_weights,
        Y_scores,
        Y_loadings,
        projection,
        X_means,
        Y_means,
        fitted_values,
        residuals,
        X_variance,
        X_total_variance,
        gammas,
        canonical_correlations,
        small_norm_indices,
        canonical_coefficients,
        canonical_coefficients_y,
        W0_weights,
        Z,
        sample_labels,
        predictor_labels,
        response_labels,
        analysis_mode,
        da_categories,
    )
end


"""
    CPPLSLight{T}

Memory-lean CPPLS variant retaining only the pieces needed for prediction. `T`
is the floating-point element type shared by all stored matrices.

# Fields
- `regression_coefficients::Array{T, 3}` — stacked regression matrices.
- `X_means::Matrix{T}` — predictor means copied from the training data.
- `Y_means::Matrix{T}` — response means copied from the training data.
- `analysis_mode::Symbol` — either `:regression` or `:discriminant`.
"""
struct CPPLSLight{T<:Real} <: AbstractCPPLS
    regression_coefficients::Array{T,3}
    X_means::Matrix{T}
    Y_means::Matrix{T}
    analysis_mode::Symbol
end
