"""
    fit(model::CPPLSSpec,
        X::AbstractMatrix{<:Real},
        Y_prim::AbstractMatrix{<:Real};
        kwargs...
    )
    fit(model::CPPLSSpec,
        X::AbstractMatrix{<:Real},
        sample_classes::AbstractCategoricalArray{T,1,R,V,C,U};
        kwargs...
    ) where {T,R,V,C,U}
    fit(model::CPPLSSpec,
        X::AbstractMatrix{<:Real},
        sample_classes::AbstractVector;
        kwargs...
    )

Fit a CPPLS model using the StatsAPI entry point and an explicit CPPLSSpec. The model
specification supplies the number of components, the gamma configuration, centering, the
analysis mode, and all numerical tolerances, while the call to `fit` supplies data,
optional weights, auxiliary responses, and label metadata.

When `Y_prim` is provided, it is treated as the primary response block. When
`sample_classes` is provided, the labels are converted to a one-hot response matrix,
class names are inferred as response labels, and the fit is forced to discriminant
analysis; `model.analysis_mode` must be `:discriminant` or an ArgumentError is thrown.

The `gamma` setting in `model` may be a fixed scalar, a `(lo, hi)` tuple, or a vector
mixing scalars and tuples. Non-scalar settings trigger per-component selection by
choosing the value that yields the largest leading canonical correlation between the
supervised projection and the primary responses, and the resulting gamma values are
stored in the fitted model.

Keyword arguments accepted by `fit` include `obs_weights` for per-sample weighting,
`Y_aux` for auxiliary response columns, and optional `sample_labels`, `predictor_labels`,
`response_labels`, and `sample_classes` metadata for diagnostics and plotting. `Y_aux`
must have the same number of rows as `X` and is concatenated to `Y_prim` internally to
build the supervised projection, while prediction targets always remain the primary
responses.

The return value is a `CPPLSFit` containing scores, loadings, regression coefficients,
and the metadata needed for downstream prediction and diagnostics. Use `CPPLS.fit` or
`StatsAPI.fit` when disambiguation is required in your namespace.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit), 
[`CPPLSSpec`](@ref CPPLS.CPPLSSpec), 
[`gamma`](@ref CPPLS.gamma(::CPPLSFit)), 
[`invfreqweights`](@ref invfreqweights(::AbstractVector))
[`predictor_labels`](@ref predictor_labels(::CPPLSFit)),
[`response_labels`](@ref response_labels(::CPPLSFit)),
[`sample_classes`](@ref sample_classes(::CPPLSFit)),
[`sample_labels`](@ref sample_labels(::CPPLSFit)),
[`X_scores`](@ref X_scores(::CPPLSFit))

# Examples
```jldoctest
julia> using JLD2; file = CPPLS.dataset("synthetic_cppls_da_dataset.jld2");

julia> labels, X, classes, Y_aux = load(file, "sample_labels", "X", "classes", "Y_aux");

julia> spec = CPPLSSpec(n_components=2, gamma=0.01:0.01:1.00, analysis_mode=:discriminant);

julia> model = fit(spec, X, classes; sample_labels=labels)
CPPLSFit
  mode: discriminant
  samples: 100
  predictors: 14
  responses: 2
  components: 2

julia> CPPLS.gamma(model) == [0.84, 0.78]
true

julia> size(CPPLS.X_scores(model))
(100, 2)

julia> spec = CPPLSSpec(n_components=2, gamma=0.75, analysis_mode=:discriminant);

julia> model = fit(spec, X, classes; obs_weights=invfreqweights(classes), Y_aux=Y_aux)
CPPLSFit
  mode: discriminant
  samples: 100
  predictors: 14
  responses: 2
  components: 2
```
"""
function fit(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    kwargs...
)
    fit_cppls(model, X, Y_prim; kwargs...)
end

function fit(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...
) where {T,R,V,C,U}

    fit_cppls(model, X, sample_classes; kwargs...)
end

function fit(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractVector;
    kwargs...
)
    fit_cppls(model, X, sample_classes; kwargs...)
end

"""
    fit_cppls(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real},
        n_components::Integer=2;
        kwargs...
    )

Low-level CPPLS fitting routine used by `fit`. Prefer `fit` with a CPPLSSpec for the
public entry point and full parameter documentation.
"""
function fit_cppls(
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    n_components::Int=2;
    gamma::T1=0.5,
    obs_weights::T2=nothing,
    Y_aux::T3=nothing,
    center::Bool=true,
    X_tolerance::T4=1e-12,
    X_loading_weight_tolerance::T5=eps(Float64),
    gamma_rel_tol::T6=1e-6,
    gamma_abs_tol::T7=1e-12,
    t_squared_norm_tolerance::T8=1e-10,
    sample_labels::T9=String[],
    predictor_labels::T10=String[],
    response_labels::T11=String[],
    analysis_mode::Symbol = :regression,
    sample_classes::T12=nothing,
) where {
    T1<:Union{
        <:Real, 
        <:NTuple{2, <:Real},
        <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
    },
    T2<:Union{AbstractVector{<:Real} ,Nothing},
    T3<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T4<:Real,
    T5<:Real,
    T6<:Real,
    T7<:Real,
    T8<:Real,
    T9<:AbstractVector,
    T10<:AbstractVector,
    T11<:AbstractVector,
    T12<:Union{AbstractVector, Nothing}
}

    analysis_mode in (:discriminant, :regression) || throw(ArgumentError(
        "analysis_mode must be :discriminant or :regression, got $analysis_mode"))

    analysis_mode ≡ :discriminant || sample_classes ≡ nothing || throw(ArgumentError(
        "sample_classes can only be provided for discriminant analysis"))

    n_predictors = size(X, 2)

    (X, Y_prim, Y, obs_weights, X_bar, Y_bar, X_def, W_comp, P, C, zero_mask, B,
        n_samples_X, n_targets_Y) = cppls_prepare_data(X, Y_prim, n_components, Y_aux,
        obs_weights, center)

    sample_labels = default_sample_labels(validate_label_length(sample_labels, n_samples_X,
        "sample_labels"), n_samples_X)
    predictor_labels = validate_label_length(predictor_labels, n_predictors, 
        "predictor_labels")
    response_labels = validate_response_labels(response_labels, n_targets_Y)
    if analysis_mode ≡ :discriminant && isempty(response_labels)
        throw(ArgumentError(
            "response_labels must list class names for discriminant analysis"))
    end

    T = Matrix{Float64}(undef, n_samples_X, n_components)
    a = Matrix{Float64}(undef, size(Y, 2), n_components)
    b = Matrix{Float64}(undef, n_targets_Y, n_components)
    rho = Vector{Float64}(undef, n_components)
    gamma_vals = fill(0.5, n_components)
    t_norms = Vector{Float64}(undef, n_components)
    U = Matrix{Float64}(undef, n_samples_X, n_components)
    Y_hat = Array{Float64}(undef, n_samples_X, n_targets_Y, n_components)
    W0 = Array{Float64}(undef, n_predictors, size(Y, 2), n_components)
    Z = Array{Float64}(undef, n_samples_X, size(Y, 2), n_components)

    for i = 1:n_components
        wᵢ, rho[i], a[:, i], b[:, i], gamma_vals[i], W0ᵢ = compute_cppls_weights(
            X_def, Y, Y_prim, obs_weights, gamma, gamma_rel_tol, gamma_abs_tol)
        
        W0[:, :, i] = W0ᵢ
        Z[:, :, i] = X_def * W0ᵢ

        tᵢ, tᵢ_squared_norm, cᵢ = process_component!(i, X_def, wᵢ, Y_prim, W_comp, P, C, B,
            zero_mask, X_tolerance, X_loading_weight_tolerance, t_squared_norm_tolerance)

        T[:, i] = tᵢ
        t_norms[i] = tᵢ_squared_norm
        U[:, i] = Y_prim * cᵢ / (cᵢ' * cᵢ)

        if i > 1
            U[:, i] -= T * (T' * U[:, i] ./ t_norms)
        end
        Y_hat[:, :, i] = X * B[:, :, i]
    end

    Y_hat .+= reshape(repeat(Y_bar, n_samples_X), n_samples_X, length(Y_bar), 1)
    F = Y_prim .- Y_hat
    R = W_comp * pinv(P' * W_comp)
    X_var = vec(sum(P .* P, dims = 1)) .* t_norms
    X_var_total = sum(X .* X)

    CPPLSFit(B, T, P, W_comp, U, C, R, X_bar, Y_bar, Y_hat, F, X_var, X_var_total,
        gamma_vals, rho, zero_mask, a, b, W0, Z; sample_labels=sample_labels,
        predictor_labels=predictor_labels, response_labels=response_labels,
        analysis_mode=analysis_mode, sample_classes=sample_classes)
end

function fit_cppls(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Y_aux::T2=nothing,
    sample_labels::T3=String[],
    predictor_labels::T4=String[],
    response_labels::T5=String[],
    sample_classes::T6=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:AbstractVector,
    T6<:Union{AbstractVector, Nothing}  
}
    fit_cppls(X, Y_prim, n_components(model); cppls_model_fit_kwargs_with_mode(model)...,
        obs_weights=obs_weights, Y_aux=Y_aux, sample_labels=sample_labels,
        predictor_labels=predictor_labels, response_labels=response_labels,
        sample_classes=sample_classes)
end

"""
    fit_cppls(
        X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real},
        n_components::Int=2;
        kwargs...
    )

Convenience wrapper that reshapes a single response vector to a one-column matrix and
forwards into `fit_cppls`. Prefer `fit` for the public entry point and full docs.
"""
function fit_cppls(
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real},
    n_components::Int=2;
    gamma::T1=0.5,
    obs_weights::T2=nothing,
    Y_aux::T3=nothing,
    center::Bool=true,
    X_tolerance::T4=1e-12,
    X_loading_weight_tolerance::T5=eps(Float64),
    gamma_rel_tol::T6=1e-6,
    gamma_abs_tol::T7=1e-12,
    t_squared_norm_tolerance::T8=1e-10,
    sample_labels::T9=String[],
    predictor_labels::T10=String[],
    response_labels::T11=String[],
) where {
    T1<:Union{
        <:Real,
        <:NTuple{2, <:Real},
        <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
    },
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T4<:Real,
    T5<:Real,
    T6<:Real,
    T7<:Real,
    T8<:Real,
    T9<:AbstractVector,
    T10<:AbstractVector,
    T11<:AbstractVector  
}

    Y_matrix = reshape(Y_prim, :, 1)

    fit_cppls(X, Y_matrix, n_components; gamma=gamma, obs_weights=obs_weights, Y_aux=Y_aux,
        center=center, X_tolerance=X_tolerance, 
        X_loading_weight_tolerance=X_loading_weight_tolerance, gamma_rel_tol=gamma_rel_tol,
        gamma_abs_tol=gamma_abs_tol, t_squared_norm_tolerance=t_squared_norm_tolerance,
        sample_labels=sample_labels, predictor_labels=predictor_labels, 
        response_labels=response_labels, analysis_mode=:regression
    )
end

function fit_cppls(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real};
    obs_weights::T1=nothing,
    Y_aux::T2=nothing,
    sample_labels::T3=String[],
    predictor_labels::T4=String[],
    response_labels::T5=String[],
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:AbstractVector
}

    fit_cppls(X, Y_prim, n_components(model); cppls_model_fit_kwargs_with_mode(model)...,
        obs_weights=obs_weights, Y_aux=Y_aux, sample_labels=sample_labels,
        predictor_labels=predictor_labels, response_labels=response_labels)
end

"""
    fit_cppls(X, sample_classes::AbstractCategoricalArray, n_components::Int=2; kwargs...)
    fit_cppls(X, sample_classes::AbstractVector, n_component::Int=2; kwargs...)

Label-based convenience wrappers that convert class labels to a one-hot response matrix
and forward into `fit_cppls`. Prefer `fit` for the public entry point and full docs.
"""
function fit_cppls(
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractCategoricalArray{T,1,R,V,C,U},
    n_components::Int=2;
    kwargs...
) where {T,R,V,C,U}
    fit_cppls_from_sample_classes(X, sample_classes, n_components; kwargs...)
end

function fit_cppls(
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractVector,
    n_components::Int=2;
    kwargs...
)
    fit_cppls_from_sample_classes(X, sample_classes, n_components; kwargs...)
end

function fit_cppls(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...
) where {T,R,V,C,U}
    model.analysis_mode ≡ :discriminant || throw(ArgumentError(
        "CPPLSSpec must use analysis_mode=:discriminant when fitting from sample_classes."))
    
    fit_cppls_from_sample_classes(X, sample_classes, n_components(model);
        cppls_model_fit_kwargs(model)..., kwargs...)
end

function fit_cppls(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractVector;
    kwargs...
)
    model.analysis_mode ≡ :discriminant || throw(ArgumentError(
        "CPPLSSpec must use analysis_mode=:discriminant when fitting from sample_classes."))
    
    fit_cppls_from_sample_classes(X, sample_classes, n_components(model);
        cppls_model_fit_kwargs(model)..., kwargs...)
end

"""
    fit_cppls_from_sample_classes(
        X::AbstractMatrix{<:Real},
        sample_classes,
        n_components::Int=2;
        kwargs...
    )

Internal helper that converts class labels to a one-hot response matrix before fitting
and is primarily used by the label-based `fit_cppls` wrappers. Prefer `fit` for user
documentation and public entry points.
"""
function fit_cppls_from_sample_classes(
    X::AbstractMatrix{<:Real},
    sample_classes,
    n_components::Int=2;
    gamma::T1=0.5,
    obs_weights::T2=nothing,
    Y_aux::T3=nothing,
    center::Bool=true,
    X_tolerance::T4=1e-12,
    X_loading_weight_tolerance::T5=eps(Float64),
    gamma_rel_tol::T6=1e-6,
    gamma_abs_tol::T7=1e-12,
    t_squared_norm_tolerance::T8=1e-10,
    sample_labels::T9=String[],
    predictor_labels::T10=String[],
    response_labels::T11=String[]
) where {
    T1<:Union{
        <:Real, 
        <:NTuple{2, <:Real}, 
        <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
    },
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T4<:Real,
    T5<:Real,
    T6<:Real,
    T7<:Real,
    T8<:Real,
    T9<:AbstractVector,
    T10<:AbstractVector,
    T11<:AbstractVector
}
    isempty(response_labels) || throw(ArgumentError("`response_labels` cannot be provided" *
        " when passing sample classes; response labels are inferred automatically."))

    Y_prim, classes = labels_to_one_hot(sample_classes)

    fit_cppls(X, Y_prim, n_components; gamma=gamma, obs_weights=obs_weights, Y_aux=Y_aux,
        center=center, X_tolerance=X_tolerance, 
        X_loading_weight_tolerance=X_loading_weight_tolerance, gamma_rel_tol=gamma_rel_tol,
        gamma_abs_tol=gamma_abs_tol, t_squared_norm_tolerance=t_squared_norm_tolerance,
        sample_labels=sample_labels, predictor_labels=predictor_labels, 
        response_labels=classes, analysis_mode=:discriminant, 
        sample_classes=copy(sample_classes)
    )
end

############################################################################################
# Helper
############################################################################################
"""
    cppls_model_fit_kwargs(model::CPPLSSpec)

Collect the CPPLSSpec fields that correspond to `fit_cppls` keyword arguments and return
them as a NamedTuple for forwarding to fit helpers.
"""
function cppls_model_fit_kwargs(model::CPPLSSpec)
    (
        gamma = model.gamma,
        center = model.center,
        X_tolerance = model.X_tolerance,
        X_loading_weight_tolerance = model.X_loading_weight_tolerance,
        t_squared_norm_tolerance = model.t_squared_norm_tolerance,
        gamma_rel_tol = model.gamma_rel_tol,
        gamma_abs_tol = model.gamma_abs_tol,
    )
end

"""
    cppls_model_fit_kwargs_with_mode(model::CPPLSSpec)

Return the same NamedTuple as `cppls_model_fit_kwargs` but include `analysis_mode` to
preserve regression versus discriminant intent in wrapper calls.
"""
function cppls_model_fit_kwargs_with_mode(model::CPPLSSpec)
    merge(cppls_model_fit_kwargs(model), (analysis_mode = analysis_mode(model),))
end

"""
    validate_label_length(
        labels::AbstractVector,
        expected::Integer,
        name::AbstractString,
    )

Return `labels` after checking that it is empty or matches the expected length. An
`ArgumentError` is thrown when a non-empty label vector has the wrong length.
"""
function validate_label_length(
    labels::AbstractVector,
    expected::Integer,
    name::AbstractString,
)
    isempty(labels) || length(labels) == expected || throw(ArgumentError(
        "`$name` must have length $expected, got $(length(labels))"))
    labels
end

"""
    default_sample_labels(labels::AbstractVector, n_samples::Integer)

Return `labels` when provided, otherwise generate default row-index labels `"1"` through
`string(n_samples)`.
"""
function default_sample_labels(labels::AbstractVector, n_samples::Integer)
    isempty(labels) || return labels
    string.(1:n_samples)
end

"""
    validate_response_labels(labels::AbstractVector, n_targets::Integer)

Return `labels` after verifying that it is empty or has length `n_targets`. This is used
to ensure the provided response names align with the response matrix.
"""
function validate_response_labels(labels::AbstractVector, n_targets::Integer)
    isempty(labels) || length(labels) == n_targets || throw(ArgumentError(
        "`response_labels` must have length $n_targets, got $(length(labels))"))
    labels
end

"""
    process_component!(
        i::Int,
        X_def::AbstractMatrix{<:Real},
        wᵢ::AbstractVector{<:Real},
        Y_prim::AbstractMatrix{<:Real},
        W_comp::AbstractMatrix{<:Real},
        P::AbstractMatrix{<:Real},
        C::AbstractMatrix{<:Real},
        B::Array{<:Real, 3},
        zero_mask::AbstractMatrix{Bool},
        X_tolerance::Real,
        X_loading_weight_tolerance::Real,
        tᵢ_squared_norm_tolerance::Real
    )

Compute the i-th component from the current deflated predictors and update the CPPLS work 
arrays. The weight vector is normalized and thresholded, the score t is formed as
X_def * wᵢ, and the X and Y loadings are computed by regressing the predictors and
responses on t. The predictor block is then deflated along the component, the zero mask
and regression coefficients are updated, and the function returns the score, its squared
norm, and the Y loading vector.
"""
function process_component!(
    i::Int,
    X_def::AbstractMatrix{<:Real},
    wᵢ::AbstractVector{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    W_comp::AbstractMatrix{<:Real},
    P::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    B::Array{<:Real, 3},
    zero_mask::AbstractMatrix{Bool},
    X_tolerance::Real,
    X_loading_weight_tolerance::Real,
    tᵢ_squared_norm_tolerance::Real
)

    wᵢ .= wᵢ ./ norm(wᵢ) .* (abs.(wᵢ) .≥ X_loading_weight_tolerance)

    tᵢ = X_def * wᵢ
    tᵢ_squared_norm = tᵢ' * tᵢ

    if isapprox(tᵢ_squared_norm, 0.0)
        tᵢ_squared_norm += tᵢ_squared_norm_tolerance
    end

    pᵢ = (X_def' * tᵢ) / tᵢ_squared_norm
    cᵢ = (Y_prim' * tᵢ) / tᵢ_squared_norm

    X_def .-= tᵢ * pᵢ'

    zero_mask[i, :] .= vec(sum(abs.(X_def), dims=1) .< X_tolerance)
    X_def[:, zero_mask[i, :]] .= 0

    W_comp[:, i] .= wᵢ
    P[:, i] .= pᵢ
    C[:, i] .= cᵢ
    B[:, :, i] .= W_comp[:, 1:i] * pinv(P[:, 1:i]' * W_comp[:, 1:i]) * C[:, 1:i]'

    tᵢ, tᵢ_squared_norm, cᵢ
end
