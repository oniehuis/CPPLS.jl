function validate_label_length(
    labels::AbstractVector,
    expected::Integer,
    name::AbstractString,
)
    isempty(labels) ||
        length(labels) == expected ||
        throw(ArgumentError("`$name` must have length $expected, got $(length(labels))"))
    return labels
end

function validate_response_labels(labels::AbstractVector, n_targets::Integer)
    isempty(labels) ||
        length(labels) == n_targets ||
        throw(
            ArgumentError(
                "`response_labels` must have length $n_targets, got $(length(labels))",
            ),
        )
    return labels
end

"""
    fit_cppls(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real},
        n_components::Integer;
        gamma::Union{<:Real, <:NTuple{2, <:Real}, <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}}=0.5,
        observation_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat, Nothing}=nothing,
        center::Bool=true,
        X_tolerance::Real=1e-12,
        X_loading_weight_tolerance::Real=eps(Float64), 
        gamma_rel_tol::Real=1e-6,
        gamma_abs_tol::Real=1e-12,
        t_squared_norm_tolerance::Real=1e-10,
        sample_labels::AbstractVector=String[],
        predictor_labels::AbstractVector=String[],
        response_labels::AbstractVector=String[])

Fit a Canonical Powered Partial Least Squares (CPPLS) model.

# Arguments
- `X`: A matrix of predictor variables (observations × features). `NA`s and `Inf`s are not 
  allowed.
- `Y`: A matrix of response variables (observations × targets). `NA`s and `Inf`s are not 
  allowed.

# Optional Positional Argument
- `n_components`: The number of components to extract in the CPPLS model. Defaults to 2.

# Optional Keyword Arguments
- `gamma`: Either (i) a fixed power parameter (`γ`), (ii) a `(lo, hi)` tuple describing the
  bounds for per-component optimization, or (iii) a vector mixing both forms. Defaults to
  `0.5`, i.e. no optimization.
- `observation_weights`: A vector of individual weights for the observations (e.g., 
  experimental data or samples). Defaults to `nothing`.
- `Y_auxiliary`: A matrix (or vector) of auxiliary response variables containing additional information 
  about the observations. Defaults to `nothing`.
- `center`: Whether to mean-center the `X` and `Y` matrices. Defaults to `true`.
- `X_tolerance`: Tolerance for small norms in `X`. Columns of `X` with norms below this 
  threshold are set to zero during deflation. Defaults to `1e-12`.
- `X_loading_weight_tolerance`: Tolerance for small weights. Elements of the weight vector 
  below this threshold are set to zero. Defaults to `eps(Float64)`.
- `gamma_rel_tol`: Relative tolerance for the γ optimizer. Defaults to `1e-6`.
- `gamma_abs_tol`: Absolute tolerance for the γ optimizer. Defaults to `1e-12`.
- `t_squared_norm_tolerance`: Small positive value added to near-zero score norms to keep
  downstream divisions stable. Defaults to `1e-10`.
- `sample_labels`: Optional labels describing each observation. Defaults to `String[]`.
- `predictor_labels`: Optional labels for the predictor columns (in order). Defaults to 
  `String[]`.
- `response_labels`: Optional labels for the response variables / classes (in order).
  Defaults to `String[]` for regressions. When passing categorical responses (see below),
  class labels are inferred automatically.
- `analysis_mode`: Internal flag distinguishing regression from discriminant analysis.
  Advanced callers can override this, but public wrappers set it automatically.
- `da_categories`: Original categorical responses for discriminant analysis. This is set
  by the label-based wrapper and must remain `nothing` for regression problems.

# Returns
A `CPPLS` object containing the following fields:
- `regression_coefficients`: A 3D array of regression coefficients for 1, ..., 
  `n_components`.
- `X_scores`: A matrix of scores (latent variables) for the predictor matrix `X`.
- `X_loadings`: A matrix of loadings for the predictor matrix `X`.
- `X_loading_weights`: A matrix of loading weights for the predictor matrix `X`.
- `Y_scores`: A matrix of scores (latent variables) for the response matrix `Y`.
- `Y_loadings`: A matrix of loadings for the response matrix `Y`.
- `projection`: The projection matrix used to convert `X` to scores.
- `X_means`: A vector of means of the `X` variables (used for centering).
- `Y_means`: A vector of means of the `Y` variables (used for centering).
- `fitted_values`: An array of fitted values for the response matrix `Y`.
- `residuals`: An array of residuals for the response matrix `Y`.
- `X_variance`: A vector containing the amount of variance in `X` explained by each 
   component.
- `X_total_variance`: The total variance in `X`.
- `gammas`: The power parameter (`γ`) values obtained during power optimization.
- `canonical_correlations`: Canonical correlation values for each component.
- `small_norm_indices`: Indices of explanatory variables with norms close to or equal to 
   zero.
- `canonical_coefficients`: A matrix containing the canonical coefficients (`a`) from 
  canonical correlation analysis (`cor(Za, Yb)`).
- `canonical_coefficients_y`: A matrix containing the canonical coefficients (`b`) for the
  responses from canonical correlation analysis.
- `W0_weights`: Initial CPPLS weight matrices per component.
- `Z`: Supervised predictor projections per component (`X_deflated * W0_weights`).
- `sample_labels`: The provided sample labels (or an empty vector if none were supplied).
- `predictor_labels`: The provided predictor labels (or an empty vector).
- `response_labels`: The provided response labels (or an empty vector).
- `analysis_mode`: Tracks whether the model was fit for regression or discriminant analysis.
- `da_categories`: The original categorical responses for discriminant analysis (otherwise
  `nothing`).

# Notes
- The CPPLS model is an extension of Partial Least Squares (PLS) that incorporates 
  canonical correlation analysis (CCA) and power parameter optimization to maximize the 
  correlation between linear combinations of `X` and `Y`.
- The power parameter (`γ`) controls the balance between variance maximization and 
  correlation maximization. It is optimized within the specified bounds (`gamma_bounds`).
- If `Y_auxiliary` is provided, it is concatenated with `Y` to form a combined response 
  matrix (`Y_combined`), which is used during the fitting process.
- Passing a categorical response vector instead of a numeric matrix automatically triggers
  the discriminant-analysis variant of `fit_cppls` and infers class labels.

# Example
```
julia> X = Float64[1 0 2
                   0 1 2
                   1 1 1
                   2 3 0
                   3 2 1];

julia> labels = ["red", "blue", "red", "blue", "red"];

julia> Y, classes = labels_to_one_hot(labels);

julia> model = fit_cppls(X, Y, 2; gamma=(0.7, 1.0));

julia> model.X_means ≈ Matrix([1.4 1.4 1.2])

julia> model.gammas ≈ [0.700185836799654, 0.9366214237592033]
true
```
"""
function fit_cppls(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    n_components::Integer = 2;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    t_squared_norm_tolerance::Real = 1e-10,
    sample_labels::AbstractVector = String[],
    predictor_labels::AbstractVector = String[],
    response_labels::AbstractVector = String[],
    analysis_mode::Symbol = :regression,
    da_categories = nothing,
) where {T1<:Real,T2<:Real}

    analysis_mode in (:regression, :discriminant) || throw(
        ArgumentError(
            "analysis_mode must be :regression or :discriminant, got $analysis_mode",
        ),
    )
    analysis_mode === :discriminant ||
        da_categories === nothing ||
        throw(ArgumentError("da_categories can only be provided for discriminant analysis"))

    n_predictors = size(X_predictors, 2)

    (
        X_predictors,
        Y_responses,
        Y_combined,
        observation_weights,
        X̄_mean,
        Ȳ_mean,
        X_deflated,
        X_loading_weights,
        X_loadings,
        Y_loadings,
        small_norm_flags,
        regression_coefficients,
        n_samples_X,
        n_targets_Y,
    ) = cppls_prepare_data(
        X_predictors,
        Y_responses,
        n_components,
        Y_auxiliary,
        observation_weights,
        center,
    )

    sample_labels = validate_label_length(sample_labels, n_samples_X, "sample_labels")
    predictor_labels =
        validate_label_length(predictor_labels, n_predictors, "predictor_labels")
    response_labels = validate_response_labels(response_labels, n_targets_Y)
    if analysis_mode === :discriminant && isempty(response_labels)
        throw(
            ArgumentError(
                "response_labels must list class names for discriminant analysis",
            ),
        )
    end

    X_scores = Matrix{Float64}(undef, n_samples_X, n_components)
    canonical_coefficients = Matrix{Float64}(undef, size(Y_combined, 2), n_components)
    canonical_coefficients_y = Matrix{Float64}(undef, n_targets_Y, n_components)
    max_canonical_correlations = Vector{Float64}(undef, n_components)
    gamma_values = fill(0.5, n_components)
    X_score_norms = Vector{Float64}(undef, n_components)
    Y_scores = Matrix{Float64}(undef, n_samples_X, n_components)
    fitted_values = Array{Float64}(undef, n_samples_X, n_targets_Y, n_components)
    W0_weights = Array{Float64}(undef, n_predictors, size(Y_combined, 2), n_components)
    Z = Array{Float64}(undef, n_samples_X, size(Y_combined, 2), n_components)

    for i = 1:n_components
        (
            X_loading_weightsᵢ,
            max_canonical_correlations[i],
            canonical_coefficients[:, i],
            canonical_coefficients_y[:, i],
            gamma_values[i],
            W0_weightsᵢ,
        ) = (compute_cppls_weights(
            X_deflated,
            Y_combined,
            Y_responses,
            observation_weights,
            gamma,
            gamma_rel_tol,
            gamma_abs_tol,
        ))
        W0_weights[:, :, i] = W0_weightsᵢ
        Z[:, :, i] = X_deflated * W0_weightsᵢ

        X_scoresᵢ, tᵢ_squared_norm, Y_loadingsᵢ = process_component!(
            i,
            X_deflated,
            X_loading_weightsᵢ,
            Y_responses,
            X_loading_weights,
            X_loadings,
            Y_loadings,
            regression_coefficients,
            small_norm_flags,
            X_tolerance,
            X_loading_weight_tolerance,
            t_squared_norm_tolerance,
        )

        X_scores[:, i] = X_scoresᵢ
        X_score_norms[i] = tᵢ_squared_norm
        Y_scores[:, i] = Y_responses * Y_loadingsᵢ / (Y_loadingsᵢ' * Y_loadingsᵢ)

        if i > 1
            Y_scores[:, i] -= X_scores * (X_scores' * Y_scores[:, i] ./ X_score_norms)
        end
        fitted_values[:, :, i] = X_predictors * regression_coefficients[:, :, i]
    end

    fitted_values .+= reshape(repeat(Ȳ_mean, n_samples_X), n_samples_X, length(Ȳ_mean), 1)
    Y_residuals = Y_responses .- fitted_values
    projection = X_loading_weights * pinv(X_loadings' * X_loading_weights)
    X_variance_explained = vec(sum(X_loadings .* X_loadings, dims = 1)) .* X_score_norms
    X_total_variance = sum(X_predictors .* X_predictors)

    CPPLS(
        regression_coefficients,
        X_scores,
        X_loadings,
        X_loading_weights,
        Y_scores,
        Y_loadings,
        projection,
        X̄_mean,
        Ȳ_mean,
        fitted_values,
        Y_residuals,
        X_variance_explained,
        X_total_variance,
        gamma_values,
        max_canonical_correlations,
        small_norm_flags,
        canonical_coefficients,
        canonical_coefficients_y,
        W0_weights,
        Z;
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = response_labels,
        analysis_mode = analysis_mode,
        da_categories = da_categories,
    )
end

"""
    fit_cppls(X, labels::AbstractCategoricalArray, n_components=2; kwargs...)
    fit_cppls(X, labels::AbstractVector, n_components=2; kwargs...)

Discriminant-analysis variants of [`fit_cppls`](@ref). The first method dispatches
specifically on `CategoricalVector`/`CategoricalArray` inputs so users can opt into DA
behaviour through the type signature alone. The second method accepts any other label
container (e.g. plain `Vector{String}` or `Vector{Symbol}`) but follows the exact same
code path. Both convert the labels to a one-hot response matrix internally and store
the inferred class names inside the returned `CPPLS` model.

# Example
```
julia> using CategoricalArrays

julia> X = Float64[1 0; 0 1; 1 1; 2 1];

julia> cat_labels = categorical(["red", "blue", "red", "blue"]);

julia> cppls_cat = fit_cppls(X, cat_labels, 2; gamma=0.5);

julia> cppls_cat.analysis_mode
:discriminant

julia> plain_labels = ["red", "blue", "red", "blue"];

julia> cppls_plain = fit_cppls(X, plain_labels, 2; gamma=0.5);

julia> cppls_plain.response_labels == cppls_cat.response_labels
true
```
"""
function fit_cppls(
    X_predictors::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U},
    n_components::Integer = 2;
    kwargs...,
) where {T,R,V,C,U}
    fit_cppls_from_labels(X_predictors, labels, n_components; kwargs...)
end

function fit_cppls(
    X_predictors::AbstractMatrix{<:Real},
    labels::AbstractVector,
    n_components::Integer = 2;
    kwargs...,
)
    fit_cppls_from_labels(X_predictors, labels, n_components; kwargs...)
end

function fit_cppls_from_labels(
    X_predictors::AbstractMatrix{<:Real},
    labels,
    n_components::Integer;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    t_squared_norm_tolerance::Real = 1e-10,
    sample_labels::AbstractVector = String[],
    predictor_labels::AbstractVector = String[],
    response_labels::AbstractVector = String[],
) where {T1<:Real,T2<:Real}
    isempty(response_labels) || throw(
        ArgumentError(
            "`response_labels` cannot be provided when passing categorical responses; class names are inferred automatically.",
        ),
    )

    Y_responses, classes = labels_to_one_hot(labels)

    return fit_cppls(
        X_predictors,
        Y_responses,
        n_components;
        gamma = gamma,
        observation_weights = observation_weights,
        Y_auxiliary = Y_auxiliary,
        center = center,
        X_tolerance = X_tolerance,
        X_loading_weight_tolerance = X_loading_weight_tolerance,
        gamma_rel_tol = gamma_rel_tol,
        gamma_abs_tol = gamma_abs_tol,
        t_squared_norm_tolerance = t_squared_norm_tolerance,
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = classes,
        analysis_mode = :discriminant,
        da_categories = copy(labels),
    )
end

"""
    fit_cppls(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, n_components=2; kwargs...)

Regression-friendly convenience wrapper around [`fit_cppls`](@ref) that accepts a
single numeric response vector instead of a full response matrix. The vector is reshaped
to `(n_samples, 1)` internally and all keyword arguments are forwarded to the standard
matrix-based implementation.

# Example
```
julia> X = Float64[1 2; 3 4; 5 6];

julia> y = [0.1, 0.5, 0.9];

julia> model = fit_cppls(X, y, 2; gamma=0.5);

julia> model.analysis_mode
:regression
```
"""
function fit_cppls(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractVector{<:Real},
    n_components::Integer = 2;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    t_squared_norm_tolerance::Real = 1e-10,
    sample_labels::AbstractVector = String[],
    predictor_labels::AbstractVector = String[],
    response_labels::AbstractVector = String[],
) where {T1<:Real,T2<:Real}

    Y_matrix = reshape(Y_responses, :, 1)

    return fit_cppls(
        X_predictors,
        Y_matrix,
        n_components;
        gamma = gamma,
        observation_weights = observation_weights,
        Y_auxiliary = Y_auxiliary,
        center = center,
        X_tolerance = X_tolerance,
        X_loading_weight_tolerance = X_loading_weight_tolerance,
        gamma_rel_tol = gamma_rel_tol,
        gamma_abs_tol = gamma_abs_tol,
        t_squared_norm_tolerance = t_squared_norm_tolerance,
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = response_labels,
        analysis_mode = :regression,
    )
end

"""
    fit_cppls_light(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real},
        n_components::Integer;
        gamma::Union{<:Real, <:NTuple{2, <:Real}, <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}}=0.5,
        observation_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat, Nothing}=nothing,
        center::Bool=true,
        X_tolerance::Real=1e-12,
        X_loading_weight_tolerance::Real=eps(Float64),
        gamma_rel_tol::Real=1e-6,
        gamma_abs_tol::Real=1e-12,
        t_squared_norm_tolerance::Real=1e-10,
        analysis_mode::Symbol=:regression)

Fit a CPPLS model but retain only the parts needed for prediction (`CPPLSLight`).

Arguments mirror `fit_cppls`, including support for scalar γ, `(lo, hi)` bounds, or
vectors that mix scalars and tuples as candidate sets. The returned `CPPLSLight` stores
only the stacked regression coefficients plus the `X`/`Y` centering means. 
`analysis_mode` is an internal keyword that tags the resulting object as either a
regression or discriminant model; most users rely on the wrappers below instead of
setting it manually.

# Notes
- Use this when you only need predictions, not the intermediate diagnostics.
- The same preprocessing, weighting, and tolerance settings apply as in `fit_cppls`.

# Example
```
julia> X = Float64[1 0 2
                   0 1 2
                   1 1 1
                   2 3 0
                   3 2 1];

julia> labels = ["red", "blue", "red", "blue", "red"];

julia> Y, classes = labels_to_one_hot(labels);

julia> model = fit_cppls_light(X, Y, 2; gamma=(0.7, 1.0));

julia> model.X_means ≈ Matrix([1.4 1.4 1.2])
true
```
"""
function fit_cppls_light(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    n_components::Integer = 2;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    t_squared_norm_tolerance::Real = 1e-10,
    analysis_mode::Symbol = :regression,
) where {T1<:Real,T2<:Real}

    (
        X_predictors,
        Y_responses,
        Y_combined,
        observation_weights,
        X̄_mean,
        Ȳ_mean,
        X_deflated,
        X_loading_weights,
        X_loadings,
        Y_loadings,
        small_norm_flags,
        regression_coefficients,
        _,
        _,
    ) = cppls_prepare_data(
        X_predictors,
        Y_responses,
        n_components,
        Y_auxiliary,
        observation_weights,
        center,
    )

    for i = 1:n_components
        X_loading_weightsᵢ, _, _, _, _, _ = compute_cppls_weights(
            X_deflated,
            Y_combined,
            Y_responses,
            observation_weights,
            gamma,
            gamma_rel_tol,
            gamma_abs_tol,
        )

        process_component!(
            i,
            X_deflated,
            X_loading_weightsᵢ,
            Y_responses,
            X_loading_weights,
            X_loadings,
            Y_loadings,
            regression_coefficients,
            small_norm_flags,
            X_tolerance,
            X_loading_weight_tolerance,
            t_squared_norm_tolerance,
        )

    end

    analysis_mode in (:regression, :discriminant) || throw(
        ArgumentError(
            "analysis_mode must be :regression or :discriminant, got $analysis_mode",
        ),
    )

    CPPLSLight(regression_coefficients, X̄_mean, Ȳ_mean, analysis_mode)
end

"""
    fit_cppls_light(X, labels::AbstractCategoricalArray, n_components=2; kwargs...)
    fit_cppls_light(X, labels::AbstractVector, n_components=2; kwargs...)

Discriminant-analysis convenience wrappers for [`fit_cppls_light`](@ref). The first
signature dispatches explicitly on categorical arrays so callers can rely on the method
table to distinguish regression from DA. The second accepts any other label container
(e.g. vectors of strings, symbols, or enums) and forwards into the same code path.
Regardless of signature, labels are converted to a one-hot matrix for fitting, class
names are inferred once, and the returned `CPPLSLight` only retains the components needed
for prediction.

# Example
```
julia> using CategoricalArrays

julia> X = Float64[1 0; 0 1; 1 1; 2 1];

julia> labels = categorical(["classA", "classB", "classA", "classB"]);

julia> light_cat = fit_cppls_light(X, labels, 2; gamma=0.5);

julia> light_cat.analysis_mode
:discriminant

julia> light_plain = fit_cppls_light(X, ["classA", "classB", "classA", "classB"], 2; gamma=0.5);

julia> light_plain.regression_coefficients ≈ light_cat.regression_coefficients
true
```
"""
function fit_cppls_light(
    X_predictors::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U},
    n_components::Integer = 2;
    kwargs...,
) where {T,R,V,C,U}
    fit_cppls_light_from_labels(X_predictors, labels, n_components; kwargs...)
end

function fit_cppls_light(
    X_predictors::AbstractMatrix{<:Real},
    labels::AbstractVector,
    n_components::Integer = 2;
    kwargs...,
)
    fit_cppls_light_from_labels(X_predictors, labels, n_components; kwargs...)
end

function fit_cppls_light_from_labels(
    X_predictors::AbstractMatrix{<:Real},
    labels,
    n_components::Integer;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    t_squared_norm_tolerance::Real = 1e-10,
) where {T1<:Real,T2<:Real}
    Y_responses, _ = labels_to_one_hot(labels)

    fit_cppls_light(
        X_predictors,
        Y_responses,
        n_components;
        gamma = gamma,
        observation_weights = observation_weights,
        Y_auxiliary = Y_auxiliary,
        center = center,
        X_tolerance = X_tolerance,
        X_loading_weight_tolerance = X_loading_weight_tolerance,
        gamma_rel_tol = gamma_rel_tol,
        gamma_abs_tol = gamma_abs_tol,
        t_squared_norm_tolerance = t_squared_norm_tolerance,
        analysis_mode = :discriminant,
    )
end

"""
    fit_cppls_light(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, n_components=2; kwargs...)

Regression convenience wrapper for [`fit_cppls_light`](@ref) that accepts a single
numeric response vector. Internally reshapes `y` to `(n_samples, 1)` and forwards all
keyword arguments to the matrix-based implementation.

# Example
```
julia> X = Float64[1 2; 3 4; 5 6];

julia> y = [0.1, 0.5, 0.9];

julia> light = fit_cppls_light(X, y, 2; gamma=0.5);

julia> light.analysis_mode
:regression
```
"""
function fit_cppls_light(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractVector{<:Real},
    n_components::Integer = 2;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    t_squared_norm_tolerance::Real = 1e-10,
) where {T1<:Real,T2<:Real}

    Y_matrix = reshape(Y_responses, :, 1)

    fit_cppls_light(
        X_predictors,
        Y_matrix,
        n_components;
        gamma = gamma,
        observation_weights = observation_weights,
        Y_auxiliary = Y_auxiliary,
        center = center,
        X_tolerance = X_tolerance,
        X_loading_weight_tolerance = X_loading_weight_tolerance,
        gamma_rel_tol = gamma_rel_tol,
        gamma_abs_tol = gamma_abs_tol,
        t_squared_norm_tolerance = t_squared_norm_tolerance,
        analysis_mode = :regression,
    )
end

function process_component!(
    i::Integer,
    X_deflated::AbstractMatrix{<:Real},
    X_loading_weightsᵢ::AbstractVector{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    X_loading_weights::AbstractMatrix{<:Real},
    X_loadings::AbstractMatrix{<:Real},
    Y_loadings::AbstractMatrix{<:Real},
    regression_coefficients::Array{<:Real,3},
    small_norm_flags::AbstractMatrix{Bool},
    X_tolerance::Real,
    X_loading_weight_tolerance::Real,
    tᵢ_squared_norm_tolerance::Real,
)

    X_loading_weightsᵢ .= (
        X_loading_weightsᵢ ./ norm(X_loading_weightsᵢ) .*
        (abs.(X_loading_weightsᵢ) .>= X_loading_weight_tolerance)
    )

    X_scoresᵢ = X_deflated * X_loading_weightsᵢ
    tᵢ_squared_norm = X_scoresᵢ' * X_scoresᵢ

    if isapprox(tᵢ_squared_norm, 0.0)
        tᵢ_squared_norm += tᵢ_squared_norm_tolerance
    end
    X_loadingsᵢ = (X_deflated' * X_scoresᵢ) / tᵢ_squared_norm
    Y_loadingsᵢ = (Y_responses' * X_scoresᵢ) / tᵢ_squared_norm

    X_deflated .-= X_scoresᵢ * X_loadingsᵢ'

    small_norm_flags[i, :] .= vec(sum(abs.(X_deflated), dims = 1) .< X_tolerance)
    X_deflated[:, small_norm_flags[i, :]] .= 0

    X_loading_weights[:, i] .= X_loading_weightsᵢ
    X_loadings[:, i] .= X_loadingsᵢ
    Y_loadings[:, i] .= Y_loadingsᵢ
    regression_coefficients[:, :, i] .= (
        X_loading_weights[:, 1:i] *
        pinv(X_loadings[:, 1:i]' * X_loading_weights[:, 1:i]) *
        Y_loadings[:, 1:i]'
    )

    X_scoresᵢ, tᵢ_squared_norm, Y_loadingsᵢ
end
