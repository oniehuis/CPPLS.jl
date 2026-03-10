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

function resolve_Y_aux(
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing},
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing},
)
    if Y_aux !== nothing && Y_auxiliary !== nothing
        throw(ArgumentError("Provide either Y_aux or Y_auxiliary, not both."))
    end
    return Y_aux === nothing ? Y_auxiliary : Y_aux
end

function _cppls_model_fit_kwargs(model::CPPLSSpec)
    return (
        gamma = model.gamma,
        center = model.center,
        X_tolerance = model.X_tolerance,
        X_loading_weight_tolerance = model.X_loading_weight_tolerance,
        t_squared_norm_tolerance = model.t_squared_norm_tolerance,
        gamma_rel_tol = model.gamma_rel_tol,
        gamma_abs_tol = model.gamma_abs_tol,
    )
end

function _cppls_model_fit_kwargs_with_mode(model::CPPLSSpec)
    return merge(_cppls_model_fit_kwargs(model), (analysis_mode = model.analysis_mode,))
end

"""
    fit_cppls(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real},
        n_components::Integer;
        gamma::Union{<:Real, <:NTuple{2, <:Real}, <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}}=0.5,
        observation_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        Y_aux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}=nothing,
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
- `Y_aux`: A matrix (or vector) of auxiliary response variables containing additional
  information about the observations. Defaults to `nothing`. The legacy keyword
  `Y_auxiliary` is accepted as an alias.
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
- `sample_classes`: Original categorical responses for discriminant analysis. This is set
  by the label-based wrapper and must remain `nothing` for regression problems.

# Returns
A `CPPLSFit` object containing the following fields:
- `B`: A 3D array of regression coefficients for 1, ..., 
  `n_components`.
- `T`: A matrix of scores (latent variables) for the predictor matrix `X`.
- `P`: A matrix of loadings for the predictor matrix `X`.
- `W_comp`: A matrix of loading weights for the predictor matrix `X`.
- `U`: A matrix of scores (latent variables) for the response matrix `Y`.
- `C`: A matrix of loadings for the response matrix `Y`.
- `R`: The R matrix used to convert `X` to scores.
- `X_bar`: A vector of means of the `X` variables (used for centering).
- `Y_bar`: A vector of means of the `Y` variables (used for centering).
- `Y_hat`: An array of fitted values for the response matrix `Y`.
- `F`: An array of F for the response matrix `Y`.
- `X_var`: A vector containing the amount of variance in `X` explained by each 
   component.
- `X_var_total`: The total variance in `X`.
- `gamma`: The power parameter (`γ`) values obtained during power optimization.
- `rho`: Canonical correlation values for each component.
- `zero_mask`: Indices of explanatory variables with norms close to or equal to 
   zero.
- `a`: A matrix containing the canonical coefficients (`a`) from 
  canonical correlation analysis (`cor(Za, Yb)`).
- `b`: A matrix containing the canonical coefficients (`b`) for the
  responses from canonical correlation analysis.
- `W0`: Initial CPPLS weight matrices per component.
- `Z`: Supervised predictor projections per component (`X_def * W0`).
- `sample_labels`: The provided sample labels (or an empty vector if none were supplied).
- `predictor_labels`: The provided predictor labels (or an empty vector).
- `response_labels`: The provided response labels (or an empty vector).
- `analysis_mode`: Tracks whether the model was fit for regression or discriminant analysis.
- `sample_classes`: The original categorical responses for discriminant analysis (otherwise
  `nothing`).

# Notes
- The CPPLS model is an extension of Partial Least Squares (PLS) that incorporates 
  canonical correlation analysis (CCA) and power parameter optimization to maximize the 
  correlation between linear combinations of `X` and `Y`.
- The power parameter (`γ`) controls the balance between variance maximization and 
  correlation maximization. It is optimized within the specified bounds (`gamma_bounds`).
- If `Y_aux` is provided, it is concatenated with `Y` to form a combined response 
  matrix (`Y`), which is used during the fitting process.
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

julia> model.X_bar ≈ Matrix([1.4 1.4 1.2])

julia> model.gamma ≈ [0.700185836799654, 0.9366214237592033]
true
```
"""
function fit_cppls(
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    n_components::Integer = 2;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
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
    sample_classes = nothing,
) where {T1<:Real,T2<:Real}

    analysis_mode in (:regression, :discriminant) || throw(
        ArgumentError(
            "analysis_mode must be :regression or :discriminant, got $analysis_mode",
        ),
    )
    analysis_mode === :discriminant ||
        sample_classes === nothing ||
        throw(ArgumentError("sample_classes can only be provided for discriminant analysis"))

    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)

    n_predictors = size(X, 2)

    (
        X,
        Y_prim,
        Y,
        observation_weights,
        X_bar,
        Y_bar,
        X_def,
        W_comp,
        P,
        C,
        zero_mask,
        B,
        n_samples_X,
        n_targets_Y,
    ) = cppls_prepare_data(
        X,
        Y_prim,
        n_components,
        Y_aux,
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
        (
            wᵢ,
            rho[i],
            a[:, i],
            b[:, i],
            gamma_vals[i],
            W0ᵢ,
        ) = (compute_cppls_weights(
            X_def,
            Y,
            Y_prim,
            observation_weights,
            gamma,
            gamma_rel_tol,
            gamma_abs_tol,
        ))
        W0[:, :, i] = W0ᵢ
        Z[:, :, i] = X_def * W0ᵢ

        tᵢ, tᵢ_squared_norm, cᵢ = process_component!(
            i,
            X_def,
            wᵢ,
            Y_prim,
            W_comp,
            P,
            C,
            B,
            zero_mask,
            X_tolerance,
            X_loading_weight_tolerance,
            t_squared_norm_tolerance,
        )

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

    CPPLSFit(
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
        gamma_vals,
        rho,
        zero_mask,
        a,
        b,
        W0,
        Z;
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = response_labels,
        analysis_mode = analysis_mode,
        sample_classes = sample_classes,
    )
end

function fit_cppls(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    observation_weights::Union{AbstractVector{<:Real},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    sample_labels::AbstractVector = String[],
    predictor_labels::AbstractVector = String[],
    response_labels::AbstractVector = String[],
    sample_classes = nothing,
)
    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)
    return fit_cppls(
        X,
        Y_prim,
        model.n_components;
        _cppls_model_fit_kwargs_with_mode(model)...,
        observation_weights = observation_weights,
        Y_aux = Y_aux,
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = response_labels,
        sample_classes = sample_classes,
    )
end

"""
    StatsAPI.fit(model::CPPLSSpec, X, Y; kwargs...)

Fit a CPPLS model using the StatsAPI interface with an explicit model specification.
You normally call this as `fit(spec, X, Y)` after `using CPPLS`. Use `CPPLS.fit` or
`StatsAPI.fit` only when you need to disambiguate name conflicts.

Keywords mirror `fit_cppls` and are split into:

Model-spec settings (already stored in `model`):
- `n_components`, `gamma`, `center`, `analysis_mode`,
- `X_tolerance`, `X_loading_weight_tolerance`,
- `t_squared_norm_tolerance`, `gamma_rel_tol`, `gamma_abs_tol`.

Data/metadata settings (passed to `fit`):
- `observation_weights`: optional sample weights (vector, length = n_samples).
- `Y_aux`: optional auxiliary responses (matrix with n_samples rows). The legacy
  keyword `Y_auxiliary` is accepted as an alias.
- `sample_labels`, `predictor_labels`, `response_labels`: metadata for diagnostics.
- `sample_classes`: override categorical levels for discriminant analysis.

# Example
```
using CPPLS

spec = CPPLSSpec(n_components=2, gamma=0.5)
model = fit(spec, X, Y; observation_weights=weights, sample_labels=ids)
preds = predict(model, X)
```
"""
function fit(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim;
    kwargs...,
)
    fit_cppls(model, X, Y_prim; kwargs...)
end

"""
    fit_cppls(X, labels::AbstractCategoricalArray, n_components=2; kwargs...)
    fit_cppls(X, labels::AbstractVector, n_components=2; kwargs...)

Discriminant-analysis variants of `fit_cppls`. The first method dispatches
specifically on `CategoricalVector`/`CategoricalArray` inputs so users can opt into DA
behaviour through the type signature alone. The second method accepts any other label
container (e.g. plain `Vector{String}` or `Vector{Symbol}`) but follows the exact same
code path. Both convert the labels to a one-hot response matrix internally and store
the inferred class names inside the returned `CPPLSFit` model.

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
    X::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U},
    n_components::Integer = 2;
    kwargs...,
) where {T,R,V,C,U}
    fit_cppls_from_labels(X, labels, n_components; kwargs...)
end

function fit_cppls(
    X::AbstractMatrix{<:Real},
    labels::AbstractVector,
    n_components::Integer = 2;
    kwargs...,
)
    fit_cppls_from_labels(X, labels, n_components; kwargs...)
end

function fit_cppls(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...,
) where {T,R,V,C,U}
    model.analysis_mode === :discriminant || throw(
        ArgumentError(
            "CPPLSSpec must use analysis_mode=:discriminant when fitting from labels.",
        ),
    )
    fit_cppls_from_labels(
        X,
        labels,
        model.n_components;
        _cppls_model_fit_kwargs(model)...,
        kwargs...,
    )
end

function fit_cppls(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    labels::AbstractVector;
    kwargs...,
)
    model.analysis_mode === :discriminant || throw(
        ArgumentError(
            "CPPLSSpec must use analysis_mode=:discriminant when fitting from labels.",
        ),
    )
    fit_cppls_from_labels(
        X,
        labels,
        model.n_components;
        _cppls_model_fit_kwargs(model)...,
        kwargs...,
    )
end

function fit_cppls_from_labels(
    X::AbstractMatrix{<:Real},
    labels,
    n_components::Integer;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
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

    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)

    Y_prim, classes = labels_to_one_hot(labels)

    return fit_cppls(
        X,
        Y_prim,
        n_components;
        gamma = gamma,
        observation_weights = observation_weights,
        Y_aux = Y_aux,
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
        sample_classes = copy(labels),
    )
end

"""
    fit_cppls(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, n_components=2; kwargs...)

Regression-friendly convenience wrapper around `fit_cppls` that accepts a
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
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real},
    n_components::Integer = 2;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
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

    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)

    Y_matrix = reshape(Y_prim, :, 1)

    return fit_cppls(
        X,
        Y_matrix,
        n_components;
        gamma = gamma,
        observation_weights = observation_weights,
        Y_aux = Y_aux,
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

function fit_cppls(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real};
    observation_weights::Union{AbstractVector{<:Real},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    sample_labels::AbstractVector = String[],
    predictor_labels::AbstractVector = String[],
    response_labels::AbstractVector = String[],
)
    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)
    return fit_cppls(
        X,
        Y_prim,
        model.n_components;
        _cppls_model_fit_kwargs_with_mode(model)...,
        observation_weights = observation_weights,
        Y_aux = Y_aux,
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = response_labels,
    )
end


"""
    fit_cppls_light(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real},
        n_components::Integer;
        gamma::Union{<:Real, <:NTuple{2, <:Real}, <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}}=0.5,
        observation_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        Y_aux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}=nothing,
        center::Bool=true,
        X_tolerance::Real=1e-12,
        X_loading_weight_tolerance::Real=eps(Float64),
        gamma_rel_tol::Real=1e-6,
        gamma_abs_tol::Real=1e-12,
        t_squared_norm_tolerance::Real=1e-10,
        analysis_mode::Symbol=:regression)

Fit a CPPLS model but retain only the parts needed for prediction (`CPPLSFitLight`).

Arguments mirror `fit_cppls`, including support for scalar γ, `(lo, hi)` bounds, or
vectors that mix scalars and tuples as candidate sets. The returned `CPPLSFitLight` stores
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

julia> model.X_bar ≈ Matrix([1.4 1.4 1.2])
true
```
"""
function fit_cppls_light(
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    n_components::Integer = 2;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    t_squared_norm_tolerance::Real = 1e-10,
    analysis_mode::Symbol = :regression,
) where {T1<:Real,T2<:Real}

    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)

    (
        X,
        Y_prim,
        Y,
        observation_weights,
        X_bar,
        Y_bar,
        X_def,
        W_comp,
        P,
        C,
        zero_mask,
        B,
        _,
        _,
    ) = cppls_prepare_data(
        X,
        Y_prim,
        n_components,
        Y_aux,
        observation_weights,
        center,
    )

    for i = 1:n_components
        wᵢ, _, _, _, _, _ = compute_cppls_weights(
            X_def,
            Y,
            Y_prim,
            observation_weights,
            gamma,
            gamma_rel_tol,
            gamma_abs_tol,
        )

        process_component!(
            i,
            X_def,
            wᵢ,
            Y_prim,
            W_comp,
            P,
            C,
            B,
            zero_mask,
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

    CPPLSFitLight(B, X_bar, Y_bar, analysis_mode)
end

"""
    fit_cppls_light(X, labels::AbstractCategoricalArray, n_components=2; kwargs...)
    fit_cppls_light(X, labels::AbstractVector, n_components=2; kwargs...)

Discriminant-analysis convenience wrappers for `fit_cppls_light`. The first
signature dispatches explicitly on categorical arrays so callers can rely on the method
table to distinguish regression from DA. The second accepts any other label container
(e.g. vectors of strings, symbols, or enums) and forwards into the same code path.
Regardless of signature, labels are converted to a one-hot matrix for fitting, class
names are inferred once, and the returned `CPPLSFitLight` only retains the components needed
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

julia> light_plain.B ≈ light_cat.B
true
```
"""
function fit_cppls_light(
    X::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U},
    n_components::Integer = 2;
    kwargs...,
) where {T,R,V,C,U}
    fit_cppls_light_from_labels(X, labels, n_components; kwargs...)
end

function fit_cppls_light(
    X::AbstractMatrix{<:Real},
    labels::AbstractVector,
    n_components::Integer = 2;
    kwargs...,
)
    fit_cppls_light_from_labels(X, labels, n_components; kwargs...)
end

function fit_cppls_light(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...,
) where {T,R,V,C,U}
    model.analysis_mode === :discriminant || throw(
        ArgumentError(
            "CPPLSSpec must use analysis_mode=:discriminant when fitting from labels.",
        ),
    )
    fit_cppls_light_from_labels(
        X,
        labels,
        model.n_components;
        _cppls_model_fit_kwargs(model)...,
        kwargs...,
    )
end

function fit_cppls_light(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    labels::AbstractVector;
    kwargs...,
)
    model.analysis_mode === :discriminant || throw(
        ArgumentError(
            "CPPLSSpec must use analysis_mode=:discriminant when fitting from labels.",
        ),
    )
    fit_cppls_light_from_labels(
        X,
        labels,
        model.n_components;
        _cppls_model_fit_kwargs(model)...,
        kwargs...,
    )
end

function fit_cppls_light_from_labels(
    X::AbstractMatrix{<:Real},
    labels,
    n_components::Integer;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    t_squared_norm_tolerance::Real = 1e-10,
) where {T1<:Real,T2<:Real}
    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)
    Y_prim, _ = labels_to_one_hot(labels)

    fit_cppls_light(
        X,
        Y_prim,
        n_components;
        gamma = gamma,
        observation_weights = observation_weights,
        Y_aux = Y_aux,
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

Regression convenience wrapper for `fit_cppls_light` that accepts a single
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
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real},
    n_components::Integer = 2;
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    t_squared_norm_tolerance::Real = 1e-10,
) where {T1<:Real,T2<:Real}

    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)

    Y_matrix = reshape(Y_prim, :, 1)

    fit_cppls_light(
        X,
        Y_matrix,
        n_components;
        gamma = gamma,
        observation_weights = observation_weights,
        Y_aux = Y_aux,
        center = center,
        X_tolerance = X_tolerance,
        X_loading_weight_tolerance = X_loading_weight_tolerance,
        gamma_rel_tol = gamma_rel_tol,
        gamma_abs_tol = gamma_abs_tol,
        t_squared_norm_tolerance = t_squared_norm_tolerance,
        analysis_mode = :regression,
    )
end

function fit_cppls_light(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    observation_weights::Union{AbstractVector{<:Real},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
)
    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)
    return fit_cppls_light(
        X,
        Y_prim,
        model.n_components;
        _cppls_model_fit_kwargs_with_mode(model)...,
        observation_weights = observation_weights,
        Y_aux = Y_aux,
    )
end

function fit_cppls_light(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real};
    observation_weights::Union{AbstractVector{<:Real},Nothing} = nothing,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing} = nothing,
)
    Y_aux = resolve_Y_aux(Y_aux, Y_auxiliary)
    return fit_cppls_light(
        X,
        Y_prim,
        model.n_components;
        _cppls_model_fit_kwargs_with_mode(model)...,
        observation_weights = observation_weights,
        Y_aux = Y_aux,
    )
end


function process_component!(
    i::Integer,
    X_def::AbstractMatrix{<:Real},
    wᵢ::AbstractVector{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    W_comp::AbstractMatrix{<:Real},
    P::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    B::Array{<:Real,3},
    zero_mask::AbstractMatrix{Bool},
    X_tolerance::Real,
    X_loading_weight_tolerance::Real,
    tᵢ_squared_norm_tolerance::Real,
)

    wᵢ .= (
        wᵢ ./ norm(wᵢ) .*
        (abs.(wᵢ) .>= X_loading_weight_tolerance)
    )

    tᵢ = X_def * wᵢ
    tᵢ_squared_norm = tᵢ' * tᵢ

    if isapprox(tᵢ_squared_norm, 0.0)
        tᵢ_squared_norm += tᵢ_squared_norm_tolerance
    end
    pᵢ = (X_def' * tᵢ) / tᵢ_squared_norm
    cᵢ = (Y_prim' * tᵢ) / tᵢ_squared_norm

    X_def .-= tᵢ * pᵢ'

    zero_mask[i, :] .= vec(sum(abs.(X_def), dims = 1) .< X_tolerance)
    X_def[:, zero_mask[i, :]] .= 0

    W_comp[:, i] .= wᵢ
    P[:, i] .= pᵢ
    C[:, i] .= cᵢ
    B[:, :, i] .= (
        W_comp[:, 1:i] *
        pinv(P[:, 1:i]' * W_comp[:, 1:i]) *
        C[:, 1:i]'
    )

    tᵢ, tᵢ_squared_norm, cᵢ
end

function fitted(model::CPPLSFit)
    @views model.Y_hat[:, :, end]
end

function fitted(model::CPPLSFit, n_components::Integer)
    @views model.Y_hat[:, :, n_components]
end

function residuals(model::CPPLSFit)
    @views model.F[:, :, end]
end

function residuals(model::CPPLSFit, n_components::Integer)
    @views model.F[:, :, n_components]
end

function coef(model::AbstractCPPLSFit)
    @views model.B[:, :, end]
end

function coef(model::AbstractCPPLSFit, n_components::Integer)
    @views model.B[:, :, n_components]
end
