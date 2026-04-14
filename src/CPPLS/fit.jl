"""
    fit(m::CPPLSModel,
        X::AbstractMatrix{<:Real},
        Yprim::AbstractMatrix{<:Real};
        kwargs...
    )
    fit(m::CPPLSModel,
        X::AbstractMatrix{<:Real},
        Yprim::AbstractVector{<:Real};
        kwargs...
    )
    fit(m::CPPLSModel,
        X::AbstractMatrix{<:Real},
        sampleclasses::AbstractCategoricalArray{T,1,R,V,C,U};
        kwargs...
    ) where {T,R,V,C,U}

Fit a CPPLS model using the StatsAPI entry point and an explicit `CPPLSModel`. The model
specification supplies the number of components, the gamma configuration, centering, the
analysis mode, and all numerical tolerances, while the call to `fit` supplies data,
optional weights, additional responses, and label metadata.

The interpretation of the third argument depends on its type. When the third argument is
an `AbstractMatrix{<:Real}`, it is treated as the primary response block and used as-is;
the matrix may contain continuous responses, one-hot encoded class indicators, or other
custom encodings chosen by the user. When the third argument is an
`AbstractVector{<:Real}`, it is interpreted as a univariate numeric response and is
internally reshaped to a one-column matrix. When the third argument is an
`AbstractCategoricalArray`, it is interpreted as a vector of class labels; the labels are
converted internally to a one-hot response matrix, class names are inferred as response
labels, and the fit is performed in discriminant mode. In that case,
`m.analysis_mode` must be `:discriminant`, otherwise an `ArgumentError` is thrown.

The `gamma` setting in `model` may be a fixed scalar, a `(lo, hi)` tuple, or a vector
mixing scalars and tuples. Non-scalar settings trigger per-component selection by
choosing the value that yields the largest leading canonical correlation between the
supervised projection and the primary responses, and the resulting gamma values are
stored in the fitted model. The per-candidate gamma and squared canonical correlation
values examined during that search are also stored in the fitted model as matrices for
downstream diagnostics and plotting. A range such as `0:0.01:1` is treated as a grid of
fixed gamma values. To convert such a range into adjacent search intervals, use
`intervalize(0:0.01:1)`, which yields `[(0.0, 0.01), (0.01, 0.02), ...]`. If interval-wise
Brent searches are desired, pass `intervalize(...)` to `gamma`. Tuple intervals are
treated as closed intervals: both endpoints are evaluated explicitly, and the final
choice is the best among the two endpoints and the interior Brent minimizer.

Keyword arguments accepted by `fit` include `obs_weights` for per-sample weighting,
`Yadd` for additional response columns, and optional `samplelabels`, `predictorlabels`,
`responselabels`, and `sampleclasses` metadata for diagnostics and plotting. `Yadd` must
have the same number of rows as `X` and is concatenated internally to `Yprim` to build
the supervised projection, while prediction targets always remain the primary responses.
When `Yprim` is supplied as a matrix, `sampleclasses` may be used to store sample-level
class metadata on the fitted object without changing the response block used for fitting.
When class labels are passed positionally as an `AbstractCategoricalArray`, they define
the supervised response and are also stored as metadata. When `Yprim` is supplied as a
numeric vector, the fit is treated as univariate numeric-response fitting.

The return value is a `CPPLSFit` containing scores, loadings, regression coefficients,
gamma-selection diagnostics, and the metadata needed for downstream prediction and
diagnostics. Use `CPPLS.fit` or `StatsAPI.fit` when disambiguation is required in your
namespace.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`CPPLSModel`](@ref CPPLS.CPPLSModel),
[`coef`](@ref CPPLS.coef(::AbstractCPPLSFit)),
[`fitted`](@ref CPPLS.fitted(::CPPLSFit)),
[`gamma`](@ref CPPLS.gamma(::CPPLSFit)),
[`intervalize`](@ref),
[`invfreqweights`](@ref invfreqweights(::AbstractVector)),
[`predictorlabels`](@ref predictorlabels(::CPPLSFit)),
[`responselabels`](@ref responselabels(::CPPLSFit)),
[`residuals`](@ref residuals(::CPPLSFit)),
[`sampleclasses`](@ref sampleclasses(::CPPLSFit)),
[`samplelabels`](@ref samplelabels(::CPPLSFit)),
[`xscores`](@ref xscores(::CPPLSFit))

# Examples
```jldoctest
julia> using CategoricalArrays; using JLD2;

julia> file = CPPLS.dataset("synthetic_cppls_da_dataset.jld2");

julia> labels, X, classes, Yadd = load(file, "sample_labels", "X", "classes", "Y_add");

julia> classes = categorical(classes);  # make categorical

julia> m = CPPLSModel(ncomponents=2, gamma=0.01:0.01:1.00, analysis_mode=:discriminant)
CPPLSModel
  ncomponents: 2
  gamma: 0.01:0.01:1.0
  center_X: true
  scale_X: false
  scale_Yprim: false
  analysis_mode: discriminant

julia> cpplsfit = fit(m, X, classes; samplelabels=labels);

julia> size(CPPLS.xscores(cpplsfit))
(100, 2)

julia> m = CPPLSModel(ncomponents=2, gamma=0.75, analysis_mode=:discriminant)
CPPLSModel
  ncomponents: 2
  gamma: 0.75
  center_X: true
  scale_X: false
  scale_Yprim: false
  analysis_mode: discriminant

julia> cpplsfit = fit(m, X, classes; obs_weights=invfreqweights(classes), Yadd=Yadd)
CPPLSFit
  analysis_mode: discriminant
  samples: 100
  predictors: 14
  responses: 2
  components: 2
  gamma: [0.75, 0.75]
```
"""
function fit(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real};
    kwargs...
)
    fit_cppls(m, X, Yprim; kwargs...)
end

function fit(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Yprim::AbstractVector{<:Real};
    kwargs...
)
    fit_cppls(m, X, Yprim; kwargs...)
end

function fit(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    sampleclasses::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...
) where {T,R,V,C,U}

    fit_cppls(m, X, sampleclasses; kwargs...)
end

function fit_cppls(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Yadd::T2=nothing,
    samplelabels::T3=String[],
    predictorlabels::T4=String[],
    responselabels::T5=String[],
    sampleclasses::T6=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:AbstractVector,
    T6<:Union{AbstractVector, Nothing}  
}
    fit_cppls_core(m, X, Yprim;
        obs_weights=obs_weights, 
        Yadd=Yadd, 
        samplelabels=samplelabels,
        predictorlabels=predictorlabels, 
        responselabels=responselabels,
        sampleclasses=sampleclasses)
end

function fit_cppls(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Yprim::AbstractVector{<:Real};
    obs_weights::T1=nothing,
    Yadd::T2=nothing,
    samplelabels::T3=String[],
    predictorlabels::T4=String[],
    responselabels::T5=String[],
    sampleclasses::T6=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:AbstractVector,
    T6<:Union{AbstractVector, Nothing}
}

    if m.analysis_mode ≡ :discriminant
        throw(ArgumentError(
            "`Yprim::AbstractVector{<:Real}` is interpreted as a univariate numeric " * 
            "response and is not valid for `analysis_mode=:discriminant`. " *
            "Pass class labels as an `AbstractCategoricalArray`, or pass an explicitly " * 
            "encoded response matrix."
        ))
    end

    isnothing(sampleclasses) || throw(ArgumentError(
        "`sampleclasses` cannot be provided when `Yprim` is a numeric vector. " *
        "Use an `AbstractCategoricalArray` as the third argument for discriminant " * 
        "analysis instead."
    ))

    Yprim_matrix = reshape(Yprim, :, 1)

    fit_cppls_core(
        m, 
        X, 
        Yprim_matrix; 
        obs_weights=obs_weights, 
        Yadd=Yadd,
        samplelabels=samplelabels, 
        predictorlabels=predictorlabels, 
        responselabels=responselabels
    )
end

function fit_cppls(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    sampleclasses::AbstractCategoricalArray{T,1,R,V,C,U};
    Yadd::T1=nothing,
    obs_weights::T2=nothing,
    samplelabels::T3=String[],
    responselabels::T4=String[],
    predictorlabels::T5=String[]
) where {
    T, R, V, C, U,
    T1<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:AbstractVector
}

    isempty(responselabels) || throw(ArgumentError("`responselabels` cannot be provided" *
        " when passing sample classes; response labels are inferred automatically."))

    m.analysis_mode ≡ :discriminant || throw(ArgumentError(
        "CPPLSModel must use analysis_mode=:discriminant when passing class labels as " * 
        "an `AbstractCategoricalArray`"))
    
    Yprim, classes = onehot(sampleclasses)

    fit_cppls_core(m, X, Yprim; 
        obs_weights=obs_weights, 
        Yadd=Yadd,
        samplelabels=samplelabels, 
        predictorlabels=predictorlabels, 
        responselabels=classes,
        sampleclasses=copy(sampleclasses)
    )
end

"""
    fit_cppls_core(
        m::CPPLSModel,
        X::AbstractMatrix{<:Real},
        Yprim::AbstractMatrix{<:Real};
        kwargs...
    )

Low-level CPPLS fitting routine used by `fit`. Prefer `fit` with a CPPLSModel for the
public entry point and full parameter documentation.
"""
function fit_cppls_core(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Yadd::T2=nothing,
    samplelabels::T3=String[],
    predictorlabels::T4=String[],
    responselabels::T5=String[],
    sampleclasses::T6=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:AbstractVector,
    T6<:Union{AbstractVector, Nothing}
}
    # Validate that sampleclasses are only provided for discriminant analysis.
    m.analysis_mode ≡ :discriminant || isnothing(sampleclasses) || throw(ArgumentError(
        "sampleclasses can only be provided for discriminant analysis"))

    # Get predictor count.
    n_predictors = size(X, 2)

    # Preprocess data: center/scale, optionally with weights, and concatenate Yadd.
    d = preprocess(m, X, Yprim, Yadd, obs_weights)

    # Validate label lengths and generate default sample labels if needed.
    samplelabels = validate_label_length(samplelabels, d.nrow_X, "samplelabels")
    samplelabels = default_sample_labels(samplelabels, d.nrow_X)
    
    # Validate predictor label length (none or exact X column count).
    predictorlabels = 
        validate_label_length(predictorlabels, n_predictors, "predictorlabels")
    
    # Validate response label length (none or exact Yprim column count).
    responselabels = validate_label_length(responselabels, d.ncol_Y, "responselabels")

    # For discriminant analysis, responselabels must be provided to name the classes.
    if m.analysis_mode ≡ :discriminant && isempty(responselabels)
        throw(ArgumentError(
            "responselabels must list class names for discriminant analysis"))
    end

    # Preallocate arrays for scores, loadings, regression coefficients, and diagnostics.
    T = Matrix{Float64}(undef, d.nrow_X, m.ncomponents)
    a = Matrix{Float64}(undef, size(d.Y, 2), m.ncomponents)
    b = Matrix{Float64}(undef, d.ncol_Y, m.ncomponents)
    rho = Vector{Float64}(undef, m.ncomponents)
    gamma_vals = fill(0.5, m.ncomponents)
    gammas = Matrix{Float64}(undef, gamma_search_candidate_count(m.gamma),
        m.ncomponents)
    rhos = Matrix{Float64}(undef, gamma_search_candidate_count(m.gamma),
        m.ncomponents)
    t_norms = Vector{Float64}(undef, m.ncomponents)
    U = Matrix{Float64}(undef, d.nrow_X, m.ncomponents)
    Y_hat = Array{Float64}(undef, d.nrow_X, d.ncol_Y, m.ncomponents)
    W0 = Array{Float64}(undef, n_predictors, size(d.Y, 2), m.ncomponents)
    Z = Array{Float64}(undef, d.nrow_X, size(d.Y, 2), m.ncomponents)

    # Main loop over components: compute weights, scores, loadings, deflate, and 
    # store results.
    for i = 1:m.ncomponents
        wᵢ, rho[i], a[:, i], b[:, i], gamma_vals[i], W0ᵢ, gammas[:, i],
        rhos[:, i] = compute_cppls_weights(m, d.X_def, d.Y, d.Yprim, obs_weights,
            m.gamma)
        
        W0[:, :, i] = W0ᵢ
        Z[:, :, i] = d.X_def * W0ᵢ

        tᵢ, tᵢ_squared_norm, cᵢ = process_component!(m, i, d.X_def, wᵢ, d.Yprim, d.W_comp, 
            d.P, d.C, d.B, d.zero_mask)

        T[:, i] = tᵢ
        t_norms[i] = tᵢ_squared_norm
        U[:, i] = d.Yprim * cᵢ / (cᵢ' * cᵢ)

        if i > 1
            U[:, i] -= T * (T' * U[:, i] ./ t_norms)
        end
        Y_hat[:, :, i] = d.X * d.B[:, :, i]
    end

    F = d.Yprim .- Y_hat
    R = d.W_comp * pinv(d.P' * d.W_comp)
    X_var = vec(sum(d.P .* d.P, dims = 1)) .* t_norms
    X_var_total = sum(d.X .* d.X)

    CPPLSFit(d.B, T, d.P, d.W_comp, U, d.C, R, Y_hat, F, X_var, X_var_total, gamma_vals, 
        rho, gammas, rhos, d.zero_mask, a, b, W0, d.X_mean, d.X_std,
        d.Yprim_std; 
        samplelabels=samplelabels,
        predictorlabels=predictorlabels, 
        responselabels=responselabels,
        sampleclasses=sampleclasses,
        analysis_mode=m.analysis_mode
    )
end

############################################################################################
# Helper
############################################################################################
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
    default_sample_labels(labels::AbstractVector, n_samples::Integer) -> Vector{String}

Return `labels` when provided, otherwise generate default row-index labels `"1"` through
`string(n_samples)`.

Type stablity tested: 03/24/2026
"""
function default_sample_labels(labels::AbstractVector, n_samples::Integer)
    isempty(labels) ? string.(1:n_samples) : String[labels...]
end

"""
    process_component!(
        m::CPPLSModel,
        i::Int,
        X_def::AbstractMatrix{<:Real},
        wᵢ::AbstractVector{<:Real},
        Yprim::AbstractMatrix{<:Real},
        W_comp::AbstractMatrix{<:Real},
        P::AbstractMatrix{<:Real},
        C::AbstractMatrix{<:Real},
        B::Array{<:Real, 3},
        zero_mask::AbstractMatrix{Bool}
    )

Compute the i-th component from the current deflated predictors and update the CPPLS work 
arrays. The weight vector is normalized and thresholded, the score t is formed as
X_def * wᵢ, and the X and Y loadings are computed by regressing the predictors and
responses on t. The predictor block is then deflated along the component, the zero mask
and regression coefficients are updated, and the function returns the score, its squared
norm, and the Y loading vector.

Type stablity tested: 03/24/2026
"""
function process_component!(
    m::CPPLSModel,
    i::Int,
    X_def::AbstractMatrix{<:Real},
    wᵢ::AbstractVector{<:Real},
    Yprim::AbstractMatrix{<:Real},
    W_comp::AbstractMatrix{<:Real},
    P::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    B::Array{<:Real, 3},
    zero_mask::AbstractMatrix{Bool}
)

    wᵢ .= wᵢ ./ norm(wᵢ) .* (abs.(wᵢ) .≥ m.X_loading_weight_tolerance)

    tᵢ = X_def * wᵢ
    tᵢ_squared_norm = tᵢ' * tᵢ

    if isapprox(tᵢ_squared_norm, 0.0)
        tᵢ_squared_norm += m.t_squared_norm_tolerance
    end

    pᵢ = (X_def' * tᵢ) / tᵢ_squared_norm
    cᵢ = (Yprim' * tᵢ) / tᵢ_squared_norm

    X_def .-= tᵢ * pᵢ'

    zero_mask[i, :] .= vec(sum(abs.(X_def), dims=1) .< m.X_tolerance)
    X_def[:, zero_mask[i, :]] .= 0

    W_comp[:, i] .= wᵢ
    P[:, i] .= pᵢ
    C[:, i] .= cᵢ
    B[:, :, i] .= W_comp[:, 1:i] * pinv(P[:, 1:i]' * W_comp[:, 1:i]) * C[:, 1:i]'

    tᵢ, tᵢ_squared_norm, cᵢ
end
