"""
    preprocess(
        m::CPPLSModel,
        X::AbstractMatrix{<:Real}, 
        Yprim::AbstractMatrix{<:Real}, 
        Yaux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing},
        response_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        target_weights::Union{AbstractVector{<:Real}, Nothing}=nothing
    ) -> Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}

Prepare input data for CPPLS by converting to `Float64`, validating dimensions, applying
optional centering and scaling, constructing the combined response matrix, and
allocating working arrays needed by the fit routine. `response_weights` and
`target_weights` are validated and converted here, but their mathematical effect is
applied later during supervised-weight construction and CCA alignment.

Type stablity tested: 03/26/2026
"""
function preprocess(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real},
    Yaux::Union{LinearAlgebra.AbstractVecOrMat, Nothing},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
    response_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
    target_weights::Union{AbstractVector{<:Real}, Nothing}=nothing
)
    nrow_X, ncol_X = size(X)
    nrow_Y, ncol_Y = size(Yprim)

    nrow_X ≠ nrow_Y && throw(DimensionMismatch(
        "Number of rows in X and Yprim must be equal"))
    !isnothing(obs_weights) && length(obs_weights) ≠ nrow_X && throw(DimensionMismatch(
        "Length of observation_weights must match the number of rows in X and Yprim"))

    X, X_mean, X_std = centerscale(float64(X), m.center_X, m.scale_X, obs_weights)

    Yprim, _, Yprim_std = centerscale(float64(Yprim), false, m.scale_Yprim, obs_weights)

    Y = isnothing(Yaux) ? Yprim : hcat(Yprim, float64(Yaux))
    response_weights = prepare_response_weights(response_weights, size(Y, 2), "response_weights")
    target_weights = prepare_response_weights(target_weights, size(Yprim, 2), "target_weights")

    X_def     = copy(X)
    B         = Array{Float64}(undef, (ncol_X, ncol_Y, ncomponents(m)))
    C         = Matrix{Float64}(undef, ncol_Y, ncomponents(m))
    P         = Matrix{Float64}(undef, ncol_X, ncomponents(m))
    W_comp    = Matrix{Float64}(undef, ncol_X, ncomponents(m))
    zero_mask = Matrix{Bool}(undef, (ncomponents(m), ncol_X))
    
    (   # Preprocessed predictors
        X=X,
        X_mean=X_mean, 
        X_std=X_std, 

        # Preprocessed primary responses
        Yprim=Yprim,
        Yprim_std=Yprim_std, 
        response_weights=response_weights,
        target_weights=target_weights,
      
        # Preprocessed combined (Yprim and Yaux) responses
        Y=Y, 
      
        # Dimensions
        nrow_X=nrow_X, 
        ncol_Y=ncol_Y, 
      
        # Working arrays for fit routine
        X_def=X_def, 
        B=B, 
        C=C,
        P=P, 
        W_comp=W_comp, 
        zero_mask=zero_mask
    )
end

function prepare_response_weights(
    weights::Union{AbstractVector{<:Real}, Nothing},
    expected::Integer,
    name::AbstractString
)
    if isnothing(weights)
        return ones(Float64, expected)
    end

    length(weights) == expected || throw(DimensionMismatch(
        "Length of $name must be $expected, got $(length(weights))"))

    weights_float = float64(weights)
    all(isfinite, weights_float) || throw(ArgumentError(
        "$name must contain only finite values"))
    any(w -> w < 0, weights_float) && throw(ArgumentError(
        "$name must be non-negative"))
    sum(weights_float) > 0 || throw(ArgumentError(
        "$name must sum to a positive value"))

    weights_float
end

# Helper functions

"""
    centerscale(
        M::AbstractMatrix{<:Real},
        center::Bool, 
        scale::Bool, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
)

Center and/or scale the columns of M. Returns (M_trans, means, stds).
If obs_weights is provided, use weighted mean and std.

Type stablity tested: 03/26/2026
"""
function centerscale(
    M::AbstractMatrix{<:Real},
    center::Bool,
    scale::Bool,
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
) 
    validate_obs_weights(M, obs_weights)

    M_float = float64(M)
    M_working = M_float

    if center
        μ = colmean(M_float, obs_weights)
        M_working = M_float .- μ
    else
        μ = zeros(Float64, 1, size(M, 2))
    end

    if scale
        σ = center ? centeredcolstd(M_working, obs_weights) : colstd(M_float, obs_weights)
        σ = map(x -> isfinite(x) && x ≠ 0.0 ? x : 1.0, σ)
        M_working = M_working ./ σ
    else
        σ = ones(Float64, 1, size(M, 2))
    end

    M_working, vec(μ), vec(σ)
end

function validate_obs_weights(
    M::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)
    isnothing(obs_weights) && return nothing

    length(obs_weights) == size(M, 1) || throw(DimensionMismatch(
        "Length of obs_weights must match the number of rows in M"))
    all(isfinite, obs_weights) || throw(ArgumentError(
        "obs_weights must contain only finite values"))
    any(w -> w < 0, obs_weights) && throw(ArgumentError(
        "obs_weights must be non-negative"))
    sum(obs_weights) > 0 || throw(ArgumentError(
        "obs_weights must sum to a positive value"))

    nothing
end

"""
    colmean(M::AbstractMatrix{<:Real}, ::Nothing)
    colmean(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})

Compute the (optionally weighted) column means of M.

Type stablity tested: 03/26/2026
"""
colmean(M::AbstractMatrix{<:Real}, ::Nothing) = mean(M, dims=1)

colmean(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real}) =
    reshape((M' * obs_weights) / sum(obs_weights), 1, :)

"""
    colstd(M::AbstractMatrix{<:Real}, ::Nothing)
    colstd(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})

Compute the (optionally weighted) column standard deviations of `M`.

Type stablity tested: 03/26/2026
"""
function colstd(M::AbstractMatrix{<:Real}, ::Nothing)
    μ = colmean(M, nothing)
    sqrt.(sum((M .- μ).^2, dims=1) / size(M, 1))
end

function colstd(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})
    μ = colmean(M, obs_weights)
    sqrt.(sum(obs_weights .* (M .- μ).^2, dims=1) / sum(obs_weights))
end

"""
    centeredcolstd(M::AbstractMatrix{<:Real}, ::Nothing)
    centeredcolstd(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})

Compute column standard deviations of a matrix that is already centered.
Type stablity tested: 03/26/2026
"""
centeredcolstd(M::AbstractMatrix{<:Real}, ::Nothing) =
    sqrt.(sum(M .^ 2, dims = 1) / size(M, 1))

centeredcolstd(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real}) =
    sqrt.(sum(obs_weights .* M .^ 2, dims = 1) / sum(obs_weights))

"""
    float64(M::AbstractMatrix{T}) where {T<:Real}
    float64(v::AbstractVector{T}) where {T<:Real}

Convert numeric matrices and vectors to `Float64` storage when needed, returning the
input unchanged if it is already `Float64`.

Type stablity tested: 03/25/2026
"""
float64(M::AbstractMatrix{T}) where {T<:Real} =
    (T ≠ Float64 ? convert(Matrix{Float64}, M) : M)

float64(v::AbstractVector{T}) where {T<:Real} = 
    (T ≠ Float64 ? convert(Vector{Float64}, v) : v)
