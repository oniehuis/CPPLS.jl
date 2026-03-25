"""
    preprocess(
        m::CPPLSModel,
        X::AbstractMatrix{<:Real}, 
        Yprim::AbstractMatrix{<:Real}, 
        Yaux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Prepare input data for CPPLS by converting to `Float64`, validating dimensions, applying
optional centering and scaling, constructing the combined response matrix, and
allocating working arrays needed by the fit routine.

Type stablity tested: 03/25/2026
"""
function preprocess(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real},
    Yaux::Union{LinearAlgebra.AbstractVecOrMat, Nothing},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)
    nrow_X, ncol_X = size(X)
    nrow_Y, ncol_Y = size(Yprim)

    nrow_X ≠ nrow_Y && throw(DimensionMismatch(
        "Number of rows in X and Yprim must be equal"))
    !isnothing(obs_weights) && length(obs_weights) ≠ nrow_X && throw(DimensionMismatch(
        "Length of observation_weights must match the number of rows in X and Yprim"))

    X = float64(X)
    X_z, X_mean, X_std = centerscale(X, m.center_X, m.scale_X, obs_weights)

    Yprim = float64(Yprim)
    Yprim_z, Yprim_mean, Yprim_std = centerscale(Yprim, m.center_Y, m.scale_Y, obs_weights)

    if isnothing(Yaux)
        Yaux_z, Yaux_mean, Yaux_std = nothing, nothing, nothing
        Y_z = Yprim_z
    else
        Yaux = float64(Yaux)
        Yaux_z, Yaux_mean, Yaux_std = 
            centerscale(Yaux, m.center_Yaux, m.scale_Yaux, obs_weights)
        Y_z = hcat(Yprim_z, Yaux_z)
    end

    Xdef      = copy(X_z)
    B         = Array{Float64}(undef, (ncol_X, ncol_Y, ncomponents(m)))
    C         = Matrix{Float64}(undef, ncol_Y, ncomponents(m))
    P         = Matrix{Float64}(undef, ncol_X, ncomponents(m))
    W_comp    = Matrix{Float64}(undef, ncol_X, ncomponents(m))
    zero_mask = Matrix{Bool}(undef, (ncomponents(m), ncol_X))
    
    (   # Preprocessed predictors
        X=X_z,
        X_mean=X_mean, 
        X_std=X_std, 
      
        # Preprocessed combined (Yprim and Yaux) responses
        Y=Y_z, 
      
        # Preprocessed primary responses
        Yprim=Yprim_z,
        Yprim_mean=Yprim_mean, 
        Yprim_std=Yprim_std, 

        # Preprocessed auxiliary responses
        Yaux_mean=Yaux_mean, 
        Yaux_std=Yaux_std,

        # Dimensions
        n_samples_X=nrow_X, 
        n_targets_Y=ncol_Y, 
      
        # Working arrays for fit routine
        X_def=Xdef, 
        B=B, 
        C=C,
        P=P, 
        W_comp=W_comp, 
        zero_mask=zero_mask
    )
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

Type stablity tested: 03/25/2026
"""
function centerscale(
    M::AbstractMatrix{T},
    center::Bool,
    scale::Bool,
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
) where {T<:Real}
    M_working = M

    if center
        μ = convert(Matrix{T}, colmean(M, obs_weights))
        M_working = M .- μ
    else
        μ = zeros(T, 1, size(M, 2))
    end

    if scale
        σ = scale && center ? 
            convert(Matrix{T}, colstd(M_working, obs_weights, μ)) : 
            convert(Matrix{T}, colstd(M, obs_weights))
        σ = map(x -> x == 0.0 ? one(T) : x, σ)
        M_working = M_working ./ σ
    else
        σ = ones(T, 1, size(M, 2))
    end

    M_working, vec(μ), vec(σ)
end

"""
    colmean(M::AbstractMatrix{<:Real}, ::Nothing)
    colmean(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})

Compute the (optionally weighted) column means of M.

Type stablity tested: 03/25/2026
"""
colmean(M::AbstractMatrix{<:Real}, ::Nothing) = mean(M, dims=1)

colmean(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real}) = 
    (obs_weights' * M) / sum(obs_weights)

"""
    colstd(M::AbstractMatrix{<:Real}, ::Nothing)
    colstd(M::AbstractMatrix{<:Real}, ::Nothing, μ)
    colstd(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})
    colstd(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real}, μ)

Compute the (optionally weighted) column standard deviations of `M`.

Type stablity tested: 03/25/2026
"""
function colstd(M::AbstractMatrix{<:Real}, ::Nothing)
    μ = colmean(M, nothing)
    sqrt.(sum((M .- μ).^2, dims=1) / (size(M, 1) - 1))
end

function colstd(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})
    μ = colmean(M, obs_weights)
    sqrt.(sum(obs_weights .* (M .- μ).^2, dims=1) / sum(obs_weights))
end

function colstd(
    M::AbstractMatrix{<:Real}, 
    ::Nothing, 
    μ
)
    sqrt.(sum((M .- μ).^2, dims=1) / (size(M, 1) - 1))
end

function colstd(
    M::AbstractMatrix{<:Real}, 
    obs_weights::AbstractVector{<:Real}, 
    μ
)
    sqrt.(sum(obs_weights .* (M .- μ).^2, dims=1) / sum(obs_weights))
end

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
