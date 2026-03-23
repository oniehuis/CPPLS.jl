"""
    colmean(M::AbstractMatrix{<:Real}, ::Nothing)
    colmean(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})

Compute the (optionally weighted) column means of M.
"""
colmean(M::AbstractMatrix{<:Real}, ::Nothing) = mean(M, dims=1)
colmean(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real}) = 
    (obs_weights' * M) / sum(obs_weights)

"""
    colstd(M::AbstractMatrix{<:Real}, ::Nothing)
    colstd(M::AbstractMatrix{<:Real}, ::Nothing, μ::AbstractVector{<:Real})
    colstd(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})
    colstd(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real}, μ::AbstractVector{<:Real})

Compute the (optionally weighted) column standard deviations of `M`.
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
    μ # ::AbstractVector{<:Real} # can be LinearAlgebar.adjoint
)
    sqrt.(sum((M .- μ).^2, dims=1) / (size(M, 1) - 1))
end
function colstd(
    M::AbstractMatrix{<:Real}, 
    obs_weights::AbstractVector{<:Real}, 
    μ # ::AbstractVector{<:Real} # can be LinearAlgebar.adjoint
)
    sqrt.(sum(obs_weights .* (M .- μ).^2, dims=1) / sum(obs_weights))
end

"""
    center_and_scale(
        M::AbstractMatrix{<:Real},
        center::Bool, 
        scale::Bool, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
)

Center and/or scale the columns of M. Returns (M_trans, means, stds).
If obs_weights is provided, use weighted mean and std.
"""
function center_and_scale(
    M::AbstractMatrix{<:Real}, 
    center::Bool, 
    scale::Bool, 
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)
    if center
        μ = colmean(M, obs_weights)
        M_centered = M .- μ
    else
        μ = zeros(1, size(M,2))
        M_centered = M
    end

    if scale && center
        σ = colstd(M_centered, obs_weights, μ)
        # Safeguard: set zero stds to 1.0 to avoid division by zero
        σ = map(x -> x == 0.0 ? 1.0 : x, σ)
        M_scaled = M_centered ./ σ
    elseif scale && !center
        σ = colstd(M, obs_weights)
        σ = map(x -> x == 0.0 ? 1.0 : x, σ)
        M_scaled = M ./ σ
    else
        σ = ones(1, size(M,2))
        M_scaled = M_centered
    end

    M_scaled, vec(μ), vec(σ)
end

"""
    center_mean(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})
    center_mean(M::AbstractMatrix{<:Real}, ::Nothing)

Center the columns of `M` and return the centered matrix together with the column means.
When weights are provided, the mean is computed as a weighted mean using the observation
weights, otherwise it is the ordinary column mean.
"""
function center_mean(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})
    M̄ = Matrix((obs_weights' * M) / sum(obs_weights))
    M .- M̄, M̄
end

function center_mean(M::AbstractMatrix{<:Real}, ::Nothing)
    M̄ = mean(M, dims=1)
    M .- M̄, M̄
end

"""
    centerscale(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})
    centerscale(M::AbstractMatrix{<:Real}, ::Nothing)

Center `M` and apply observation weights in a single step. With weights, each row is
centered by the weighted mean and then scaled by the weights. Without weights, only
centering is performed.
"""
centerscale(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real}) =
    (M .- (obs_weights' * M) / sum(obs_weights)) .* obs_weights

centerscale(M::AbstractMatrix{<:Real}, ::Nothing) = M .- mean(M, dims=1)

"""
    convert_to_float64(M::AbstractMatrix{T}) where {T<:Real}
    convert_to_float64(v::AbstractVector{T}) where {T<:Real}

Convert numeric matrices and vectors to `Float64` storage when needed, returning the
input unchanged if it is already `Float64`.
"""
convert_to_float64(M::AbstractMatrix{T}) where {T<:Real} =
    (T ≠ Float64 ? convert(Matrix{Float64}, M) : M)

convert_to_float64(v::AbstractVector{T}) where {T<:Real} =
    (T ≠ Float64 ? convert(Vector{Float64}, v) : v)

"""
    convert_auxiliary_to_float64(Y::LinearAlgebra.AbstractVecOrMat)

Convert auxiliary response data to `Float64`, returning either a `Matrix{Float64}` or
`Vector{Float64}` depending on the input shape.
"""
function convert_auxiliary_to_float64(Y::LinearAlgebra.AbstractVecOrMat)
    if Y isa AbstractMatrix
        return Y isa Matrix{Float64} ? Y : Matrix{Float64}(Y)
    else
        return Y isa Vector{Float64} ? Y : Vector{Float64}(Y)
    end
end

"""
    convert_auxiliary_to_float64(Y)

Raise an error when auxiliary responses are not provided as a vector or matrix.
"""
function convert_auxiliary_to_float64(Y)
    throw(ArgumentError("Y_aux must be a vector or matrix"))
end

"""
    cppls_prepare_data(
        m::CPPLSModel,
        X::AbstractMatrix{<:Real}, 
        Yprim::AbstractMatrix{<:Real}, 
        Yaux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Prepare input data for CPPLS by converting to `Float64`, validating dimensions, applying
optional centering and scaling, constructing the combined response matrix, and
allocating working arrays needed by the fit routine.
"""
function cppls_prepare_data(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real},
    Yaux::Union{LinearAlgebra.AbstractVecOrMat, Nothing},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)
    n_samples_X, n_features_X = size(X)
    n_samples_Y, n_targets_Y = size(Yprim)
    n_samples_X ≠ n_samples_Y && throw(DimensionMismatch(
        "Number of rows in X and Yprim must be equal"))
    !isnothing(obs_weights) && length(obs_weights) ≠ n_samples_X && throw(
        DimensionMismatch(
            "Length of observation_weights must match the number of rows in X and Yprim"))

    X = convert_to_float64(X)
    X_z, X_mean, X_std = center_and_scale(X, m.center_X, m.scale_X, obs_weights)

    Yprim = convert_to_float64(Yprim)
    Yprim_z, Yprim_mean, Yprim_std = center_and_scale(Yprim, m.center_Y, m.scale_Y, obs_weights)

    if isnothing(Yaux)
        Yaux_z, Yaux_mean, Yaux_std = nothing, nothing, nothing
        Y_z = Yprim_z
    else
        Yaux = convert_auxiliary_to_float64(Yaux)
        Yaux_z, Yaux_mean, Yaux_std = center_and_scale(Yaux, m.center_Yaux, m.scale_Yaux, obs_weights)
        Y_z = hcat(Yprim_z, Yaux_z)
    end

    X_def = copy(X_z)
    W_comp = Matrix{Float64}(undef, n_features_X, ncomponents(m))
    P = Matrix{Float64}(undef, n_features_X, ncomponents(m))
    C = Matrix{Float64}(undef, n_targets_Y, ncomponents(m))
    zero_mask = Matrix{Bool}(undef, (ncomponents(m), n_features_X))
    B = Array{Float64}(undef, (n_features_X, n_targets_Y, ncomponents(m)))
    
    (X=X_z, Y_prim=Yprim_z, Y=Y_z, X_def=X_def, W_comp=W_comp, P=P, C=C, zero_mask=zero_mask, B=B, n_samples_X=n_samples_X, 
    n_targets_Y=n_targets_Y, X_z=X_z, X_mean=X_mean, X_std=X_std, Yprim_z=Yprim_z, 
    Yprim_mean=Yprim_mean, Yprim_std=Yprim_std, Yaux_z=Yaux_z, Yaux_mean=Yaux_mean, 
    Yaux_std=Yaux_std)
end

# X_z is redundnantly returned
# Yprim_z is redundantly returned