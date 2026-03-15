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
        X::AbstractMatrix{<:Real}, 
        Y_prim::AbstractMatrix{<:Real}, 
        n_components::Integer, 
        Y_aux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}, 
        center::Bool
    )

Prepare input data for CPPLS by converting to `Float64`, validating dimensions, applying
optional centering, constructing the combined response matrix, and allocating working
arrays needed by the fit routine.
"""
function cppls_prepare_data(
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    n_components::Integer,
    Y_aux::Union{LinearAlgebra.AbstractVecOrMat, Nothing},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
    center::Bool
)

    X = convert_to_float64(X)
    Y_prim = convert_to_float64(Y_prim)

    if !isnothing(Y_aux)
        Y_aux = convert_auxiliary_to_float64(Y_aux)
    end

    Y = isnothing(Y_aux) ? Y_prim : hcat(Y_prim, Y_aux)

    n_samples_X, n_features_X = size(X)
    n_samples_Y, n_targets_Y = size(Y_prim)
    n_samples_X ≠ n_samples_Y && throw(
        DimensionMismatch("Number of rows in X and Y_prim must be equal"))

    if !isnothing(obs_weights) && length(obs_weights) ≠ n_samples_X
        throw(DimensionMismatch(
            "Length of observation_weights must match the number of rows in X and Y_prim"))
    end

    if center
        X, X_bar = center_mean(X, obs_weights)
        Y_bar = mean(Y_prim, dims = 1)
    else
        X_bar = zeros(Float64, (1, n_features_X))
        Y_bar = zeros(Float64, (1, n_targets_Y))
    end

    X_def = copy(X)
    W_comp = Matrix{Float64}(undef, n_features_X, n_components)
    P = Matrix{Float64}(undef, n_features_X, n_components)
    C = Matrix{Float64}(undef, n_targets_Y, n_components)
    zero_mask = Matrix{Bool}(undef, (n_components, n_features_X))
    B = Array{Float64}(undef, (n_features_X, n_targets_Y, n_components))

    (X, Y_prim, Y, obs_weights, X_bar, Y_bar, X_def, W_comp, P, C, zero_mask, B,
        n_samples_X, n_targets_Y)
end
