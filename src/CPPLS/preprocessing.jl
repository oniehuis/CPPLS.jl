function center_mean(M::AbstractMatrix{<:Real}, observation_weights::AbstractVector{<:Real})
    M̄ = Matrix((observation_weights' * M) / sum(observation_weights))
    M .- M̄, M̄
end


function center_mean(M::AbstractMatrix{<:Real}, ::Nothing)
    M̄ = mean(M, dims = 1)
    M .- M̄, M̄
end


centerscale(M::AbstractMatrix{<:Real}, observation_weights::AbstractVector{<:Real}) =
    (M .- (observation_weights' * M) / sum(observation_weights)) .* observation_weights


centerscale(M::AbstractMatrix{<:Real}, ::Nothing) = M .- mean(M, dims = 1)


convert_to_float64(M::AbstractMatrix{T}) where {T<:Real} =
    (T ≠ Float64 ? convert(Matrix{Float64}, M) : M)

convert_to_float64(v::AbstractVector{T}) where {T<:Real} =
    (T ≠ Float64 ? convert(Vector{Float64}, v) : v)

function convert_auxiliary_to_float64(Y::LinearAlgebra.AbstractVecOrMat)
    if Y isa AbstractMatrix
        return Y isa Matrix{Float64} ? Y : Matrix{Float64}(Y)
    else
        return Y isa Vector{Float64} ? Y : Vector{Float64}(Y)
    end
end

function convert_auxiliary_to_float64(Y)
    throw(ArgumentError("Y_auxiliary must be a vector or matrix"))
end

function cppls_prepare_data(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    n_components::Integer,
    Y_auxiliary::Union{LinearAlgebra.AbstractVecOrMat,Nothing},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    center::Bool,
)

    X_predictors = convert_to_float64(X_predictors)
    Y_responses = convert_to_float64(Y_responses)

    if Y_auxiliary !== nothing
        Y_auxiliary = convert_auxiliary_to_float64(Y_auxiliary)
    end

    Y_combined = isnothing(Y_auxiliary) ? Y_responses : hcat(Y_responses, Y_auxiliary)

    n_samples_X, n_features_X = size(X_predictors)
    n_samples_Y, n_targets_Y = size(Y_responses)
    n_samples_X ≠ n_samples_Y && throw(
        DimensionMismatch("Number of rows in X_predictors and Y_responses must be equal"),
    )
    if !isnothing(observation_weights) && length(observation_weights) ≠ n_samples_X
        throw(
            DimensionMismatch(
                "Length of observation_weights must match the number of " *
                "rows in X_predictors and Y_responses",
            ),
        )
    end

    if center
        X_predictors, X̄_mean = center_mean(X_predictors, observation_weights)
        Ȳ_mean = mean(Y_responses, dims = 1)
    else
        X̄_mean = zeros(Float64, (1, n_features_X))
        Ȳ_mean = zeros(Float64, (1, n_targets_Y))
    end

    X_deflated = copy(X_predictors)
    X_loading_weights = Matrix{Float64}(undef, n_features_X, n_components)
    X_loadings = Matrix{Float64}(undef, n_features_X, n_components)
    Y_loadings = Matrix{Float64}(undef, n_targets_Y, n_components)
    small_norm_flags = Matrix{Bool}(undef, (n_components, n_features_X))
    regression_coefficients =
        Array{Float64}(undef, (n_features_X, n_targets_Y, n_components))


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
    )
end
