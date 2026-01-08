function Iᵣ(rowcount::Integer, columncount::Integer)
    M = zeros(rowcount, columncount)
    @inbounds for i = 1:min(rowcount, columncount)
        M[i, i] = 1
    end
    M
end


function cca_decomposition(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, ::Nothing)

    cca_decomposition(X, Y)
end


function cca_decomposition(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    observation_weights::AbstractVector{<:Real},
)

    cca_decomposition(X .* observation_weights, Y .* observation_weights)
end


function cca_decomposition(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    n_rows, n_cols = size(X)

    qx = qr(X, ColumnNorm())
    qy = qr(Y, ColumnNorm())

    dx = rank(qx.R)
    dy = rank(qy.R)

    @inbounds if dx == 0
        throw(ErrorException("X has rank 0"))
    end
    @inbounds if dy == 0
        throw(ErrorException("Y has rank 0"))
    end

    A = ((qx.Q'*qy.Q)*Iᵣ(n_rows, dy))[1:dx, :]
    left_singular_vecs, singular_vals, right_singular_vecs_t = svd(A; full = true)
    right_singular_vecs = right_singular_vecs_t'
    max_canonical_correlation = clamp(first(singular_vals), 0.0, 1.0)

    n_rows,
    n_cols,
    qx,
    qy,
    dx,
    dy,
    left_singular_vecs,
    right_singular_vecs,
    max_canonical_correlation
end


cca_corr(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
) = last(cca_decomposition(X, Y, observation_weights))


function cca_coeffs_and_corr(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
)
    (
        (
            n_rows,
            n_cols,
            qx,
            qy,
            dx,
            dy,
            left_singular_vectors,
            right_singular_vectors,
            max_canonical_correlation,
        ) = cca_decomposition(X, Y, observation_weights)
    )

    k = min(dx, dy)
    canonical_coefficients = qx.R[1:dx, 1:dx] \ left_singular_vectors[:, 1:k]
    canonical_coefficients *= sqrt(n_rows - 1)

    remaining_rows = n_cols - size(canonical_coefficients, 1)
    if remaining_rows > 0
        canonical_coefficients =
            vcat(canonical_coefficients, zeros(remaining_rows, k))
    end

    canonical_coefficients = canonical_coefficients[invperm(qx.p), :]

    canonical_coefficients_y = qy.R[1:dy, 1:dy] \ right_singular_vectors[:, 1:k]
    canonical_coefficients_y *= sqrt(n_rows - 1)
    remaining_rows = size(Y, 2) - size(canonical_coefficients_y, 1)
    if remaining_rows > 0
        canonical_coefficients_y =
            vcat(canonical_coefficients_y, zeros(remaining_rows, k))
    end
    canonical_coefficients_y = canonical_coefficients_y[invperm(qy.p), :]

    canonical_coefficients, canonical_coefficients_y, max_canonical_correlation
end


cca_coeffs(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
) = first(cca_coeffs_and_corr(X, Y, observation_weights))

cca_coeffs_y(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
) = cca_coeffs_and_corr(X, Y, observation_weights)[2]


function correlation(
    X_deflated::AbstractMatrix{<:Real},
    Y_combined::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
)

    correlation(
        centerscale(X_deflated, observation_weights),
        centerscale(Y_combined, observation_weights),
    )
end


function correlation(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    n = size(X, 1)
    X_standard_deviations = sqrt.(mean(X .^ 2, dims = 1))
    zero_std_mask = vec(X_standard_deviations .== 0.0)
    X_standard_deviations[zero_std_mask] .= 1

    col_norms = sqrt.(mean(Y .^ 2, dims = 1))
    X_Y_correlations = (X' * Y) ./ (n * (X_standard_deviations' * col_norms))

    X_standard_deviations[zero_std_mask] .= 0
    X_Y_correlations[zero_std_mask, :, :] .= 0

    X_Y_correlations, X_standard_deviations
end


function compute_variance_weights(
    X_standard_deviations::AbstractMatrix{<:Real},
)::Matrix{Float64}
    mask = X_standard_deviations .== maximum(X_standard_deviations)
    (mask .* X_standard_deviations)'
end


function compute_correlation_weights(X_Y_correlations::AbstractMatrix{<:Real})
    mask = X_Y_correlations .== maximum(X_Y_correlations)
    sum(mask .* X_Y_correlations, dims = 2)
end


function compute_general_weights(
    X_standard_deviations::AbstractMatrix{<:Real},
    X_Y_correlations::AbstractMatrix{<:Real},
    gamma::Real,
    correlation_signs::AbstractMatrix{<:Real})

    transformed_X_standard_deviations = X_standard_deviations .^ ((1 - gamma) / gamma)
    transformed_X_Y_correlations =
        correlation_signs .* abs.(X_Y_correlations) .^ (gamma / (1 - gamma))
    transformed_X_Y_correlations .* transformed_X_standard_deviations'
end


function evaluate_canonical_correlation(
    gamma::Real,
    X_deflated::AbstractMatrix{<:Real},
    X_standard_deviations::AbstractMatrix{<:Real},
    X_Y_correlations::AbstractMatrix{<:Real},
    correlation_signs::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
)

    initial_weights = if gamma == 0
        compute_variance_weights(X_standard_deviations)
    elseif gamma == 1
        compute_correlation_weights(X_Y_correlations)
    else
        compute_general_weights(
            X_standard_deviations,
            X_Y_correlations,
            gamma,
            correlation_signs,
        )
    end

    X_projected = X_deflated * initial_weights
    max_canonical_correlation = cca_corr(X_projected, Y_responses, observation_weights)

    -max_canonical_correlation^2
end


function compute_best_gamma(
    X_deflated::AbstractMatrix{<:Real},
    X_standard_deviations::AbstractMatrix{<:Real},
    X_Y_correlations::AbstractMatrix{<:Real},
    correlation_signs::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma_bounds::NTuple{2,<:Real},
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
)

    a = first(gamma_bounds)
    b = last(gamma_bounds)

    f = gamma -> evaluate_canonical_correlation(
        gamma,
        X_deflated,
        X_standard_deviations,
        X_Y_correlations,
        correlation_signs,
        Y_responses,
        observation_weights,
    )

    # 1) endpoint evaluations
    fa = f(a)
    fb = f(b)

    # 2) interior Brent minimizer
    result = optimize(f, a, b, Brent();
        rel_tol = gamma_rel_tol,
        abs_tol = gamma_abs_tol,
    )
    Optim.converged(result) || @warn("gamma optimization failed to converge.")

    γm = result.minimizer
    fm = result.minimum

    # 3) choose best among {a, γm, b}  (f is negative corr^2, so smaller is better)
    γbest, fbest = a, fa
    if fm < fbest
        γbest, fbest = γm, fm
    end
    if fb < fbest
        γbest, fbest = b, fb
    end

    # return squared canonical correlation (positive)
    canonical_correlation2 = -fbest
    γbest, canonical_correlation2
end


function compute_best_gamma(
    X_deflated::AbstractMatrix{<:Real},
    X_standard_deviations::AbstractMatrix{<:Real},
    X_Y_correlations::AbstractMatrix{<:Real},
    correlation_signs::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma_bounds::AbstractVector{<:Union{NTuple{2,<:Real},Real}},
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
)

    n = length(gamma_bounds)
    gamma_values = zeros(Float64, n)
    corr2_values  = zeros(Float64, n)  # store squared canonical correlations (positive)

    for i = 1:n
        if gamma_bounds[i] isa NTuple{2,<:Real}
            if first(gamma_bounds[i]) ≠ last(gamma_bounds[i])
                gamma_values[i], corr2_values[i] = compute_best_gamma(
                    X_deflated,
                    X_standard_deviations,
                    X_Y_correlations,
                    correlation_signs,
                    Y_responses,
                    observation_weights,
                    gamma_bounds[i],
                    gamma_rel_tol,
                    gamma_abs_tol,
                )
            else
                gamma_values[i] = first(gamma_bounds[i])
                corr2_values[i] = -evaluate_canonical_correlation(
                    gamma_values[i],
                    X_deflated,
                    X_standard_deviations,
                    X_Y_correlations,
                    correlation_signs,
                    Y_responses,
                    observation_weights,
                )
            end
        else
            gamma_values[i] = gamma_bounds[i]
            corr2_values[i] = -evaluate_canonical_correlation(
                gamma_values[i],
                X_deflated,
                X_standard_deviations,
                X_Y_correlations,
                correlation_signs,
                Y_responses,
                observation_weights,
            )
        end
    end

    idx = argmax(corr2_values)
    gamma_values[idx], corr2_values[idx]
end


function compute_best_loadings(
    X_deflated::AbstractMatrix{<:Real},
    X_standard_deviations::AbstractMatrix{<:Real},
    X_Y_correlations::AbstractMatrix{<:Real},
    correlation_signs::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma_bounds::Union{
        <:NTuple{2,<:Real},
        <:AbstractVector{<:Union{<:Real,<:NTuple{2,<:Real}}},
    },
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
    n_targets_Y_combined::Integer,
)

    observation_weights =
        (isnothing(observation_weights) ? observation_weights : sqrt.(observation_weights))

    gamma, canonical_correlation = compute_best_gamma(
        X_deflated,
        X_standard_deviations,
        X_Y_correlations,
        correlation_signs,
        Y_responses,
        observation_weights,
        gamma_bounds,
        gamma_rel_tol,
        gamma_abs_tol,
    )

    if gamma == 0
        base_weights = compute_variance_weights(X_standard_deviations)
        initial_weights = repeat(base_weights, 1, n_targets_Y_combined)
        optimal_loadings = vec(base_weights)
        canonical_coefficients = fill(NaN, (n_targets_Y_combined, 1))
        canonical_coefficients_y = fill(NaN, (size(Y_responses, 2), 1))
    elseif gamma == 1
        base_weights = compute_correlation_weights(X_Y_correlations)
        initial_weights = repeat(base_weights, 1, n_targets_Y_combined)
        optimal_loadings = vec(base_weights)
        canonical_coefficients = fill(NaN, (n_targets_Y_combined, 1))
        canonical_coefficients_y = fill(NaN, (size(Y_responses, 2), 1))
    else
        initial_weights = compute_general_weights(
            X_standard_deviations,
            X_Y_correlations,
            gamma,
            correlation_signs,
        )

        X_projected = X_deflated * initial_weights
        canonical_coefficients, canonical_coefficients_y, _ =
            cca_coeffs_and_corr(X_projected, Y_responses, observation_weights)

        optimal_loadings = vec((initial_weights * canonical_coefficients[:, 1])')
    end

    optimal_loadings,
    canonical_correlation,
    canonical_coefficients[:, 1],
    canonical_coefficients_y[:, 1],
    gamma,
    initial_weights
end


function compute_cppls_weights(
    X_deflated::AbstractMatrix{<:Real},
    Y_combined::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma::Real,
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
)

    if gamma == 0.5

        initial_weights = X_deflated' * Y_combined
        canonical_coefficients, canonical_coefficients_y, max_canonical_correlation =
            cca_coeffs_and_corr(
            X_deflated * initial_weights,
            Y_responses,
            observation_weights,
        )

        optimal_loadings = initial_weights * canonical_coefficients[:, 1]

        return optimal_loadings,
        max_canonical_correlation^2,
        canonical_coefficients[:, 1],
        canonical_coefficients_y[:, 1],
        0.5,
        initial_weights
    else
        return compute_cppls_weights(
            X_deflated,
            Y_combined,
            Y_responses,
            observation_weights,
            (gamma, gamma),
            gamma_rel_tol,
            gamma_abs_tol,
        )
    end
end


function compute_cppls_weights(
    X_deflated::AbstractMatrix{<:Real},
    Y_combined::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma::Union{<:NTuple{2,<:Real},<:AbstractVector{<:Union{<:Real,<:NTuple{2,<:Real}}}},
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
)

    X_Y_correlations, X_standard_deviations =
        correlation(X_deflated, Y_combined, observation_weights)

    correlation_signs = sign.(X_Y_correlations)
    X_Y_correlations = abs.(X_Y_correlations) ./ maximum(X_Y_correlations)
    X_standard_deviations /= maximum(X_standard_deviations)

    compute_best_loadings(
        X_deflated,
        X_standard_deviations,
        X_Y_correlations,
        correlation_signs,
        Y_responses,
        observation_weights,
        gamma,
        gamma_rel_tol,
        gamma_abs_tol,
        size(Y_combined, 2),
    )
end
