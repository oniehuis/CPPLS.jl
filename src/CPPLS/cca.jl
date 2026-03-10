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
    rho = clamp(first(singular_vals), 0.0, 1.0)

    n_rows,
    n_cols,
    qx,
    qy,
    dx,
    dy,
    left_singular_vecs,
    right_singular_vecs,
    rho
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
            rho,
        ) = cca_decomposition(X, Y, observation_weights)
    )

    k = min(dx, dy)
    a = qx.R[1:dx, 1:dx] \ left_singular_vectors[:, 1:k]
    a *= sqrt(n_rows - 1)

    remaining_rows = n_cols - size(a, 1)
    if remaining_rows > 0
        a =
            vcat(a, zeros(remaining_rows, k))
    end

    a = a[invperm(qx.p), :]

    b = qy.R[1:dy, 1:dy] \ right_singular_vectors[:, 1:k]
    b *= sqrt(n_rows - 1)
    remaining_rows = size(Y, 2) - size(b, 1)
    if remaining_rows > 0
        b =
            vcat(b, zeros(remaining_rows, k))
    end
    b = b[invperm(qy.p), :]

    a, b, rho
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
    X_def::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
)

    correlation(
        centerscale(X_def, observation_weights),
        centerscale(Y, observation_weights),
    )
end


function correlation(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    n = size(X, 1)
    S_x = sqrt.(mean(X .^ 2, dims = 1))
    zero_std_mask = vec(S_x .== 0.0)
    S_x[zero_std_mask] .= 1

    S_y = sqrt.(mean(Y .^ 2, dims = 1))
    zero_norm_mask = vec(S_y .== 0.0)
    S_y[zero_norm_mask] .= 1
    C = (X' * Y) ./ (n * (S_x' * S_y))

    S_x[zero_std_mask] .= 0
    C[zero_std_mask, :, :] .= 0
    C[:, zero_norm_mask, :] .= 0

    C, S_x
end


function compute_variance_weights(
    S_x::AbstractMatrix{<:Real},
)::Matrix{Float64}
    mask = S_x .== maximum(S_x)
    (mask .* S_x)'
end


function compute_correlation_weights(C::AbstractMatrix{<:Real})
    mask = C .== maximum(C)
    sum(mask .* C, dims = 2)
end


function compute_general_weights(
    S_x::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    gamma::Real,
    C_sign::AbstractMatrix{<:Real})

    S_x_pow = S_x .^ ((1 - gamma) / gamma)
    C_gamma =
        C_sign .* abs.(C) .^ (gamma / (1 - gamma))
    C_gamma .* S_x_pow'
end


function evaluate_canonical_correlation(
    gamma::Real,
    X_def::AbstractMatrix{<:Real},
    S_x::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    C_sign::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
)

    W0 = if gamma == 0
        compute_variance_weights(S_x)
    elseif gamma == 1
        compute_correlation_weights(C)
    else
        compute_general_weights(
            S_x,
            C,
            gamma,
            C_sign,
        )
    end

    Z = X_def * W0
    rho = cca_corr(Z, Y_prim, observation_weights)

    -rho^2
end


function compute_best_gamma(
    X_def::AbstractMatrix{<:Real},
    S_x::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    C_sign::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma_bounds::NTuple{2,<:Real},
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
)

    a = first(gamma_bounds)
    b = last(gamma_bounds)

    f = gamma -> evaluate_canonical_correlation(
        gamma,
        X_def,
        S_x,
        C,
        C_sign,
        Y_prim,
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
    rho2 = -fbest
    γbest, rho2
end


function compute_best_gamma(
    X_def::AbstractMatrix{<:Real},
    S_x::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    C_sign::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma_bounds::AbstractVector{<:Union{NTuple{2,<:Real},Real}},
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
)

    n = length(gamma_bounds)
    gamma_vals = zeros(Float64, n)
    rho2_vals  = zeros(Float64, n)  # store squared canonical correlations (positive)

    for i = 1:n
        if gamma_bounds[i] isa NTuple{2,<:Real}
            if first(gamma_bounds[i]) ≠ last(gamma_bounds[i])
                gamma_vals[i], rho2_vals[i] = compute_best_gamma(
                    X_def,
                    S_x,
                    C,
                    C_sign,
                    Y_prim,
                    observation_weights,
                    gamma_bounds[i],
                    gamma_rel_tol,
                    gamma_abs_tol,
                )
            else
                gamma_vals[i] = first(gamma_bounds[i])
                rho2_vals[i] = -evaluate_canonical_correlation(
                    gamma_vals[i],
                    X_def,
                    S_x,
                    C,
                    C_sign,
                    Y_prim,
                    observation_weights,
                )
            end
        else
            gamma_vals[i] = gamma_bounds[i]
            rho2_vals[i] = -evaluate_canonical_correlation(
                gamma_vals[i],
                X_def,
                S_x,
                C,
                C_sign,
                Y_prim,
                observation_weights,
            )
        end
    end

    idx = argmax(rho2_vals)
    gamma_vals[idx], rho2_vals[idx]
end


function compute_best_loadings(
    X_def::AbstractMatrix{<:Real},
    S_x::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    C_sign::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma_bounds::Union{
        <:NTuple{2,<:Real},
        <:AbstractVector{<:Union{<:Real,<:NTuple{2,<:Real}}},
    },
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
    q::Integer,
)

    observation_weights =
        (isnothing(observation_weights) ? observation_weights : sqrt.(observation_weights))

    gamma, rho2 = compute_best_gamma(
        X_def,
        S_x,
        C,
        C_sign,
        Y_prim,
        observation_weights,
        gamma_bounds,
        gamma_rel_tol,
        gamma_abs_tol,
    )

    if gamma == 0
        w_base = compute_variance_weights(S_x)
        W0 = repeat(w_base, 1, q)
        w = vec(w_base)
        a = fill(NaN, (q, 1))
        b = fill(NaN, (size(Y_prim, 2), 1))
    elseif gamma == 1
        w_base = compute_correlation_weights(C)
        W0 = repeat(w_base, 1, q)
        w = vec(w_base)
        a = fill(NaN, (q, 1))
        b = fill(NaN, (size(Y_prim, 2), 1))
    else
        W0 = compute_general_weights(
            S_x,
            C,
            gamma,
            C_sign,
        )

        Z = X_def * W0
        a, b, _ = cca_coeffs_and_corr(Z, Y_prim, observation_weights)

        w = vec((W0 * a[:, 1])')
    end

    w,
    rho2,
    a[:, 1],
    b[:, 1],
    gamma,
    W0
end


function compute_cppls_weights(
    X_def::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma::Real,
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
)

    if gamma == 0.5

        W0 = X_def' * Y
        a, b, rho =
            cca_coeffs_and_corr(
            X_def * W0,
            Y_prim,
            observation_weights,
        )

        w = W0 * a[:, 1]

        return w,
        rho^2,
        a[:, 1],
        b[:, 1],
        0.5,
        W0
    else
        return compute_cppls_weights(
            X_def,
            Y,
            Y_prim,
            observation_weights,
            (gamma, gamma),
            gamma_rel_tol,
            gamma_abs_tol,
        )
    end
end


function compute_cppls_weights(
    X_def::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    gamma::Union{<:NTuple{2,<:Real},<:AbstractVector{<:Union{<:Real,<:NTuple{2,<:Real}}}},
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
)

    C, S_x =
        correlation(X_def, Y, observation_weights)

    max_corr = maximum(C)
    max_std = maximum(S_x)

    C_sign = sign.(C)
    if max_corr > 0
        C = abs.(C) ./ max_corr
    else
        C .= 0
    end
    if max_std > 0
        S_x ./= max_std
    else
        S_x .= 0
    end

    compute_best_loadings(
        X_def,
        S_x,
        C,
        C_sign,
        Y_prim,
        observation_weights,
        gamma,
        gamma_rel_tol,
        gamma_abs_tol,
        size(Y, 2),
    )
end
