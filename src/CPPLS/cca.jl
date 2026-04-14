"""
    compute_cppls_weights(
        m::CPPLSModel,
        X_def::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}, 
        Yprim::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}, 
        gamma::Real
    )
    compute_cppls_weights(
        m::CPPLSModel,
        X_def::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}, 
        Yprim::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}, 
        gamma::Union{
            <:NTuple{2, <:Real}, 
            <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
        }
    )

Compute CPPLS supervised weights and the associated CCA quantities. This is the entry
point that follows the CPPLS pipeline of building W0(gamma), projecting X to Z, and
aligning Z with the primary responses through CCA. A fixed gamma can be supplied directly,
or a search/grid specification can be used to select gamma by maximizing the leading
canonical correlation.

Type stablity tested: 03/25/2026
"""
function compute_cppls_weights(
    m::CPPLSModel,
    X_def::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
    gamma::Real
)

    if gamma == 0.5
        # Special-case gamma = 0.5 uses the X'Y shortcut used in the CPPLS formulation.
        W0 = X_def' * Y
        a, b, rho = cca_coeffs_and_corr(X_def * W0, Yprim, obs_weights)
        w = vec(W0 * a[:, 1])
        return w, rho^2, a[:, 1], b[:, 1], 0.5, W0, [0.5], [rho^2]
    else
        return compute_cppls_weights(m, X_def, Y, Yprim, obs_weights, (gamma, gamma))
    end
end

function compute_cppls_weights(
    m::CPPLSModel,
    X_def::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
    gamma::Union{<:NTuple{2, <:Real},
                 <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
    }
)

    # Correlation and scale statistics form the ingredients of W0(gamma).
    C, S_x = correlation(X_def, Y, obs_weights)

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

    compute_best_loadings(m, X_def, S_x, C, C_sign, Yprim, obs_weights, gamma,
        size(Y, 2))
end

# Type stablity tested: 03/25/2026

"""
    gamma_search_candidate_count(gamma)

Return the number of candidate gamma values for CPPLS grid or search. If `gamma` is a 
scalar `Real` or a 2-tuple of `Real`, returns 1 (single candidate). If `gamma` is a vector 
of `Real` and/or 2-tuples, returns the vector length (number of candidates).

Type stability tested: 03/25/2026
"""
gamma_search_candidate_count(::Real) = 1
gamma_search_candidate_count(::NTuple{2, <:Real}) = 1
gamma_search_candidate_count(
    gamma::AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
) = length(gamma)

"""
    compute_best_loadings(
        m::CPPLSModel,
        X_def::AbstractMatrix{<:Real}, 
        S_x::AbstractMatrix{<:Real}, 
        C::AbstractMatrix{<:Real}, 
        C_sign::AbstractMatrix{<:Real}, 
        Yprim::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}, 
        gamma_bounds::Union{
            <:NTuple{2, <:Real}, 
            <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
        },
        q::Integer
    )

Compute the supervised weight vector w and related CCA quantities after selecting the
best gamma, mirroring the CPPLS step where W0(gamma) defines Z and CCA yields the
canonical direction used to form the component. The return values include w, the squared
canonical correlation, canonical coefficient vectors, the selected gamma, and W0. q is the 
total number of response columns (primary + auxiliary) used to construct the supervised 
projection space.

Type stablity tested: 03/25/2026
"""
function compute_best_loadings(
    m::CPPLSModel,
    X_def::AbstractMatrix{<:Real},
    S_x::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    C_sign::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
    gamma_bounds::Union{
        <:NTuple{2, <:Real},
        <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}},
    },
    q::Integer
)

    # Apply observation weights consistently with covariance weighting (sqrt for covariance).
    obs_weights = isnothing(obs_weights) ? obs_weights : sqrt.(obs_weights)

    gamma, rho2, gammas, rhos = compute_best_gamma(
        m,
        X_def,
        S_x,
        C,
        C_sign,
        Yprim,
        obs_weights,
        gamma_bounds
    )

    if gamma == 0
        w_base = compute_variance_weights(S_x)
        W0 = repeat(w_base, 1, q)
        w = vec(w_base)
        a = fill(NaN, (q, 1))
        b = fill(NaN, (size(Yprim, 2), 1))
    elseif gamma == 1
        w_base = compute_correlation_weights(C)
        W0 = repeat(w_base, 1, q)
        w = vec(w_base)
        a = fill(NaN, (q, 1))
        b = fill(NaN, (size(Yprim, 2), 1))
    else
        # General case uses the power-law W0(gamma) and a full CCA.
        W0 = compute_general_weights(S_x, C, gamma, C_sign)

        Z = X_def * W0
        a, b, _ = cca_coeffs_and_corr(Z, Yprim, obs_weights)

        w = vec((W0 * a[:, 1])')
    end

    w, rho2, a[:, 1], b[:, 1], gamma, W0, gammas, rhos
end

"""
    compute_best_gamma(
        m::CPPLSModel,
        X_def::AbstractMatrix{<:Real}, 
        S_x::AbstractMatrix{<:Real}, 
        C::AbstractMatrix{<:Real}, 
        C_sign::AbstractMatrix{<:Real}, 
        Yprim::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}, 
        gamma_bounds::NTuple{2, <:Real}, 
        gamma_rel_tol::Real, 
        gamma_abs_tol::Real
    )
    compute_best_gamma(
        m::CPPLSModel,
        X_def::AbstractMatrix{<:Real}, 
        S_x::AbstractMatrix{<:Real}, 
        C::AbstractMatrix{<:Real}, 
        C_sign::AbstractMatrix{<:Real}, 
        Yprim::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}, 
        gamma_bounds::AbstractVector{<:Union{NTuple{2, <:Real}, Real}}, 
        gamma_rel_tol::Real, 
        gamma_abs_tol::Real
    )

Select the gamma that maximizes the leading canonical correlation between Z = X W0(gamma)
and Yprim. The tuple form uses a bounded Brent search, while the vector form evaluates a
set of candidate values or bounds and returns the best.

Type stablity tested: 03/25/2026
"""
function compute_best_gamma(
    m::CPPLSModel,
    X_def::AbstractMatrix{<:Real},
    S_x::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    C_sign::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
    gamma_bounds::NTuple{2, <:Real}
)

    a = first(gamma_bounds)
    b = last(gamma_bounds)

    if a == b
        rho2 = -evaluate_canonical_correlation(a, X_def, S_x, C, C_sign, Yprim,
            obs_weights)
        return a, rho2, [Float64(a)], [rho2]
    end

    f = gamma -> try
        evaluate_canonical_correlation(gamma, X_def, S_x, C, C_sign, Yprim,
            obs_weights)
    catch error
        if error isa ErrorException && (error.msg == "X has rank 0" || error.msg == "Y has rank 0")
            0.0
        else
            rethrow()
        end
    end

    # Evaluate endpoints and then a Brent minimizer over the interval.
    fa = f(a)
    fb = f(b)

    result = optimize(f, a, b, Brent(); rel_tol=m.gamma_rel_tol, abs_tol=m.gamma_abs_tol)
    Optim.converged(result) || @warn("gamma optimization failed to converge.")

    γm = result.minimizer
    fm = result.minimum

    # Choose the best among the endpoints and the interior minimizer.
    γbest, fbest = a, fa
    if fm < fbest
        γbest, fbest = γm, fm
    end
    if fb < fbest
        γbest, fbest = b, fb
    end

    # Return squared canonical correlation (positive).
    rho2 = -fbest
    γbest, rho2, [Float64(γbest)], [rho2]
end

function compute_best_gamma(
    m::CPPLSModel,
    X_def::AbstractMatrix{<:Real},
    S_x::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    C_sign::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
    gamma_bounds::AbstractVector{<:Union{NTuple{2, <:Real}, Real}}
)

    n = length(gamma_bounds)
    gamma_vals = zeros(Float64, n)
    rho2_vals  = zeros(Float64, n)  # store squared canonical correlations (positive)

    # Evaluate each candidate gamma or gamma interval and keep the best score.
    for i = 1:n
        if gamma_bounds[i] isa NTuple{2,<:Real}
            if first(gamma_bounds[i]) ≠ last(gamma_bounds[i])
                gamma_vals[i], rho2_vals[i] = compute_best_gamma(m, X_def, S_x, C, C_sign,
                    Yprim, obs_weights, gamma_bounds[i])
            else
                gamma_vals[i] = first(gamma_bounds[i])
                rho2_vals[i] = -evaluate_canonical_correlation(gamma_vals[i], X_def, S_x,
                    C, C_sign, Yprim, obs_weights)
            end
        else
            gamma_vals[i] = gamma_bounds[i]
            rho2_vals[i] = -evaluate_canonical_correlation(gamma_vals[i], X_def, S_x, C,
                C_sign, Yprim, obs_weights)
        end
    end

    idx = argmax(rho2_vals)
    gamma_vals[idx], rho2_vals[idx], gamma_vals, rho2_vals
end

"""
    evaluate_canonical_correlation(
        gamma::Real, X_def::AbstractMatrix{<:Real}, 
        S_x::AbstractMatrix{<:Real}, 
        C::AbstractMatrix{<:Real}, 
        C_sign::AbstractMatrix{<:Real}, 
        Yprim::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Evaluate the negative squared leading canonical correlation for a given gamma. This
corresponds to constructing Z = X W0(gamma) and measuring how well Z aligns with the
primary responses under CCA. The sign is flipped so that scalar optimizers can minimize
the value.

Type stablity tested: 03/25/2026
"""
function evaluate_canonical_correlation(
    gamma::Real,
    X_def::AbstractMatrix{<:Real},
    S_x::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    C_sign::AbstractMatrix{<:Real},
    Yprim::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
)

    # Build the supervised weights W0(gamma) from the scale-correlation tradeoff.
    W0 = if gamma == 0
        compute_variance_weights(S_x)
    elseif gamma == 1
        compute_correlation_weights(C)
    else
        compute_general_weights(S_x, C, gamma, C_sign)
    end

    Z = X_def * W0
    rho = cca_corr(Z, Yprim, obs_weights)

    -rho^2
end

"""
    centerweight(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real})
    centerweight(M::AbstractMatrix{<:Real}, ::Nothing)

Center `M` and apply observation weights in a single step. With weights, each row is
centered by the weighted mean and then scaled by the weights. Without weights, only
centering is performed.

Type stablity tested: 03/25/2026
"""
centerweight(M::AbstractMatrix{<:Real}, obs_weights::AbstractVector{<:Real}) =
    (M .- (obs_weights' * M) / sum(obs_weights)) .* obs_weights

centerweight(M::AbstractMatrix{<:Real}, ::Nothing) = M .- mean(M, dims=1)

"""
    correlation(
        X_def::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}
    )
    correlation(
        X_def::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Compute the weighted predictor-response correlation matrix C and weighted predictor scales
S_x used to build W0(gamma). The weighted centering and scaling follow the same
sample-weighting logic used throughout CPPLS.

Type stablity tested: 03/25/2026
"""
@inline function correlation(
    X_def::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
)

    correlation(centerweight(X_def, obs_weights), centerweight(Y, obs_weights))
end

function correlation(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    n = size(X, 1)

    S_x = sqrt.(mean(X .^ 2, dims=1))
    zero_std_mask = vec(S_x .== 0.0)
    S_x[zero_std_mask] .= 1

    S_y = sqrt.(mean(Y .^ 2, dims=1))
    zero_norm_mask = vec(S_y .== 0.0)
    S_y[zero_norm_mask] .= 1

    # Weighted correlation matrix between predictors and responses.
    C = (X' * Y) ./ (n * (S_x' * S_y))

    # Reset degenerate columns to zero to avoid propagating NaNs.
    S_x[zero_std_mask] .= 0
    C[zero_std_mask, :] .= 0
    C[:, zero_norm_mask] .= 0

    C, S_x
end

"""
    compute_general_weights(
        S::AbstractMatrix{<:Real}, 
        C::AbstractMatrix{<:Real}, 
        gamma::Real, 
        C_sign::AbstractMatrix{<:Real}
    )

Compute the supervised weight matrix W0(gamma) for 0 < gamma < 1 using the power-law
combination of weighted predictor scales and signed correlations.

Type stablity tested: 03/25/2026
"""
@inline function compute_general_weights(
    S::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    gamma::Real,
    C_sign::AbstractMatrix{<:Real}
)

    # Elementwise power construction of W0(gamma) from scales and correlations.
    S_pow = S .^ ((1 - gamma) / gamma)
    C_gamma = C_sign .* abs.(C) .^ (gamma / (1 - gamma))
    C_gamma .* S_pow'
end

"""
    compute_correlation_weights(C::AbstractMatrix{<:Real})

Construct the extreme gamma = 1 supervised weights that select predictors by maximal
weighted correlation, matching the correlation-dominant limit.

Type stablity tested: 03/25/2026
"""
@inline function compute_correlation_weights(C::AbstractMatrix{<:Real})
    mask = C .== maximum(C)
    sum(mask .* C, dims=2)
end

"""
    compute_variance_weights(S::AbstractMatrix{<:Real})::Matrix{Float64}

Construct the extreme gamma = 0 supervised weights that select predictors by maximal
weighted variance, matching the limiting case of the power formulation.

Type stablity tested: 03/25/2026
"""
@inline function compute_variance_weights(S::AbstractMatrix{<:Real})::Matrix{Float64}
    mask = S .== maximum(S)
    (mask .* S)'
end

"""
    cca_decomposition(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}
    )
    cca_decomposition(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}, 
        obs_weights::AbstractVector{<:Real}
    )
    cca_decomposition(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}, 
        ::Nothing
    )

Compute the weighted canonical correlation decomposition used by CPPLS to align the
supervised projection Z with the primary responses. The method orthonormalizes X and Y
with QR, forms the cross-covariance in the orthonormal bases, and extracts canonical
correlations and vectors via SVD.

In CPPLS, this corresponds to the CCA stage that finds directions a and b maximizing the
correlation between Z a and Yprim b, after Z has been built as X W0(gamma). Observation
weights are incorporated by scaling rows prior to the decomposition.

Type stablity tested: 03/25/2026
"""
@inline function cca_decomposition(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    ::Nothing)

    cca_decomposition(X, Y)
end

@inline function cca_decomposition(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    obs_weights::AbstractVector{<:Real}
)

    cca_decomposition(X .* obs_weights, Y .* obs_weights)
end

function cca_decomposition(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    n_rows, n_cols = size(X)

    # Orthonormal bases for the column spaces of X and Y.
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

    # Cross-covariance in the orthonormal bases; SVD yields canonical correlations.
    A = ((qx.Q' * qy.Q) * Iᵣ(n_rows, dy))[1:dx, :]
    left_singular_vecs, singular_vals, right_singular_vecs_t = svd(A; full=true)
    right_singular_vecs = right_singular_vecs_t'
    rho = clamp(first(singular_vals), 0.0, 1.0)

    n_rows, n_cols, qx, qy, dx, dy, left_singular_vecs, right_singular_vecs, rho
end

"""
    cca_corr(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Return the leading canonical correlation between X and Y under the same weighting scheme
used in CPPLS. This value is used as the score when comparing gamma values during
selection.

Type stablity tested: 03/25/2026
"""
@inline function cca_corr(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)

    last(cca_decomposition(X, Y, obs_weights))
end

"""
    cca_coeffs_and_corr(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Compute canonical coefficient matrices a and b together with the leading canonical
correlation. These coefficients map from the orthonormal CCA bases back into the original
predictor and response spaces, matching the CPPLS step where supervised directions are
combined into a single latent axis aligned with the primary responses.

Type stablity tested: 03/25/2026
"""
function cca_coeffs_and_corr(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)

    n_rows, n_cols, qx, qy, dx, dy, left_singular_vectors, right_singular_vectors, rho =
        cca_decomposition(X, Y, obs_weights)

    k = min(dx, dy)

    # Back-transform CCA vectors into the original X space.
    a = qx.R[1:dx, 1:dx] \ left_singular_vectors[:, 1:k]
    a *= sqrt(n_rows - 1)
    remaining_rows = n_cols - size(a, 1)
    if remaining_rows > 0
        a = vcat(a, zeros(remaining_rows, k))
    end
    a = a[invperm(qx.p), :]

    # Back-transform CCA vectors into the original Y space.
    b = qy.R[1:dy, 1:dy] \ right_singular_vectors[:, 1:k]
    b *= sqrt(n_rows - 1)
    remaining_rows = size(Y, 2) - size(b, 1)
    if remaining_rows > 0
        b = vcat(b, zeros(remaining_rows, k))
    end
    b = b[invperm(qy.p), :]

    a, b, rho
end

"""
    cca_coeffs(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Return the canonical coefficient matrix for predictors only. In CPPLS this corresponds to
the direction a that combines the supervised axes of Z = X W0 into a single component.

Type stablity tested: 03/25/2026
"""
@inline function cca_coeffs(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)

    cca_coeffs_and_corr(X, Y, obs_weights)[1]
end

"""
    cca_coeffs_y(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real}, 
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Return the canonical coefficient matrix for the responses. In CPPLS this is the direction
b that maximizes correlation with the supervised predictor component.

Type stablity tested: 03/25/2026
"""
@inline function cca_coeffs_y(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)

    cca_coeffs_and_corr(X, Y, obs_weights)[2]
end

"""
    Iᵣ(rowcount::Integer, columncount::Integer)

Construct a rectangular identity matrix with ones on the diagonal and zeros elsewhere.
This helper is used in the CCA step of CPPLS to align QR bases so the SVD acts on the
overlap between the column spaces.

Type stablity tested: 03/25/2026
"""
@inline function Iᵣ(rowcount::Integer, columncount::Integer)
    M = zeros(rowcount, columncount)
    @inbounds for i = 1:min(rowcount, columncount)
        M[i, i] = 1
    end
    M
end
