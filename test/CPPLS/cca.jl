using LinearAlgebra: rank

@testset "Iᵣ builds rectangular identity" begin
    ident = CPPLS.Iᵣ(3, 5)
    @test size(ident) == (3, 5)
    @test ident[1, 1] == 1.0 && ident[1, 4] == 0.0
    @test ident[3, 5] == 0.0
end

@testset "cca_decomposition validates ranks and weights" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    Y = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    rows, cols, _, _, dx, dy, _, _, corr = CPPLS.cca_decomposition(X, Y)
    @test rows == size(X, 1)
    @test dx == 2
    @test dy == 2
    @test 0.0 ≤ corr ≤ 1.0

    w = [1.0, 2.0, 1.0]
    stats = CPPLS.cca_decomposition(X, Y, w)
    @test stats[1] == rows

    @test_throws ErrorException CPPLS.cca_decomposition(zeros(3, 2), Y)
    @test_throws ErrorException CPPLS.cca_decomposition(X, zeros(3, 2))
end

@testset "correlation rescales and zeroes out constant columns" begin
    X = [1.0 2.0 2.0; 3.0 4.0 4.0]
    Y = [1.0 0.0; 0.0 1.0]
    corr, stds = CPPLS.correlation(X, Y)
    @test size(corr) == (3, 2)
    @test size(stds) == (1, 3)
end

@testset "cca_coeffs helpers return expected matrices" begin
    X = [
        1.0 2.0
        3.0 4.0
        5.0 6.0
        7.0 8.0
    ]
    Y = [
        1.0 0.0
        0.0 1.0
        1.0 1.0
        0.5 0.5
    ]

    coeffs, coeffs_y, _ = CPPLS.cca_coeffs_and_corr(X, Y, nothing)
    @test CPPLS.cca_coeffs(X, Y, nothing) ≈ coeffs
    @test CPPLS.cca_coeffs_y(X, Y, nothing) ≈ coeffs_y

    weights = [1.0, 2.0, 1.0, 0.5]
    coeffs_w, coeffs_y_w, _ = CPPLS.cca_coeffs_and_corr(X, Y, weights)
    @test CPPLS.cca_coeffs(X, Y, weights) ≈ coeffs_w
    @test CPPLS.cca_coeffs_y(X, Y, weights) ≈ coeffs_y_w
    @test size(coeffs_w, 1) == size(X, 2)
    @test size(coeffs_y_w, 1) == size(Y, 2)
end

@testset "cca_coeffs_y pads zero rows for rank-deficient Y" begin
    X = [
        1.0 2.0
        3.0 4.0
        5.0 6.0
        7.0 8.0
    ]
    y_col = [1.0, 2.0, 3.0, 4.0]
    Y = hcat(y_col, 2 .* y_col, -1 .* y_col)

    _, coeffs_y, _ = CPPLS.cca_coeffs_and_corr(X, Y, nothing)
    dy = rank(Y)
    remaining_rows = size(Y, 2) - dy
    zero_rows = count(eachrow(coeffs_y)) do row
        all(iszero, row)
    end
    @test remaining_rows > 0
    @test size(coeffs_y, 1) == size(Y, 2)
    @test zero_rows == remaining_rows
end

@testset "compute_cppls_weights handles gamma shortcuts" begin
    X = rand(5, 3)
    Y = rand(5, 2)
    Y_combined = hcat(Y, rand(5, 1))

    weights = CPPLS.compute_cppls_weights(
        X,
        Y_combined,
        Y,
        nothing,
        0.5,
        1e-6,
        1e-12,
    )
    @test length(weights) == 6

    weights_bounds = CPPLS.compute_cppls_weights(
        X,
        Y_combined,
        Y,
        nothing,
        (0.1, 0.9),
        1e-6,
        1e-12,
    )
    @test length(weights_bounds) == 6

    scalar_gamma = CPPLS.compute_cppls_weights(
        X,
        Y_combined,
        Y,
        nothing,
        0.3,
        1e-6,
        1e-12,
    )
    tuple_gamma = CPPLS.compute_cppls_weights(
        X,
        Y_combined,
        Y,
        nothing,
        (0.3, 0.3),
        1e-6,
        1e-12,
    )
    @test all(isapprox.(scalar_gamma[1], tuple_gamma[1]))
    @test scalar_gamma[2] ≈ tuple_gamma[2]
    @test all(isapprox.(scalar_gamma[3], tuple_gamma[3]))
    @test scalar_gamma[5] ≈ tuple_gamma[5]

    loadings_one, corr_one, coeffs_one, coeffs_one_y, gamma_one, _ =
        CPPLS.compute_cppls_weights(
            X,
            Y_combined,
            Y,
            nothing,
            (1.0, 1.0),
            1e-6,
            1e-12,
        )
    @test gamma_one == 1.0
    @test all(isfinite.(loadings_one))
    @test all(isnan.(coeffs_one))
    @test all(isnan.(coeffs_one_y))
end

@testset "weight helpers and gamma search utilities" begin
    σ = reshape([1.0, 2.0, 1.5], 1, :)
    variance_weights = CPPLS.compute_variance_weights(σ)
    @test size(variance_weights) == (3, 1)
    @test variance_weights[2] == 2.0
    @test variance_weights[1] == 0.0

    correlations = [
        0.1 0.3
        0.9 0.2
        0.4 0.9
    ]
    corr_weights = CPPLS.compute_correlation_weights(correlations)
    @test corr_weights[2] == maximum(correlations)
    @test corr_weights[1] == 0.0

    signs = ones(size(correlations))
    general_weights = CPPLS.compute_general_weights(
        ones(size(σ)),
        abs.(correlations),
        0.5,
        signs,
    )
    @test size(general_weights) == size(correlations)
    @test all(isfinite.(general_weights))

    X_deflated = [
        1.0 0.5
        0.0 1.0
        1.5 1.0
        2.0 0.2
    ]
    Y_responses = [
        1 0
        0 1
        1 0
        0 1
    ]
    X_Y_corr, X_std = CPPLS.correlation(X_deflated, Y_responses)
    corr_signs = sign.(X_Y_corr)
    X_Y_corr = abs.(X_Y_corr) ./ maximum(abs.(X_Y_corr))
    X_std ./= maximum(X_std)

    val0 = CPPLS.evaluate_canonical_correlation(
        0.0,
        X_deflated,
        X_std,
        X_Y_corr,
        corr_signs,
        Y_responses,
        nothing,
    )
    val1 = CPPLS.evaluate_canonical_correlation(
        1.0,
        X_deflated,
        X_std,
        X_Y_corr,
        corr_signs,
        Y_responses,
        nothing,
    )
    @test val0 ≤ 0
    @test val1 ≤ 0

    γ_tuple, corr_tuple = CPPLS.compute_best_gamma(
        X_deflated,
        X_std,
        X_Y_corr,
        corr_signs,
        Y_responses,
        nothing,
        (0.0, 1.0),
        1e-6,
        1e-12,
    )
    @test 0.0 ≤ γ_tuple ≤ 1.0
    @test 0.0 ≤ corr_tuple ≤ 1.0

    gamma_choices = Union{Float64,NTuple{2,Float64}}[0.0, 0.5, (0.2, 0.8), (0.3, 0.3)]
    γ_vec, corr_vec = CPPLS.compute_best_gamma(
        X_deflated,
        X_std,
        X_Y_corr,
        corr_signs,
        Y_responses,
        nothing,
        gamma_choices,
        1e-6,
        1e-12,
    )
    @test 0.0 ≤ γ_vec ≤ 1.0
    @test 0.0 ≤ corr_vec ≤ 1.0

    loadings_zero, corr_zero, coeffs_zero, coeffs_zero_y, γ_zero, _ =
        CPPLS.compute_best_loadings(
            X_deflated,
            X_std,
            X_Y_corr,
            corr_signs,
            Y_responses,
            nothing,
            (0.0, 0.0),
            1e-6,
            1e-12,
            size(Y_responses, 2),
        )
    @test γ_zero == 0.0
    @test all(isnan, coeffs_zero)
    @test all(isnan, coeffs_zero_y)
    @test size(loadings_zero) == (size(X_deflated, 2),)
    @test 0.0 ≤ corr_zero ≤ 1.0

    loadings_general, corr_general, coeffs_general, coeffs_general_y, γ_general, _ =
        CPPLS.compute_best_loadings(
            X_deflated,
            X_std,
            X_Y_corr,
            corr_signs,
            Y_responses,
            nothing,
            (0.2, 0.8),
            1e-6,
            1e-12,
            size(Y_responses, 2),
        )
    @test 0.2 ≤ γ_general ≤ 0.8
    @test all(isfinite.(coeffs_general))
    @test all(isfinite.(coeffs_general_y))
    @test length(loadings_general) == size(X_deflated, 2)
    @test 0.0 ≤ corr_general ≤ 1.0
end
