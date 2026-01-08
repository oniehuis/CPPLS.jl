@testset "center_mean handles weighted and unweighted inputs" begin
    M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    weights = [1.0, 2.0, 1.0]

    centered_weighted, mean_weighted = CPPLS.center_mean(M, weights)
    centered_unweighted, mean_unweighted = CPPLS.center_mean(M, nothing)

    expected_weighted_mean = (weights' * M) / sum(weights)
    expected_unweighted_mean = CPPLS.mean(M, dims = 1)

    @test mean_weighted == expected_weighted_mean
    @test all(isapprox.(centered_weighted, M .- expected_weighted_mean))
    @test mean_unweighted == expected_unweighted_mean
    @test all(isapprox.(centered_unweighted, M .- expected_unweighted_mean))
end

@testset "centerscale applies weighted centering and scaling" begin
    M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    weights = [1.0, 2.0, 1.0]

    weighted_cs = CPPLS.centerscale(M, weights)
    unweighted_cs = CPPLS.centerscale(M, nothing)

    expected_weighted = (M .- (weights' * M) / sum(weights)) .* weights
    expected_unweighted = M .- CPPLS.mean(M, dims = 1)

    @test weighted_cs == expected_weighted
    @test unweighted_cs == expected_unweighted
end

@testset "convert_to_float64 preserves Float64 and converts other types" begin
    float_input = rand(3, 3)
    @test CPPLS.convert_to_float64(float_input) === float_input

    int_input = [1 2; 3 4]
    converted = CPPLS.convert_to_float64(int_input)
    @test converted isa Matrix{Float64}
    @test converted == float.(int_input)
end

@testset "convert_to_float64 handles vectors" begin
    float_vec = rand(5)
    @test CPPLS.convert_to_float64(float_vec) === float_vec

    int_vec = [1, 2, 3]
    converted_vec = CPPLS.convert_to_float64(int_vec)
    @test converted_vec isa Vector{Float64}
    @test converted_vec == float.(int_vec)
end

@testset "convert_auxiliary_to_float64 converts matrices and vectors" begin
    aux_matrix = [1 2; 3 4]
    converted_matrix = CPPLS.convert_auxiliary_to_float64(aux_matrix)
    @test converted_matrix isa Matrix{Float64}
    @test converted_matrix == float.(aux_matrix)

    aux_vector = [1, 3, 5]
    converted_vector = CPPLS.convert_auxiliary_to_float64(aux_vector)
    @test converted_vector isa Vector{Float64}
    @test converted_vector == float.(aux_vector)

    float_matrix = rand(2, 2)
    @test CPPLS.convert_auxiliary_to_float64(float_matrix) === float_matrix

    float_vector = rand(4)
    @test CPPLS.convert_auxiliary_to_float64(float_vector) === float_vector

    @test_throws ArgumentError CPPLS.convert_auxiliary_to_float64(1.0)
end

@testset "cppls_prepare_data validates shapes and returns deflated matrices" begin
    X = Float32[1 2; 3 4; 5 6; 7 8]
    Y = Float32[1 0; 0 1; 1 0; 0 1]
    Y_aux = Float32[0.1 0.2; 0.2 0.3; 0.3 0.4; 0.4 0.5]
    weights = Float32[1, 2, 1, 2]

    X_prep,
    Y_prep,
    Y_combined,
    obs_weights,
    X_mean,
    Y_mean,
    X_deflated,
    X_loading_weights,
    X_loadings,
    Y_loadings,
    small_norm_flags,
    regression_coeffs,
    n_samples,
    n_targets = CPPLS.cppls_prepare_data(X, Y, 2, Y_aux, weights, true)

    @test X_prep isa Matrix{Float64}
    @test Y_prep isa Matrix{Float64}
    @test Y_combined == hcat(Y_prep, Float64.(Y_aux))
    @test obs_weights === weights
    @test size(X_mean) == (1, size(X, 2))
    @test size(Y_mean) == (1, size(Y, 2))
    @test size(X_deflated) == size(X_prep)
    @test size(X_loading_weights) == (size(X, 2), 2)
    @test size(X_loadings) == (size(X, 2), 2)
    @test size(Y_loadings) == (size(Y, 2), 2)
    @test size(small_norm_flags) == (2, size(X, 2))
    @test size(regression_coeffs) == (size(X, 2), size(Y, 2), 2)
    @test n_samples == size(X, 1)
    @test n_targets == size(Y, 2)

    @test_throws DimensionMismatch CPPLS.cppls_prepare_data(
        X,
        Y[1:3, :],
        2,
        nothing,
        nothing,
        true,
    )
    @test_throws DimensionMismatch CPPLS.cppls_prepare_data(
        X,
        Y,
        2,
        nothing,
        [1, 2, 3],
        true,
    )
end
