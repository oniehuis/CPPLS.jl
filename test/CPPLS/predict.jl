using LinearAlgebra: I

struct DummyProjectionModel <: CPPLS.AbstractCPPLS
    projection::Matrix{Float64}
    X_means::Matrix{Float64}
end

function mock_decision_line_cppls(scores::Matrix{Float64}, class_diff::Vector{Float64})
    n_samples, n_components = size(scores)
    n_predictors = n_components
    n_responses = 2

    regression_coefficients = zeros(Float64, n_predictors, n_responses, n_components)
    X_loadings = zeros(Float64, n_predictors, n_components)
    X_loading_weights = similar(X_loadings)
    Y_scores = zeros(Float64, n_samples, n_components)
    Y_loadings = zeros(Float64, n_responses, n_components)
    projection = Matrix{Float64}(I, n_predictors, n_components)
    X_means = zeros(Float64, 1, n_predictors)
    Y_means = zeros(Float64, 1, n_responses)

    fitted_values = zeros(Float64, n_samples, n_responses, n_components)
    for j = 1:n_components
        fitted_values[:, 1, j] .= class_diff
        fitted_values[:, 2, j] .= 0.0
    end

    residuals = similar(fitted_values)
    X_variance = ones(Float64, n_components)
    X_total_variance = 1.0
    gammas = fill(0.5, n_components)
    canonical_correlations = fill(0.9, n_components)
    small_norm_indices = zeros(Int, n_components, n_predictors)
    canonical_coefficients = zeros(Float64, n_responses, n_components)
    canonical_coefficients_y = zeros(Float64, n_responses, n_components)
    W0_weights = zeros(Float64, n_predictors, n_responses, n_components)
    Z = zeros(Float64, n_samples, n_responses, n_components)
    sample_labels = ["sample_$i" for i = 1:n_samples]
    predictor_labels = ["x_$j" for j = 1:n_predictors]
    response_labels = ["class1", "class2"]
    da_categories = [i ≤ n_samples ÷ 2 ? :a : :b for i = 1:n_samples]

    return CPPLS.CPPLS(
        regression_coefficients,
        scores,
        X_loadings,
        X_loading_weights,
        Y_scores,
        Y_loadings,
        projection,
        X_means,
        Y_means,
        fitted_values,
        residuals,
        X_variance,
        X_total_variance,
        gammas,
        canonical_correlations,
        small_norm_indices,
        canonical_coefficients,
        canonical_coefficients_y,
        W0_weights,
        Z;
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = response_labels,
        analysis_mode = :discriminant,
        da_categories = da_categories,
    )
end

@testset "predict applies centering and component selection" begin
    regression_coefficients = Array{Float64}(undef, 2, 2, 2)
    regression_coefficients[:, :, 1] = [1.0 0.0; 0.0 2.0]
    regression_coefficients[:, :, 2] = [0.5 -0.2; 0.3 0.1]
    X_means = reshape([1.0, 2.0], 1, :)
    Y_means = reshape([0.25, -0.5], 1, :)
    cppls = CPPLS.CPPLSLight(
        regression_coefficients,
        X_means,
        Y_means,
        :regression,
    )

    X = [
        1.0 2.0
        2.0 1.0
        3.0 4.0
    ]
    centered = X .- X_means

    expected = Array{Float64}(undef, size(X, 1), size(Y_means, 2), 2)
    for i = 1:2
        expected[:, :, i] = centered * regression_coefficients[:, :, i] .+ Y_means
    end

    preds_full = CPPLS.predict(cppls, X)
    preds_one = CPPLS.predict(cppls, X, 1)

    @test preds_full ≈ expected
    @test preds_one[:, :, 1] ≈ expected[:, :, 1]
    @test size(preds_full) == (size(X, 1), size(Y_means, 2), 2)
    @test_throws DimensionMismatch CPPLS.predict(cppls, X, 3)
end

@testset "predictonehot converts summed predictions to labels" begin
    regression_coefficients = ones(Float64, 1, 2, 1)
    X_means = reshape([0.0], 1, 1)
    Y_means = reshape([0.1, -0.2], 1, :)
    cppls = CPPLS.CPPLSLight(
        regression_coefficients,
        X_means,
        Y_means,
        :regression,
    )

    predictions = zeros(Float64, 3, 2, 2)
    predictions[:, :, 1] = [
        0.9 0.2
        0.1 0.8
        0.4 0.6
    ]
    predictions[:, :, 2] = [
        0.5 0.4
        0.2 0.7
        0.7 0.3
    ]

    summed = sum(predictions, dims = 3)[:, :, 1]
    adjusted = summed .- (size(predictions, 3) - 1) .* cppls.Y_means
    expected_labels = map(argmax, eachrow(adjusted))

    expected_one_hot = zeros(Int, 3, 2)
    for (row, label) in enumerate(expected_labels)
        expected_one_hot[row, label] = 1
    end

    result = CPPLS.predictonehot(cppls, predictions)
    @test result == expected_one_hot
end

@testset "project centers inputs before applying projection" begin
    X_means = reshape([0.5, -1.0], 1, :)
    projection = [
        1.0 0.0
        0.5 2.0
    ]
    dummy = DummyProjectionModel(projection, X_means)
    X = [
        0.5 -1.0
        1.0 0.0
        2.0 1.0
    ]

    centered = X .- X_means
    expected_scores = centered * projection

    scores = CPPLS.project(dummy, X)
    @test scores ≈ expected_scores
end

@testset "decision_line accepts tuple dims" begin
    X = [
        -1.0 0.0
        -0.5 0.2
        0.5 -0.1
        1.0 0.3
    ]
    labels = ["red", "red", "blue", "blue"]

    cppls = CPPLS.fit_cppls(X, labels, 2)
    line = CPPLS.decision_line(cppls; dims = (1, 2), n_components = 2)

    @test length(line.xs) == 2
    @test length(line.ys) == 2
    @test isfinite(line.intercept)
    @test length(line.normal) == 2
end

@testset "decision_line recovers separating hyperplane" begin
    scores = [
        -1.0 -0.5
        -0.2 0.0
        0.5 0.25
        1.5 0.75
    ]
    intercept = 0.4
    normal = [-0.3, 0.7]
    class_diff = intercept .+ scores * normal
    cppls = mock_decision_line_cppls(scores, class_diff)

    line = CPPLS.decision_line(cppls; dims = (1, 2), n_components = 2)

    @test line.intercept ≈ intercept atol = 1e-10
    @test line.normal ≈ normal atol = 1e-10
end

@testset "decision_line handles vertical separators" begin
    scores = [
        -0.2 -1.0
        -0.1 0.2
        -0.05 1.0
        -0.01 -0.5
    ]
    intercept = 0.1
    normal = [1.0, 0.0]
    class_diff = intercept .+ scores * normal
    cppls = mock_decision_line_cppls(scores, class_diff)

    line = CPPLS.decision_line(cppls; dims = (1, 2), n_components = 2)

    @test all(isapprox.(line.xs, fill(-intercept / normal[1], 2); atol = 1e-10))
    @test line.normal ≈ normal atol = 1e-10
end
