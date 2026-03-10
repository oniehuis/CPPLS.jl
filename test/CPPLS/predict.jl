using LinearAlgebra: I

struct DummyProjectionModel <: CPPLS.AbstractCPPLSFit
    R::Matrix{Float64}
    X_bar::Matrix{Float64}
end

function mock_decision_line_cppls(scores::Matrix{Float64}, class_diff::Vector{Float64})
    n_samples, n_components = size(scores)
    n_predictors = n_components
    n_responses = 2

    B = zeros(Float64, n_predictors, n_responses, n_components)
    P = zeros(Float64, n_predictors, n_components)
    W_comp = similar(P)
    U = zeros(Float64, n_samples, n_components)
    C = zeros(Float64, n_responses, n_components)
    R = Matrix{Float64}(I, n_predictors, n_components)
    X_bar = zeros(Float64, 1, n_predictors)
    Y_bar = zeros(Float64, 1, n_responses)

    Y_hat = zeros(Float64, n_samples, n_responses, n_components)
    for j = 1:n_components
        Y_hat[:, 1, j] .= class_diff
        Y_hat[:, 2, j] .= 0.0
    end

    F = similar(Y_hat)
    X_var = ones(Float64, n_components)
    X_var_total = 1.0
    gamma = fill(0.5, n_components)
    rho = fill(0.9, n_components)
    zero_mask = zeros(Int, n_components, n_predictors)
    a = zeros(Float64, n_responses, n_components)
    b = zeros(Float64, n_responses, n_components)
    W0 = zeros(Float64, n_predictors, n_responses, n_components)
    Z = zeros(Float64, n_samples, n_responses, n_components)
    sample_labels = ["sample_$i" for i = 1:n_samples]
    predictor_labels = ["x_$j" for j = 1:n_predictors]
    response_labels = ["class1", "class2"]
    da_categories = [i ≤ n_samples ÷ 2 ? :a : :b for i = 1:n_samples]

    return CPPLS.CPPLSFit(
        B,
        scores,
        P,
        W_comp,
        U,
        C,
        R,
        X_bar,
        Y_bar,
        Y_hat,
        F,
        X_var,
        X_var_total,
        gamma,
        rho,
        zero_mask,
        a,
        b,
        W0,
        Z;
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = response_labels,
        analysis_mode = :discriminant,
        da_categories = da_categories,
    )
end

@testset "predict applies centering and component selection" begin
    B = Array{Float64}(undef, 2, 2, 2)
    B[:, :, 1] = [1.0 0.0; 0.0 2.0]
    B[:, :, 2] = [0.5 -0.2; 0.3 0.1]
    X_bar = reshape([1.0, 2.0], 1, :)
    Y_bar = reshape([0.25, -0.5], 1, :)
    cppls = CPPLS.CPPLSFitLight(
        B,
        X_bar,
        Y_bar,
        :regression,
    )

    X = [
        1.0 2.0
        2.0 1.0
        3.0 4.0
    ]
    centered = X .- X_bar

    expected = Array{Float64}(undef, size(X, 1), size(Y_bar, 2), 2)
    for i = 1:2
        expected[:, :, i] = centered * B[:, :, i] .+ Y_bar
    end

    preds_full = CPPLS.predict(cppls, X)
    preds_one = CPPLS.predict(cppls, X, 1)

    @test preds_full ≈ expected
    @test preds_one[:, :, 1] ≈ expected[:, :, 1]
    @test size(preds_full) == (size(X, 1), size(Y_bar, 2), 2)
    @test_throws DimensionMismatch CPPLS.predict(cppls, X, 3)
end

@testset "predictonehot converts summed predictions to labels" begin
    B = ones(Float64, 1, 2, 1)
    X_bar = reshape([0.0], 1, 1)
    Y_bar = reshape([0.1, -0.2], 1, :)
    cppls = CPPLS.CPPLSFitLight(
        B,
        X_bar,
        Y_bar,
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
    adjusted = summed .- (size(predictions, 3) - 1) .* cppls.Y_bar
    expected_labels = map(argmax, eachrow(adjusted))

    expected_one_hot = zeros(Int, 3, 2)
    for (row, label) in enumerate(expected_labels)
        expected_one_hot[row, label] = 1
    end

    result = CPPLS.predictonehot(cppls, predictions)
    @test result == expected_one_hot
end

@testset "project centers inputs before applying R" begin
    X_bar = reshape([0.5, -1.0], 1, :)
    R = [
        1.0 0.0
        0.5 2.0
    ]
    dummy = DummyProjectionModel(R, X_bar)
    X = [
        0.5 -1.0
        1.0 0.0
        2.0 1.0
    ]

    centered = X .- X_bar
    expected_scores = centered * R

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
