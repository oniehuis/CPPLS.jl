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

@testset "onehot converts summed predictions to labels" begin
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

    result = CPPLS.onehot(cppls, predictions)
    @test result == expected_one_hot
end

@testset "onehot matches predict inputs" begin
    B = Array{Float64}(undef, 2, 2, 2)
    B[:, :, 1] = [0.8 0.1; -0.2 0.4]
    B[:, :, 2] = [0.3 -0.6; 0.5 0.2]
    X_bar = reshape([0.2, -0.1], 1, :)
    Y_bar = reshape([0.05, -0.05], 1, :)
    cppls = CPPLS.CPPLSFitLight(B, X_bar, Y_bar, :regression)

    X = [
        0.2 0.0
        1.0 -1.0
        -0.5 0.4
    ]

    expected = CPPLS.onehot(cppls, CPPLS.predict(cppls, X, 2))
    result = CPPLS.onehot(cppls, X, 2)

    @test result == expected
end

@testset "sampleclasses maps response labels" begin
    # model = CPPLS.CPPLSModel(
    #     gamma = 0.5,
    #     ncomponents = 1,
    #     mode = :discriminant,
    # )
    # X = Float64[
    #     1 0
    #     0 1
    #     1 1
    #     2 3
    # ]
    # labels = ["red", "blue", "red", "blue"]
    # cpplsfit = CPPLS.fit_cppls(model, X, labels)

    # preds = CPPLS.predict(cpplsfit, X, 1)
    # expected =
    #     cpplsfit.responselabels[
    #         CPPLS.sampleclasses(CPPLS.onehot(cpplsfit, preds)),
    #     ]

    # @test CPPLS.sampleclasses(cpplsfit, preds) == expected
    # @test CPPLS.sampleclasses(cpplsfit, X, 1) == expected
end

@testset "sampleclasses rejects regression models" begin
    # model = CPPLS.CPPLSModel(
    #     gamma = 0.5,
    #     ncomponents = 1
    # )
    # X = Float64[
    #     1 0
    #     0 1
    #     1 1
    # ]
    # y = Float64[1, 0, 1]
    # cpplsfit = CPPLS.fit_cppls(model, X, y)

    # preds = CPPLS.predict(cpplsfit, X, 1)
    # @test_throws ArgumentError CPPLS.sampleclasses(cpplsfit, preds)
end

@testset "project centers inputs before applying R" begin
    # model = CPPLS.CPPLSModel(
    #     gamma = 0.5,
    #     ncomponents = 2,
    #     mode = :discriminant
    # )
    # X_train = Float64[
    #     1 0
    #     0 1
    #     1 1
    #     2 3
    # ]
    # labels = ["red", "blue", "red", "blue"]
    # model = CPPLS.fit_cppls(model, X_train, labels)

    # X_new = Float64[
    #     0.5 -1.0
    #     1.0 0.0
    #     2.0 1.0
    # ]

    # centered = X_new .- model.X_bar
    # expected_scores = centered * model.R

    # scores = CPPLS.project(model, X_new)
    # @test scores ≈ expected_scores
end
