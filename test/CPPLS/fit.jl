using CategoricalArrays

@testset "fit_cppls builds diagnostic-rich model" begin
    X = Float64[
        1 0 2
        0 1 2
        1 1 1
        2 3 0
        3 2 1
    ]
    Y = Float64[
        1 0
        0 1
        1 1
        0 1
        1 0
    ]

    model = CPPLS.fit_cppls(X, Y, 2; gamma = 0.5)

    @test model isa CPPLS.CPPLS
    @test size(model.regression_coefficients) == (size(X, 2), size(Y, 2), 2)
    @test size(model.fitted_values) == (size(X, 1), size(Y, 2), 2)
    @test size(model.residuals) == size(model.fitted_values)
    @test size(model.X_scores) == (size(X, 1), 2)
    @test size(model.Y_scores) == (size(Y, 1), 2)
    @test model.X_means ≈ CPPLS.mean(X, dims = 1)
    @test model.Y_means ≈ CPPLS.mean(Y, dims = 1)
    @test model.gammas ≈ fill(0.5, 2)
    @test length(model.canonical_correlations) == 2
    @test length(model.X_variance) == 2
    @test model.X_total_variance > 0
end

@testset "fit_cppls enforces label metadata" begin
    X = Float64[
        1 2
        3 4
        5 6
    ]
    Y = Float64[
        1 0
        0 1
        1 1
    ]
    sample_labels = ["s1", "s2", "s3"]
    predictor_labels = [:p1, :p2]
    response_labels = ["r1", "r2"]

    model = CPPLS.fit_cppls(
        X,
        Y,
        1;
        gamma = 0.5,
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = response_labels,
    )

    @test model.sample_labels == sample_labels
    @test model.predictor_labels == predictor_labels
    @test model.response_labels == response_labels

    @test_throws ArgumentError CPPLS.fit_cppls(
        X,
        Y,
        1;
        gamma = 0.5,
        sample_labels = ["only_two"],
    )

    @test_throws ArgumentError CPPLS.fit_cppls(
        X,
        Y,
        1;
        gamma = 0.5,
        predictor_labels = [:p1],
    )

    @test_throws ArgumentError CPPLS.fit_cppls(
        X,
        Y,
        1;
        gamma = 0.5,
        response_labels = ["r1"],
    )

    @test_throws ArgumentError CPPLS.fit_cppls(
        X,
        Y,
        1;
        gamma = 0.5,
        analysis_mode = :discriminant,
    )

    @test_throws ArgumentError CPPLS.fit_cppls(
        X,
        Y,
        1;
        gamma = 0.5,
        da_categories = ["classA"],
    )

    @test_throws ArgumentError CPPLS.fit_cppls(
        X,
        Y,
        1;
        gamma = 0.5,
        analysis_mode = :unsupported_mode,
        response_labels = response_labels,
    )
end

@testset "fit_cppls_light matches regression from full model" begin
    X = Float64[
        2 1
        0 3
        4 5
        1 4
    ]
    Y = Float64[
        1 0
        0 1
        1 1
        0 1
    ]
    gamma_bounds = (0.2, 0.8)

    full = CPPLS.fit_cppls(X, Y, 2; gamma = gamma_bounds)
    light = CPPLS.fit_cppls_light(X, Y, 2; gamma = gamma_bounds)

    @test light isa CPPLS.CPPLSLight
    @test light.regression_coefficients ≈ full.regression_coefficients
    @test light.X_means ≈ full.X_means
    @test light.Y_means ≈ full.Y_means
end

@testset "fit_cppls categorical and vector wrappers" begin
    X = Float64[
        1 0
        0 1
        1 1
        2 3
    ]
    labels = categorical(["red", "blue", "red", "blue"])
    Y, inferred = CPPLS.labels_to_one_hot(labels)

    model = CPPLS.fit_cppls(X, labels, 2; gamma = 0.5)
    @test model.analysis_mode === :discriminant
    @test Set(model.response_labels) == Set(inferred)
    @test model.da_categories == labels
    @test !(model.da_categories === labels)
    plain_labels = ["red", "blue", "red", "blue"]
    plain_model = CPPLS.fit_cppls(X, plain_labels, 2; gamma = 0.5)
    @test plain_model.analysis_mode === :discriminant
    @test Set(plain_model.response_labels) == Set(unique(plain_labels))
    @test plain_model.da_categories == plain_labels
    @test !(plain_model.da_categories === plain_labels)
    @test plain_model.regression_coefficients ≈ model.regression_coefficients

    @test_throws ArgumentError CPPLS.fit_cppls(
        X,
        labels,
        2;
        response_labels = ["other"],
    )

    Y_vec = Float64[1, 0, 1, 0]
    vec_model = CPPLS.fit_cppls(X, Y_vec, 2; gamma = 0.5)
    mat_model = CPPLS.fit_cppls(X, reshape(Y_vec, :, 1), 2; gamma = 0.5)

    @test vec_model.regression_coefficients ≈ mat_model.regression_coefficients
    @test vec_model.analysis_mode === :regression
end

@testset "fit_cppls categorical dispatch method" begin
    X = Float64[
        1 0
        0 1
        2 1
    ]
    cat_labels = categorical(["g1", "g2", "g1"])
    plain_labels = ["g1", "g2", "g1"]

    cat_method =
        which(CPPLS.fit_cppls, Tuple{typeof(X),typeof(cat_labels),Int})

    cat_sig = Base.unwrap_unionall(cat_method.sig)
    @test cat_sig.parameters[3].name.wrapper === CategoricalArrays.AbstractCategoricalArray

    cat_model = CPPLS.fit_cppls(X, cat_labels, 2; gamma = 0.5)
    plain_model = CPPLS.fit_cppls(X, plain_labels, 2; gamma = 0.5)

    @test cat_model.analysis_mode === :discriminant
    @test cat_model.regression_coefficients ≈ plain_model.regression_coefficients
    @test cat_model.X_means ≈ plain_model.X_means
    @test cat_model.Y_means ≈ plain_model.Y_means
end

@testset "fit_cppls_light wrappers enforce analysis mode" begin
    X = Float64[
        1 0
        0 1
        2 1
        3 2
    ]
    Y = Float64[
        1 0
        0 1
        1 1
        0 1
    ]

    light = CPPLS.fit_cppls_light(X, Y, 2; gamma = 0.5)
    @test light.analysis_mode === :regression
    @test_throws ArgumentError CPPLS.fit_cppls_light(
        X,
        Y,
        2;
        gamma = 0.5,
        analysis_mode = :invalid_mode,
    )

    labels = categorical(["a", "b", "a", "b"])
    Y_one_hot, _ = CPPLS.labels_to_one_hot(labels)

    light_from_labels = CPPLS.fit_cppls_light(X, labels, 2; gamma = 0.5)
    manual_discriminant = CPPLS.fit_cppls_light(
        X,
        Y_one_hot,
        2;
        gamma = 0.5,
        analysis_mode = :discriminant,
    )

    @test light_from_labels.analysis_mode === :discriminant
    @test light_from_labels.regression_coefficients ≈
          manual_discriminant.regression_coefficients
    @test light_from_labels.X_means ≈ manual_discriminant.X_means
    @test light_from_labels.Y_means ≈ manual_discriminant.Y_means
    plain_labels = ["a", "b", "a", "b"]
    light_from_plain =
        CPPLS.fit_cppls_light(X, plain_labels, 2; gamma = 0.5)
    @test light_from_plain.analysis_mode === :discriminant
    @test light_from_plain.regression_coefficients ≈
          light_from_labels.regression_coefficients
    @test light_from_plain.X_means ≈ light_from_labels.X_means
    @test light_from_plain.Y_means ≈ light_from_labels.Y_means

    Y_vec = Float64[1, 0, 1, 0]
    light_vec = CPPLS.fit_cppls_light(X, Y_vec, 2; gamma = 0.5)
    light_vec_manual =
        CPPLS.fit_cppls_light(X, reshape(Y_vec, :, 1), 2; gamma = 0.5)

    @test light_vec.regression_coefficients ≈ light_vec_manual.regression_coefficients
end

@testset "fit_cppls_light categorical dispatch method" begin
    X = Float64[
        1 0
        0 1
        1 2
        2 3
    ]
    cat_labels = categorical(["alpha", "beta", "alpha", "beta"])
    plain_labels = ["alpha", "beta", "alpha", "beta"]

    light_method = which(
        CPPLS.fit_cppls_light,
        Tuple{typeof(X),typeof(cat_labels),Int},
    )
    light_sig = Base.unwrap_unionall(light_method.sig)
    @test light_sig.parameters[3].name.wrapper ===
          CategoricalArrays.AbstractCategoricalArray

    cat_light = CPPLS.fit_cppls_light(X, cat_labels, 2; gamma = 0.5)
    plain_light = CPPLS.fit_cppls_light(X, plain_labels, 2; gamma = 0.5)

    @test cat_light.analysis_mode === :discriminant
    @test cat_light.regression_coefficients ≈ plain_light.regression_coefficients
    @test cat_light.X_means ≈ plain_light.X_means
    @test cat_light.Y_means ≈ plain_light.Y_means
end

@testset "process_component! normalizes weights and deflates predictors" begin
    X = Float64[
        2 0
        0 1
        1 3
    ]
    Y = Float64[
        1 0
        0 1
        1 1
    ]
    n_components = 1

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
    _,
    _ = CPPLS.cppls_prepare_data(
        X,
        Y,
        n_components,
        nothing,
        nothing,
        true,
    )

    initial_weights = [3.0, 4.0]
    X_deflated_original = copy(X_deflated)

    X_scoresᵢ, t_norm, _ = CPPLS.process_component!(
        1,
        X_deflated,
        copy(initial_weights),
        Y_responses,
        X_loading_weights,
        X_loadings,
        Y_loadings,
        regression_coefficients,
        small_norm_flags,
        1e-12,
        1e-12,
        1e-10,
    )

    normalized_weights = initial_weights / CPPLS.norm(initial_weights)
    expected_scores = X_deflated_original * normalized_weights
    expected_norm = CPPLS.dot(expected_scores, expected_scores)
    expected_Y_loadings = (Y_responses' * expected_scores) / expected_norm
    expected_B =
        X_loading_weights[:, 1:1] *
        CPPLS.pinv(X_loadings[:, 1:1]' * X_loading_weights[:, 1:1]) *
        Y_loadings[:, 1:1]'

    @test X_loading_weights[:, 1] ≈ normalized_weights
    @test X_scoresᵢ ≈ expected_scores
    @test t_norm ≈ expected_norm
    @test Y_loadings[:, 1] ≈ expected_Y_loadings
    @test regression_coefficients[:, :, 1] ≈ expected_B
end

@testset "process_component! guards zero-norm scores" begin
    X = Float64[
        1 0
        0 1
        1 1
    ]
    Y = Float64[
        1 0
        0 1
        1 0
    ]
    X_predictors,
    Y_responses,
    _,
    _,
    _,
    _,
    X_deflated,
    X_loading_weights,
    X_loadings,
    Y_loadings,
    small_norm_flags,
    regression_coefficients,
    _,
    _ = CPPLS.cppls_prepare_data(X, Y, 1, nothing, nothing, true)

    X_deflated .= 0  # force zero scores regardless of weights
    initial_weights = [1.0, 2.0]
    tol = 1e-8

    _, t_norm, _ = CPPLS.process_component!(
        1,
        X_deflated,
        copy(initial_weights),
        Y_responses,
        X_loading_weights,
        X_loadings,
        Y_loadings,
        regression_coefficients,
        small_norm_flags,
        1e-12,
        1e-12,
        tol,
    )

    @test t_norm == tol
end
