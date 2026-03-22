using CategoricalArrays
using StatsAPI

@testset "fit_cppls builds diagnostic-rich model" begin
    model = CPPLS.CPPLSModel(ncomponents=2, gamma=0.5)

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

    model = CPPLS.fit_cppls_core(model, X, Y, 2; gamma = 0.5)

    @test model isa CPPLS.CPPLSFit
    @test size(model.B) == (size(X, 2), size(Y, 2), 2)
    @test size(model.Y_hat) == (size(X, 1), size(Y, 2), 2)
    @test size(model.F) == size(model.Y_hat)
    @test size(model.T) == (size(X, 1), 2)
    @test size(model.U) == (size(Y, 1), 2)
    @test model.X_bar ≈ CPPLS.mean(X, dims = 1)
    @test model.Y_bar ≈ CPPLS.mean(Y, dims = 1)
    @test model.gamma ≈ fill(0.5, 2)
    @test length(model.rho) == 2
    @test size(CPPLS.gammas(model)) == (1, 2)
    @test size(CPPLS.rhos(model)) == (1, 2)
    @test CPPLS.gammas(model, 1) == [0.5]
    @test CPPLS.rhos(model, 1) ≈ [model.rho[1]]
    @test length(model.X_var) == 2
    @test model.X_var_total > 0
    @test model.samplelabels == ["1", "2", "3", "4", "5"]

    spec = CPPLS.CPPLSModel(ncomponents = 2, gamma = 0.5)
    model_from_spec = CPPLS.fit_cppls(spec, X, Y)
    @test model_from_spec isa CPPLS.CPPLSFit
    @test size(model_from_spec.B) ==
          (size(X, 2), size(Y, 2), 2)
    @test model_from_spec.samplelabels == ["1", "2", "3", "4", "5"]
end

@testset "fit_cppls stores gamma search matrices" begin
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
    gamma_candidates = Union{Float64, Tuple{Float64, Float64}}[
        0.0,
        (0.2, 0.8),
        1.0,
    ]
    model = CPPLS.CPPLSModel(ncomponents=1, gamma=gamma_candidates)

    model = CPPLS.fit_cppls_core(model, X, Y, 1; gamma = gamma_candidates)

    @test size(CPPLS.gammas(model)) == (length(gamma_candidates), 1)
    @test size(CPPLS.rhos(model)) == (length(gamma_candidates), 1)
    @test CPPLS.gammas(model, 1) == CPPLS.gammas(model)[:, 1]
    @test CPPLS.rhos(model, 1) == CPPLS.rhos(model)[:, 1]

    for k = 1:1
        best_idx = argmax(CPPLS.rhos(model, k))
        @test model.gamma[k] ≈ CPPLS.gammas(model, k)[best_idx]
        @test model.rho[k] ≈ CPPLS.rhos(model, k)[best_idx]
    end
end

@testset "StatsAPI fit/predict/coefs round-trip" begin
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

    spec = CPPLS.CPPLSModel(ncomponents = 2, gamma = 0.5)
    model = StatsAPI.fit(spec, X, Y)
    @test model isa CPPLS.CPPLSFit

    preds = StatsAPI.predict(model, X)
    @test size(preds) == (size(X, 1), size(Y, 2), 2)

    β = StatsAPI.coef(model)
    @test size(β) == (size(X, 2), size(Y, 2))

    fitted_vals = StatsAPI.fitted(model)
    @test size(fitted_vals) == (size(X, 1), size(Y, 2))

    resid_vals = StatsAPI.residuals(model)
    @test size(resid_vals) == size(fitted_vals)

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
    samplelabels = ["s1", "s2", "s3"]
    predictorlabels = [:p1, :p2]
    responselabels = ["r1", "r2"]

    model = CPPLS.CPPLSModel(ncomponents=1, gamma=0.5)
    model = CPPLS.fit_cppls_core(model,
        X,
        Y,
        1;
        gamma = 0.5,
        samplelabels = samplelabels,
        predictorlabels = predictorlabels,
        responselabels = responselabels,
    )

    @test model.samplelabels == samplelabels
    @test model.predictorlabels == predictorlabels
    @test model.responselabels == responselabels

    model = CPPLS.CPPLSModel(ncomponents=1, gamma=0.5)
    @test_throws ArgumentError CPPLS.fit_cppls_core(
        model,
        X,
        Y,
        1;
        gamma = 0.5,
        samplelabels = ["only_two"],
    )

    model = CPPLS.CPPLSModel(ncomponents=1, gamma=0.5)
    @test_throws ArgumentError CPPLS.fit_cppls_core(
        model,
        X,
        Y,
        1;
        gamma = 0.5,
        predictorlabels = [:p1],
    )

    model = CPPLS.CPPLSModel(ncomponents=1, gamma=0.5)
    @test_throws ArgumentError CPPLS.fit_cppls_core(
        model,
        X,
        Y,
        1;
        gamma = 0.5,
        responselabels = ["r1"],
    )

    model = CPPLS.CPPLSModel(ncomponents=1, gamma=0.5)
    @test_throws ArgumentError CPPLS.fit_cppls_core(
        model,
        X,
        Y,
        1;
        gamma = 0.5,
        mode = :discriminant,
    )

    model = CPPLS.CPPLSModel(ncomponents=1, gamma=0.5)
    @test_throws ArgumentError CPPLS.fit_cppls_core(
        model,
        X,
        Y,
        1;
        gamma = 0.5,
        sampleclasses = ["classA"],
    )

    model = CPPLS.CPPLSModel(ncomponents=1, gamma=0.5)
    @test_throws ArgumentError CPPLS.fit_cppls_core(
        model,
        X,
        Y,
        1;
        gamma = 0.5,
        mode = :unsupported_mode,
        responselabels = responselabels,
    )
end

@testset "fit_cppls with CPPLSModel enforces analysis mode for labels" begin
    X = Float64[
        1 0
        0 1
        1 1
        2 3
    ]
    labels = categorical(["classA", "classB", "classA", "classB"])

    spec = CPPLS.CPPLSModel(ncomponents = 2, gamma = 0.5, mode = :discriminant)
    model = CPPLS.fit_cppls(spec, X, labels)
    @test model.mode === :discriminant

    bad_spec = CPPLS.CPPLSModel(ncomponents = 2, gamma = 0.5, mode = :regression)
    @test_throws ArgumentError CPPLS.fit_cppls(bad_spec, X, labels)
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

    model = CPPLS.CPPLSModel(ncomponents=2, gamma=gamma_bounds)
    full = CPPLS.fit_cppls_core(model, X, Y, 2; gamma = gamma_bounds)
    light = CPPLS.fit_cppls_light_core(model, X, Y, 2; gamma = gamma_bounds)

    @test light isa CPPLS.CPPLSFitLight
    @test light.B ≈ full.B
    @test light.X_bar ≈ full.X_bar
    @test light.Y_bar ≈ full.Y_bar

    spec = CPPLS.CPPLSModel(ncomponents = 2, gamma = gamma_bounds)
    light_from_spec = CPPLS.fit_cppls_light(spec, X, Y)
    @test light_from_spec isa CPPLS.CPPLSFitLight
    @test light_from_spec.B ≈ full.B
end

@testset "fit_cppls categorical and vector wrappers" begin
    X = Float64[
        1 0
        0 1
        1 1
        2 3
    ]
    labels = categorical(["red", "blue", "red", "blue"])
    Y, inferred = CPPLS.onehot(labels)

    model = CPPLS.CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant)

    cpplsfit = CPPLS.fit_cppls(model, X, labels)
    @test cpplsfit.mode === :discriminant
    @test Set(cpplsfit.responselabels) == Set(inferred)
    @test cpplsfit.sampleclasses == labels
    @test !(cpplsfit.sampleclasses === labels)
    plain_labels = ["red", "blue", "red", "blue"]
    plain_cpplsfit = CPPLS.fit_cppls(model, X, plain_labels)
    @test plain_cpplsfit.mode === :discriminant
    @test Set(plain_cpplsfit.responselabels) == Set(unique(plain_labels))
    @test plain_cpplsfit.sampleclasses == plain_labels
    @test !(plain_cpplsfit.sampleclasses === plain_labels)
    @test plain_cpplsfit.B ≈ cpplsfit.B

    @test_throws ArgumentError CPPLS.fit_cppls(
        model,
        X,
        labels;
        responselabels = ["other"],
    )

    Y_vec = Float64[1, 0, 1, 0]
    vec_model = CPPLS.fit_cppls(model, X, Y_vec)
    mat_model = CPPLS.fit_cppls_core(model, X, reshape(Y_vec, :, 1), 2; gamma=0.5)

    @test vec_model.B ≈ mat_model.B
    @test vec_model.mode === :regression
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


    model = CPPLS.CPPLSModel(ncomponents=2, gamma=0.5)
    light = CPPLS.fit_cppls_light(model, X, Y)
    @test light.mode === :regression
    @test_throws ArgumentError CPPLS.CPPLSModel(ncomponents=2, gamma=0.5, mode=:invalid_mode)

    labels = categorical(["a", "b", "a", "b"])
    Y_one_hot, _ = CPPLS.onehot(labels)

    model = CPPLS.CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant)
    light_from_labels = CPPLS.fit_cppls_light(model, X, labels)
    manual_discriminant = CPPLS.fit_cppls_light_core(
        model,
        X,
        Y_one_hot,
        2;
        gamma = 0.5,
        mode = :discriminant,
    )

    @test light_from_labels.mode === :discriminant
    @test light_from_labels.B ≈
          manual_discriminant.B
    @test light_from_labels.X_bar ≈ manual_discriminant.X_bar
    @test light_from_labels.Y_bar ≈ manual_discriminant.Y_bar
    plain_labels = ["a", "b", "a", "b"]
    model = CPPLS.CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant)
    light_from_plain =
        CPPLS.fit_cppls_light(model, X, plain_labels)
    @test light_from_plain.mode === :discriminant
    @test light_from_plain.B ≈
          light_from_labels.B
    @test light_from_plain.X_bar ≈ light_from_labels.X_bar
    @test light_from_plain.Y_bar ≈ light_from_labels.Y_bar

    Y_vec = Float64[1, 0, 1, 0]
    model = CPPLS.CPPLSModel(ncomponents=2, gamma=0.5)
    light_vec = CPPLS.fit_cppls_light(model, X, Y_vec)
    light_vec_manual =
        CPPLS.fit_cppls_light_core(model, X, reshape(Y_vec, :, 1), 2; gamma = 0.5)

    @test light_vec.B ≈ light_vec_manual.B
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

    model = CPPLS.CPPLSModel(ncomponents=2, gamma=0.5)
    light_method = which(
        CPPLS.fit_cppls_light,
        Tuple{typeof(model), typeof(X),typeof(cat_labels)},
    )

    model = CPPLS.CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant)
    cat_light = CPPLS.fit_cppls_light(model, X, cat_labels)
    plain_light = CPPLS.fit_cppls_light(model, X, plain_labels)

    @test cat_light.mode === :discriminant
    @test cat_light.B ≈ plain_light.B
    @test cat_light.X_bar ≈ plain_light.X_bar
    @test cat_light.Y_bar ≈ plain_light.Y_bar
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
    ncomponents = 1

    model = CPPLS.CPPLSModel(ncomponents=ncomponents, gamma=0.5, center=true)

    X,
    Y_prim,
    Y,
    observation_weights,
    X_bar,
    Y_bar,
    X_def,
    W_comp,
    P,
    C,
    zero_mask,
    B,
    _,
    _ = CPPLS.cppls_prepare_data(
        model,
        X,
        Y,
        nothing,
        nothing
    )

    initial_weights = [3.0, 4.0]
    X_def_original = copy(X_def)

    tᵢ, t_norm, _ = CPPLS.process_component!(
        1,
        X_def,
        copy(initial_weights),
        Y_prim,
        W_comp,
        P,
        C,
        B,
        zero_mask,
        1e-12,
        1e-12,
        1e-10,
    )

    normalized_weights = initial_weights / CPPLS.norm(initial_weights)
    expected_scores = X_def_original * normalized_weights
    expected_norm = CPPLS.dot(expected_scores, expected_scores)
    expected_Y_loadings = (Y_prim' * expected_scores) / expected_norm
    expected_B =
        W_comp[:, 1:1] *
        CPPLS.pinv(P[:, 1:1]' * W_comp[:, 1:1]) *
        C[:, 1:1]'

    @test W_comp[:, 1] ≈ normalized_weights
    @test tᵢ ≈ expected_scores
    @test t_norm ≈ expected_norm
    @test C[:, 1] ≈ expected_Y_loadings
    @test B[:, :, 1] ≈ expected_B
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

    model = CPPLS.CPPLSModel(ncomponents=1, gamma=0.5, center=true)

    X,
    Y_prim,
    _,
    _,
    _,
    _,
    X_def,
    W_comp,
    P,
    C,
    zero_mask,
    B,
    _,
    _ = CPPLS.cppls_prepare_data(model, X, Y, nothing, nothing)

    X_def .= 0  # force zero scores regardless of weights
    initial_weights = [1.0, 2.0]
    tol = 1e-8

    _, t_norm, _ = CPPLS.process_component!(
        1,
        X_def,
        copy(initial_weights),
        Y_prim,
        W_comp,
        P,
        C,
        B,
        zero_mask,
        1e-12,
        1e-12,
        tol,
    )

    @test t_norm == tol
end
