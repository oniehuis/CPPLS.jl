using CategoricalArrays

function suppress_info(f::Function)
    logger = Base.CoreLogging.SimpleLogger(IOBuffer(), Base.CoreLogging.Error)
    Base.CoreLogging.with_logger(logger) do
        f()
    end
end

const CROSSVAL_X = Float64[
    1.1 0.5 1.7 2.3
    2.2 1.1 0.4 3.6
    3.3 1.8 2.5 4.9
    4.4 2.6 1.1 6.1
    5.5 3.3 3.2 7.4
    6.6 4.0 1.8 8.8
    7.7 4.6 2.9 9.9
    8.8 5.3 0.7 11.2
    9.9 6.1 3.6 12.5
    11.0 6.8 2.2 13.7
    12.1 7.5 4.3 15.0
    13.2 8.2 1.5 16.3
    14.3 9.0 3.8 17.5
    15.4 9.7 2.0 18.9
    16.5 10.4 4.6 20.1
    17.6 11.1 1.2 21.4
]

const CROSSVAL_Y = [
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
]

const CROSSVAL_Y_REG = reshape(
    CROSSVAL_X[:, 1] .+ 0.5 .* CROSSVAL_X[:, 2],
    :,
    1,
)

@testset "random_batch_indices builds stratified folds" begin
    strata = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    folds = CPPLS.random_batch_indices(
        strata,
        3,
        CPPLS.MersenneTwister(1),
    )

    @test length(folds) == 3
    @test sort!(reduce(vcat, folds)) == collect(1:length(strata))
    @test all(length(batch) == 4 for batch in folds)
    @test_throws ArgumentError CPPLS.random_batch_indices(strata, 4)
    @test_throws ArgumentError CPPLS.random_batch_indices(strata, 0)
    @test_throws ArgumentError CPPLS.random_batch_indices(
        strata,
        length(strata) + 1,
    )

    uneven = vcat(fill(1, 5), fill(2, 4))
    @test_logs (:info, r"Stratum 1 .* not evenly divisible") begin
        CPPLS.random_batch_indices(uneven, 2, CPPLS.MersenneTwister(2))
    end
end

@testset "cv_classification builds default functions" begin
    cfg = CPPLS.cv_classification()
    @test haskey(cfg, :score_fn)
    @test haskey(cfg, :predict_fn)
    @test haskey(cfg, :select_fn)
    @test haskey(cfg, :flag_fn)

    Y_true = [1 0; 0 1]
    Y_pred = [1 0; 1 0]
    score = cfg.score_fn(Y_true, Y_pred)
    @test 0.0 ≤ score ≤ 1.0
    @test cfg.flag_fn(Y_true, Y_pred) == [false, true]
    @test cfg.select_fn([0.1, 0.2]) == 2

    Y_true_imbalanced = [1 0; 0 1; 0 1]
    Y_pred_imbalanced = [0 1; 0 1; 0 1]
    @test cfg.score_fn(Y_true_imbalanced, Y_pred_imbalanced) ≈ 0.5
end

@testset "cv_regression builds default functions" begin
    cfg = CPPLS.cv_regression()
    @test haskey(cfg, :score_fn)
    @test haskey(cfg, :predict_fn)
    @test haskey(cfg, :select_fn)

    Y_true = reshape([1.0, 2.0], :, 1)
    Y_pred = reshape([1.0, 3.0], :, 1)
    @test cfg.score_fn(Y_true, Y_pred) ≈ sqrt(0.5)
    @test cfg.select_fn([0.3, 0.2]) == 2

    B = reshape([2.0], 1, 1, 1)
    X_bar = reshape([0.0], 1, 1)
    Y_bar = reshape([0.5], 1, 1)
    model = CPPLS.CPPLSFitLight(B, X_bar, Y_bar, :regression)
    X = reshape([1.0, 2.0], :, 1)
    @test cfg.predict_fn(model, X, 1) ≈ reshape([2.5, 4.5], :, 1)
end

@testset "optimize_num_latent_variables selects component count" begin
    spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :discriminant)
    cfg = CPPLS.cv_classification()
    selected = CPPLS.optimize_num_latent_variables(
        CROSSVAL_X,
        CROSSVAL_Y,
        1,
        2,
        2,
        spec,
        (;),
        nothing,
        cfg.score_fn,
        cfg.predict_fn,
        cfg.select_fn,
        CPPLS.MersenneTwister(42),
        false;
        strata = CPPLS.one_hot_to_labels(CROSSVAL_Y),
    )
    @test selected == 1
end

@testset "resolve_obs_weights combines fixed and fold-local weights" begin
    fit_kwargs = (; obs_weights = [1.0, 2.0])
    spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :discriminant)
    resolved = CPPLS.resolve_obs_weights(
        fit_kwargs,
        (X, Y; kwargs...) -> [0.5, 0.25],
        CROSSVAL_X[1:2, :],
        CROSSVAL_Y[1:2, :],
        [3, 7],
        spec,
    )

    @test resolved.obs_weights ≈ [0.5, 0.5]
end

@testset "nestedcv returns scores and component choices" begin
    spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :discriminant)
    cfg = CPPLS.cv_classification()
    weight_calls = Ref(0)
    scores, components = suppress_info() do
        CPPLS.nestedcv(
            CROSSVAL_X,
            CROSSVAL_Y;
            spec = spec,
            fit_kwargs = (;),
            obs_weight_fn = (X, Y; kwargs...) -> begin
                weight_calls[] += 1
                ones(size(X, 1))
            end,
            score_fn = cfg.score_fn,
            predict_fn = cfg.predict_fn,
            select_fn = cfg.select_fn,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            strata = CPPLS.one_hot_to_labels(CROSSVAL_Y),
            rng = CPPLS.MersenneTwister(123),
            verbose = false,
        )
    end

    @test length(scores) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in scores)
    @test components == [1, 1]
    @test weight_calls[] == 6
end

@testset "nestedcvperm shuffles responses" begin
    spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :discriminant)
    cfg = CPPLS.cv_classification()
    perms = suppress_info() do
        CPPLS.nestedcvperm(
            CROSSVAL_X,
            CROSSVAL_Y;
            spec = spec,
            fit_kwargs = (;),
            obs_weight_fn = (X, Y; kwargs...) -> ones(size(X, 1)),
            score_fn = cfg.score_fn,
            predict_fn = cfg.predict_fn,
            select_fn = cfg.select_fn,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            num_permutations = 2,
            strata = CPPLS.one_hot_to_labels(CROSSVAL_Y),
            rng = CPPLS.MersenneTwister(321),
            verbose = false,
        )
    end

    @test length(perms) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in perms)
end

@testset "cvreg applies regression defaults" begin
    spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :regression)

    scores, components = suppress_info() do
        CPPLS.cvreg(
            CROSSVAL_X,
            CROSSVAL_Y_REG;
            spec = spec,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(666),
            verbose = false,
        )
    end

    @test length(scores) == 2
    @test components == [1, 1]
    @test all(isfinite, scores)

    scores_vec, components_vec = suppress_info() do
        CPPLS.cvreg(
            CROSSVAL_X,
            vec(CROSSVAL_Y_REG);
            spec = spec,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(666),
            verbose = false,
        )
    end

    @test scores_vec == scores
    @test components_vec == components
    @test_throws MethodError CPPLS.cvreg(CROSSVAL_X, CROSSVAL_Y_REG; spec = spec, score_fn = identity)
    @test_throws ArgumentError CPPLS.cvreg(
        CROSSVAL_X,
        CROSSVAL_Y_REG;
        spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :discriminant),
    )
end

@testset "permreg applies regression defaults" begin
    spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :regression)

    permutation_scores = suppress_info() do
        CPPLS.permreg(
            CROSSVAL_X,
            vec(CROSSVAL_Y_REG);
            spec = spec,
            num_permutations = 2,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(777),
            verbose = false,
        )
    end

    @test length(permutation_scores) == 2
    @test all(isfinite, permutation_scores)
    @test_throws MethodError CPPLS.permreg(CROSSVAL_X, CROSSVAL_Y_REG; spec = spec, predict_fn = identity)
    @test_throws ArgumentError CPPLS.permreg(
        CROSSVAL_X,
        CROSSVAL_Y_REG;
        spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :discriminant),
    )
end

@testset "cvda applies DA defaults and limits control" begin
    spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :discriminant)
    classes = repeat(["A", "B"], inner = size(CROSSVAL_X, 1) ÷ 2)

    scores, components = suppress_info() do
        CPPLS.cvda(
            CROSSVAL_X,
            classes;
            spec = spec,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(444),
            verbose = false,
        )
    end

    @test length(scores) == 2
    @test components == [1, 1]
    @test all(0.0 ≤ acc ≤ 1.0 for acc in scores)

    Y, labels = CPPLS.labels_to_one_hot(classes)
    scores_matrix, components_matrix = suppress_info() do
        CPPLS.cvda(
            CROSSVAL_X,
            Y;
            spec = spec,
            fit_kwargs = (; responselabels = labels),
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(444),
            verbose = false,
        )
    end

    @test scores_matrix == scores
    @test components_matrix == components

    @test_throws MethodError CPPLS.cvda(
        CROSSVAL_X,
        classes;
        spec = spec,
        score_fn = identity,
    )
    @test_throws MethodError CPPLS.cvda(
        CROSSVAL_X,
        classes;
        spec = spec,
        obs_weight_fn = (X, Y; kwargs...) -> ones(size(X, 1)),
    )
    @test_throws ArgumentError CPPLS.cvda(
        CROSSVAL_X,
        classes;
        spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :regression),
    )
    @test_throws ArgumentError CPPLS.cvda(CROSSVAL_X, collect(1:size(CROSSVAL_X, 1)); spec = spec)
end

@testset "permda applies DA defaults and limits control" begin
    spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :discriminant)
    classes = repeat(["A", "B"], inner = size(CROSSVAL_X, 1) ÷ 2)

    permutation_scores = suppress_info() do
        CPPLS.permda(
            CROSSVAL_X,
            classes;
            spec = spec,
            num_permutations = 2,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(555),
            verbose = false,
        )
    end

    @test length(permutation_scores) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in permutation_scores)

    @test_throws MethodError CPPLS.permda(
        CROSSVAL_X,
        classes;
        spec = spec,
        strata = ones(Int, size(CROSSVAL_X, 1)),
    )
    @test_throws MethodError CPPLS.permda(
        CROSSVAL_X,
        classes;
        spec = spec,
        predict_fn = identity,
    )
    @test_throws ArgumentError CPPLS.permda(
        CROSSVAL_X,
        classes;
        spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :regression),
    )
    @test_throws ArgumentError CPPLS.permda(CROSSVAL_X, collect(1:size(CROSSVAL_X, 1)); spec = spec)
end

@testset "outlierscan returns per-sample counts" begin
    spec = CPPLS.CPPLSSpec(n_components = 1, gamma = 0.5, analysis_mode = :discriminant)
    out = suppress_info() do
        CPPLS.outlierscan(
            CROSSVAL_X,
            CROSSVAL_Y;
            spec = spec,
            fit_kwargs = (;),
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(111),
            verbose = false,
        )
    end

    @test length(out.n_tested) == size(CROSSVAL_X, 1)
    @test length(out.n_flagged) == size(CROSSVAL_X, 1)
    @test all(0.0 ≤ r ≤ 1.0 for r in out.rate)

    expected = 2 * (size(CROSSVAL_X, 1) ÷ 2)
    @test sum(out.n_tested) == expected
    @test all(out.n_flagged .≤ out.n_tested)

    out_unweighted = suppress_info() do
        CPPLS.outlierscan(
            CROSSVAL_X,
            CROSSVAL_Y;
            spec = spec,
            fit_kwargs = (;),
            obs_weight_fn = nothing,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(111),
            verbose = false,
        )
    end

    @test length(out_unweighted.n_tested) == size(CROSSVAL_X, 1)
    @test all(out_unweighted.n_flagged .≤ out_unweighted.n_tested)
end
