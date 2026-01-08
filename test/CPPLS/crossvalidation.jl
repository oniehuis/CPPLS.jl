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
]

const CROSSVAL_LABELS = categorical([
    "class1",
    "class2",
    "class1",
    "class2",
    "class1",
    "class2",
    "class1",
    "class2",
    "class1",
    "class2",
    "class1",
    "class2",
])

const CROSSVAL_LABELS_PLAIN = [
    "class1",
    "class2",
    "class1",
    "class2",
    "class1",
    "class2",
    "class1",
    "class2",
    "class1",
    "class2",
    "class1",
    "class2",
]

@testset "random_batch_indices builds stratified folds" begin
    strata = [1, 1, 1, 2, 2, 2]
    folds = CPPLS.random_batch_indices(
        strata,
        3,
        CPPLS.MersenneTwister(1),
    )

    @test length(folds) == 3
    @test sort!(reduce(vcat, folds)) == collect(1:length(strata))
    @test all(length(batch) == 2 for batch in folds)
    @test_throws ArgumentError CPPLS.random_batch_indices(strata, 0)
    @test_throws ArgumentError CPPLS.random_batch_indices(
        strata,
        length(strata) + 1,
    )
end

@testset "optimize_num_latent_variables selects component count" begin
    selected = CPPLS.optimize_num_latent_variables(
        CROSSVAL_X,
        CROSSVAL_Y,
        1,
        2,
        2,
        0.5,
        nothing,
        nothing,
        true,
        1e-12,
        eps(Float64),
        1e-10,
        1e-6,
        1e-12,
        true,
        CPPLS.MersenneTwister(42),
        false,
    )
    @test selected == 1

    selected_labels = CPPLS.optimize_num_latent_variables(
        CROSSVAL_X,
        CROSSVAL_LABELS,
        1,
        2,
        2,
        0.5,
        nothing,
        nothing,
        true,
        1e-12,
        eps(Float64),
        1e-10,
        1e-6,
        1e-12,
        true,
        CPPLS.MersenneTwister(42),
        false,
    )
    @test selected_labels == selected

    selected_plain = CPPLS.optimize_num_latent_variables(
        CROSSVAL_X,
        CROSSVAL_LABELS_PLAIN,
        1,
        2,
        2,
        0.5,
        nothing,
        nothing,
        true,
        1e-12,
        eps(Float64),
        1e-10,
        1e-6,
        1e-12,
        true,
        CPPLS.MersenneTwister(42),
        false,
    )
    @test selected_plain == selected

    opt_method = which(
        CPPLS.optimize_num_latent_variables,
        Tuple{
            typeof(CROSSVAL_X),
            typeof(CROSSVAL_LABELS),
            Int,
            Int,
            Int,
            Float64,
            Nothing,
            Nothing,
            Bool,
            Float64,
            Float64,
            Float64,
            Float64,
            Float64,
            Bool,
            CPPLS.MersenneTwister,
            Bool,
        },
    )
    opt_sig = Base.unwrap_unionall(opt_method.sig)
    @test opt_sig.parameters[3].name.wrapper === CategoricalArrays.AbstractCategoricalArray

    reg_matrix = Float64.(CROSSVAL_Y)
    @test_throws ArgumentError CPPLS.optimize_num_latent_variables(
        CROSSVAL_X,
        reg_matrix,
        1,
        2,
        2,
        0.5,
        nothing,
        nothing,
        true,
        1e-12,
        eps(Float64),
        1e-10,
        1e-6,
        1e-12,
        true,
        CPPLS.MersenneTwister(42),
        false,
    )

    reg_vector = randn(size(CROSSVAL_X, 1))
    @test_throws ArgumentError CPPLS.optimize_num_latent_variables(
        CROSSVAL_X,
        reg_vector,
        1,
        2,
        2,
        0.5,
        nothing,
        nothing,
        true,
        1e-12,
        eps(Float64),
        1e-10,
        1e-6,
        1e-12,
        true,
        CPPLS.MersenneTwister(42),
        false,
    )
end

@testset "nested_cv returns accuracies and component choices" begin
    accuracies, components = suppress_info() do
        CPPLS.nested_cv(
            CROSSVAL_X,
            CROSSVAL_Y;
            gamma = 0.5,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(123),
            verbose = false,
        )
    end

    @test length(accuracies) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in accuracies)
    @test length(components) == 2
    @test all(comp == 1 for comp in components)

    accuracies_labels, components_labels = suppress_info() do
        CPPLS.nested_cv(
            CROSSVAL_X,
            CROSSVAL_LABELS;
            gamma = 0.5,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(1234),
            verbose = false,
        )
    end
    @test components_labels == components

    accuracies_plain, components_plain = suppress_info() do
        CPPLS.nested_cv(
            CROSSVAL_X,
            CROSSVAL_LABELS_PLAIN;
            gamma = 0.5,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(4321),
            verbose = false,
        )
    end
    @test components_plain == components

    real_vector = randn(size(CROSSVAL_X, 1))
    @test_throws ArgumentError suppress_info() do
        CPPLS.nested_cv(
            CROSSVAL_X,
            real_vector;
            gamma = 0.5,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(111),
            verbose = false,
        )
    end

    @test_throws ArgumentError suppress_info() do
        CPPLS.nested_cv(
            CROSSVAL_X,
            Float64.(CROSSVAL_Y);
            gamma = 0.5,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            rng = CPPLS.MersenneTwister(100),
            verbose = false,
        )
    end

    nested_method = which(
        CPPLS.nested_cv,
        Tuple{typeof(CROSSVAL_X),typeof(CROSSVAL_LABELS)},
    )
    nested_sig = Base.unwrap_unionall(nested_method.sig)
    @test nested_sig.parameters[3].name.wrapper ===
          CategoricalArrays.AbstractCategoricalArray
end

@testset "nested_cv_permutation shuffles responses" begin
    perms = suppress_info() do
        CPPLS.nested_cv_permutation(
            CROSSVAL_X,
            CROSSVAL_Y;
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            num_permutations = 2,
            center = false,
            rng = CPPLS.MersenneTwister(321),
            verbose = false,
        )
    end

    @test length(perms) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in perms)

    perms_labels = suppress_info() do
        CPPLS.nested_cv_permutation(
            CROSSVAL_X,
            CROSSVAL_LABELS;
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            num_permutations = 2,
            center = false,
            rng = CPPLS.MersenneTwister(654),
            verbose = false,
        )
    end
    @test length(perms_labels) == 2

    perms_plain = suppress_info() do
        CPPLS.nested_cv_permutation(
            CROSSVAL_X,
            CROSSVAL_LABELS_PLAIN;
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            num_permutations = 2,
            center = false,
            rng = CPPLS.MersenneTwister(222),
            verbose = false,
        )
    end
    @test length(perms_plain) == length(perms_labels)

    real_vector = randn(size(CROSSVAL_X, 1))
    @test_throws ArgumentError suppress_info() do
        CPPLS.nested_cv_permutation(
            CROSSVAL_X,
            real_vector;
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            num_permutations = 2,
            center = false,
            rng = CPPLS.MersenneTwister(777),
            verbose = false,
        )
    end

    @test_throws ArgumentError suppress_info() do
        CPPLS.nested_cv_permutation(
            CROSSVAL_X,
            Float64.(CROSSVAL_Y);
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            num_permutations = 2,
            center = false,
            rng = CPPLS.MersenneTwister(987),
            verbose = false,
        )
    end

    perm_method = which(
        CPPLS.nested_cv_permutation,
        Tuple{typeof(CROSSVAL_X),typeof(CROSSVAL_LABELS)},
    )
    perm_sig = Base.unwrap_unionall(perm_method.sig)
    @test perm_sig.parameters[3].name.wrapper === CategoricalArrays.AbstractCategoricalArray
end
