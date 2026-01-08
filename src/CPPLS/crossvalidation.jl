"""
    CPPLS.random_batch_indices(strata::AbstractVector{<:Integer},
        num_batches::Integer, rng::AbstractRNG=Random.GLOBAL_RNG)

Construct stratified folds. For each unique entry in `strata` the corresponding
sample indices are shuffled with `rng` and then dealt round-robin into
`num_batches` disjoint vectors. This keeps class proportions stable across
folds. Throws if `num_batches` is less than `1` or larger than the number of
samples. Returns a vector-of-vectors of 1-based indices, each representing one
fold.

# Example
```
julia> using Random; rng = MersenneTwister(1);

julia> folds = CPPLS.random_batch_indices([1, 1, 2, 2, 2, 1], 3, rng)
3-element Vector{Vector{Int64}}:
 [5, 6]
 [4, 1]
 [3, 2]
```
"""
function random_batch_indices(
    strata::AbstractVector{<:Integer},
    num_batches::Integer,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)

    n_samples = length(strata)

    if num_batches < 1
        throw(ArgumentError("Number of batches must be at least 1."))
    end
    if num_batches > n_samples
        throw(
            ArgumentError(
                "Number of batches ($num_batches) exceeds number of samples ($n_samples).",
            ),
        )
    end

    strata_groups =
        Dict(stratum => findall(==(stratum), strata) for stratum in unique(strata))

    batches = [Int[] for _ = 1:num_batches]

    for (stratum, indices) in strata_groups
        shuffled = shuffle(rng, indices)
        n = length(shuffled)
        if !(n % num_batches ≈ 0)
            @info (
                "Stratum $stratum (size = $n) not evenly divisible by " *
                "$num_batches batches."
            )
        end
        for (i, idx) in enumerate(shuffled)
            push!(batches[mod1(i, num_batches)], idx)
        end
    end

    batches
end

const CV_REGRESSION_MESSAGE = "CPPLS cross-validation currently supports only discriminant-analysis responses because the evaluation metric (NMC) is classification-specific."

cv_regression_error(caller::AbstractString) =
    throw(ArgumentError("$caller: $CV_REGRESSION_MESSAGE"))

"""
    CPPLS.optimize_num_latent_variables(
        X_train_full::AbstractMatrix{<:Real},
        Y_train_full::AbstractMatrix{<:Integer},
        max_components::Integer,
        num_inner_folds::Integer,
        num_inner_folds_repeats::Integer,
        gamma::Union{<:Real, <:NTuple{2,<:Real}, <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}},
        observation_weights::Union{AbstractVector{<:Real},Nothing},
        Y_auxiliary::Union{AbstractMatrix{<:Real},Nothing},
        center::Bool,
        X_tolerance::Real,
        X_loading_weight_tolerance::Real,
        t_squared_norm_tolerance::Real,
        gamma_rel_tol::Real,
        gamma_abs_tol::Real,
        weighted_nmc::Bool,
        rng::AbstractRNG,
        verbose::Bool)

Repeated inner cross-validation used inside `nested_cv` to pick the component
count. Argument summary:

- `X_train_full`, `Y_train_full`: numeric training matrices (observations × features/targets).
  `Y_train_full` must be an integer one-hot matrix for discriminant analysis. Wrapper
  methods accept categorical label vectors and convert them automatically.
- `max_components`: `Int` upper bound on components to evaluate (≥ 1).
- `num_inner_folds`, `num_inner_folds_repeats`: integers controlling stratified
  folds drawn via `random_batch_indices`.
- `gamma`: either a scalar γ, a `(lo, hi)` tuple of `Real`s, or a vector mixing
  both; forwarded to `fit_cppls_light`. Scalars keep γ fixed for every component,
  while tuples/vectors let each component pick the best γ from the shared
  candidate set.
- `observation_weights`: optional weight vector matching the training rows.
- `Y_auxiliary`: optional auxiliary response matrix aligned with `Y_train_full`.
- `center`: `Bool` toggling mean-centering in the inner fits.
- `X_tolerance`, `X_loading_weight_tolerance`, `t_squared_norm_tolerance`,
  `gamma_rel_tol`, `gamma_abs_tol`: `Real` tolerances passed to the fitter.
- `weighted_nmc`: choose class-weighted misclassification cost (`true` by default).
- `rng`: random-number generator used for shuffling.
- `verbose`: when `true`, prints per-fold diagnostics.
# Example
```
julia> using Random

julia> X = rand(MersenneTwister(1), 12, 4);

julia> labels = repeat(["red", "blue", "green"], 4);

julia> Y, _ = labels_to_one_hot(labels);

julia> k = CPPLS.optimize_num_latent_variables(
             X, Y,
             2,
             3, 3,
             0.5,
             nothing, nothing,
             true,
             1e-12, eps(Float64), 1e-10,
             1e-6, 1e-12,
             true,
             MersenneTwister(2),
             false);
[ Info: Stratum 2 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 3 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 1 (size = 4) not evenly divisible by 3 batches.
julia> k
1
```
For every inner repeat the routine fits a `CPPLSLight` with `max_components`,
scores validation folds for each partial component count using `predict` +
`predictonehot`, evaluates `nmc`, records the argmin, and finally returns the
median winning component number (rounded down) across repeats.
"""
function optimize_num_latent_variables(
    X_train_full::AbstractMatrix{<:Real},
    Y_train_full::AbstractMatrix{<:Integer},
    max_components::Integer,
    num_inner_folds::Integer,
    num_inner_folds_repeats::Integer,
    gamma::Union{
        <:Real,
        <:NTuple{2,<:Real},
        <:AbstractVector{<:Union{<:Real,<:NTuple{2,<:Real}}},
    },
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    Y_auxiliary::Union{AbstractMatrix{<:Real},Nothing},
    center::Bool,
    X_tolerance::Real,
    X_loading_weight_tolerance::Real,
    t_squared_norm_tolerance::Real,
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
    weighted_nmc::Bool,
    rng::AbstractRNG,
    verbose::Bool,
)

    n_samples = size(X_train_full, 1)

    class_labels = one_hot_to_labels(Y_train_full)
    inner_folds = random_batch_indices(class_labels, num_inner_folds, rng)

    best_num_latent_vars_per_fold = Vector{Int}(undef, num_inner_folds)

    for inner_fold_idx = 1:num_inner_folds_repeats

        test_indices = inner_folds[inner_fold_idx]

        verbose && println("  Inner fold: ", inner_fold_idx, " / ", num_inner_folds)

        @views X_validation = X_train_full[test_indices, :]
        @views Y_validation = Y_train_full[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X_train_full[train_indices, :]
        @views Y_train = Y_train_full[train_indices, :]

        Y_auxiliary_train =
            Y_auxiliary !== nothing ? Y_auxiliary[train_indices, :] : Y_auxiliary

        misclassification_costs = Vector{Float64}(undef, max_components)

        model = fit_cppls_light(
            X_train,
            Y_train,
            max_components,
            gamma = gamma,
            observation_weights = observation_weights,
            Y_auxiliary = Y_auxiliary_train,
            center = center,
            X_tolerance = X_tolerance,
            X_loading_weight_tolerance = X_loading_weight_tolerance,
            t_squared_norm_tolerance = t_squared_norm_tolerance,
            gamma_rel_tol = gamma_rel_tol,
            gamma_abs_tol = gamma_abs_tol,
        )

        for (num_components_idx, num_components) in enumerate(1:max_components)
            Y_pred = predictonehot(model, predict(model, X_validation, num_components))
            misclassification_costs[num_components_idx] =
                nmc(Y_validation, Y_pred, weighted_nmc)
        end

        best_num_latent_vars_per_fold[inner_fold_idx] = argmin(misclassification_costs)
        verbose && println(
            "    Best number of latent variables in fold ",
            inner_fold_idx,
            ": ",
            best_num_latent_vars_per_fold[inner_fold_idx],
        )
    end

    best_num_latent_vars = floor(Int, median(best_num_latent_vars_per_fold))
    verbose &&
        println("Best number of latent variables across folds: ", best_num_latent_vars)
    best_num_latent_vars
end

function optimize_num_latent_variables(
    X_train_full::AbstractMatrix{<:Real},
    Y_train_full::AbstractMatrix{<:Real},
    max_components::Integer,
    num_inner_folds::Integer,
    num_inner_folds_repeats::Integer,
    gamma::Union{
        <:Real,
        <:NTuple{2,<:Real},
        <:AbstractVector{<:Union{<:Real,<:NTuple{2,<:Real}}},
    },
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    Y_auxiliary::Union{AbstractMatrix{<:Real},Nothing},
    center::Bool,
    X_tolerance::Real,
    X_loading_weight_tolerance::Real,
    t_squared_norm_tolerance::Real,
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
    weighted_nmc::Bool,
    rng::AbstractRNG,
    verbose::Bool,
)
    cv_regression_error("optimize_num_latent_variables")
end

function optimize_num_latent_variables(
    X_train_full::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U},
    max_components::Integer,
    num_inner_folds::Integer,
    num_inner_folds_repeats::Integer,
    gamma::Union{
        <:Real,
        <:NTuple{2,<:Real},
        <:AbstractVector{<:Union{<:Real,<:NTuple{2,<:Real}}},
    },
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    Y_auxiliary::Union{AbstractMatrix{<:Real},Nothing},
    center::Bool,
    X_tolerance::Real,
    X_loading_weight_tolerance::Real,
    t_squared_norm_tolerance::Real,
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
    weighted_nmc::Bool,
    rng::AbstractRNG,
    verbose::Bool,
) where {T,R,V,C,U}
    Y_train_full, _ = labels_to_one_hot(labels)
    optimize_num_latent_variables(
        X_train_full,
        Y_train_full,
        max_components,
        num_inner_folds,
        num_inner_folds_repeats,
        gamma,
        observation_weights,
        Y_auxiliary,
        center,
        X_tolerance,
        X_loading_weight_tolerance,
        t_squared_norm_tolerance,
        gamma_rel_tol,
        gamma_abs_tol,
        weighted_nmc,
        rng,
        verbose,
    )
end

function optimize_num_latent_variables(
    X_train_full::AbstractMatrix{<:Real},
    labels::AbstractVector,
    max_components::Integer,
    num_inner_folds::Integer,
    num_inner_folds_repeats::Integer,
    gamma::Union{
        <:Real,
        <:NTuple{2,<:Real},
        <:AbstractVector{<:Union{<:Real,<:NTuple{2,<:Real}}},
    },
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    Y_auxiliary::Union{AbstractMatrix{<:Real},Nothing},
    center::Bool,
    X_tolerance::Real,
    X_loading_weight_tolerance::Real,
    t_squared_norm_tolerance::Real,
    gamma_rel_tol::Real,
    gamma_abs_tol::Real,
    weighted_nmc::Bool,
    rng::AbstractRNG,
    verbose::Bool,
)
    if eltype(labels) <: Real
        cv_regression_error("optimize_num_latent_variables")
    else
        Y_train_full, _ = labels_to_one_hot(labels)
        optimize_num_latent_variables(
            X_train_full,
            Y_train_full,
            max_components,
            num_inner_folds,
            num_inner_folds_repeats,
            gamma,
            observation_weights,
            Y_auxiliary,
            center,
            X_tolerance,
            X_loading_weight_tolerance,
            t_squared_norm_tolerance,
            gamma_rel_tol,
            gamma_abs_tol,
            weighted_nmc,
            rng,
            verbose,
        )
    end
end

"""
    nested_cv(X_predictors::AbstractMatrix{<:Real}, Y_responses::AbstractMatrix{<:Integer};
        gamma::Union{<:Real, <:NTuple{2, <:Real}, <:AbstractVector{<:Union{<:Real,<:NTuple{2, <:Real}}}}=0.5,
        observation_weights::Union{AbstractVector{<:Real},Nothing}=nothing,
        Y_auxiliary::Union{AbstractMatrix{<:Real},Nothing}=nothing,
        center::Bool=true,
        X_tolerance::Real=1e-12,
        X_loading_weight_tolerance::Real=eps(Float64),
        t_squared_norm_tolerance::Real=1e-10,
        gamma_rel_tol::Real=1e-6,
        gamma_abs_tol::Real=1e-12,
        num_outer_folds::Integer=8,
        num_outer_folds_repeats::Integer=num_outer_folds,
        num_inner_folds::Integer=7,
        num_inner_folds_repeats::Integer=num_inner_folds,
        max_components::Integer=5,
        weighted_nmc::Bool=true,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true)

Top-level nested CV driver for CPPLS. Parameter overview:

- `X_predictors`, `Y_responses`: feature and one-hot response matrices (integer-valued).
  Wrapper methods accept categorical label vectors. Regression responses (real-valued
  vectors or matrices) are currently unsupported because only the NMC metric is
  implemented.
- `gamma`: either a scalar γ, a `(lo, hi)` tuple, or a vector of mixed candidates
  passed to `fit_cppls_light`. Scalars enforce a single γ for all components,
  whereas tuples/vectors share candidate ranges from which each component selects
  its own optimum.
- `observation_weights`: optional sample weights; `Y_auxiliary`: extra response features.
- `center`: toggle mean-centering; tolerances control numerical stability inside
  inner fits; `weighted_nmc` chooses between weighted/unweighted misclassification cost.
- `num_outer_folds`, `num_outer_folds_repeats`: number of outer stratified folds
  and how many to evaluate (≤ `num_outer_folds`).
- `num_inner_folds`, `num_inner_folds_repeats`: same for inner CV.
- `max_components`: maximum latent components considered by the inner loop.
- `rng`: random generator shared across fold shuffles; `verbose`: emit progress.

For every outer fold the data is split into training/test partitions, an inner
CV loop selects the optimal component count via `optimize_num_latent_variables`,
the final `CPPLSLight` is fit on the outer training data with that count, and
`accuracy = 1 - nmc` is computed on the outer test split. Returns a tuple
`(outer_fold_accuracies, optimal_num_latent_variables)`.

# Example
```
julia> using Random

julia> X = rand(MersenneTwister(1), 12, 4);

julia> labels = repeat(["red", "blue", "green"], 4);

julia> Y, _ = labels_to_one_hot(labels);

julia> accs, comps = nested_cv(
           X, Y;
           gamma=0.5,
           num_outer_folds=3,
           num_inner_folds=2,
           max_components=2,
           rng=MersenneTwister(2),
           verbose=false);
[ Info: Stratum 2 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 3 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 1 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 2 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 3 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 1 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 2 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 3 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 1 (size = 3) not evenly divisible by 2 batches.
julia> accs ≈ [1.1102230246251565e-16, 0.0, 0.0]
true

julia> comps ≈ [1, 1, 1]
true
```
"""
function nested_cv(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Integer};
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_auxiliary::Union{AbstractMatrix{T3},Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    t_squared_norm_tolerance::Real = 1e-10,
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    num_outer_folds::Integer = 8,
    num_outer_folds_repeats::Integer = num_outer_folds,
    num_inner_folds::Integer = 7,
    num_inner_folds_repeats::Integer = num_inner_folds,
    max_components::Integer = 5,
    weighted_nmc::Bool = true,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    verbose::Bool = true,
) where {T1<:Real,T2<:Real,T3<:Real}

    num_outer_folds_repeats ≤ num_outer_folds || throw(
        ArgumentError(
            "The number of outer fold repeats cannot exceed the number of outer folds",
        ),
    )

    num_inner_folds_repeats ≤ num_inner_folds || throw(
        ArgumentError(
            "The number of inner fold repeats cannot exceed the number of inner folds",
        ),
    )

    max_components > 0 ||
        throw(ArgumentError("The number of components must be greater than zero"))

    n_samples = size(X_predictors, 1)
    class_labels = one_hot_to_labels(Y_responses)
    outer_folds = random_batch_indices(class_labels, num_outer_folds, rng)

    outer_fold_accuracies = Vector{Float64}(undef, num_outer_folds_repeats)
    optimal_num_latent_variables = Vector{Int}(undef, num_outer_folds_repeats)

    for outer_fold_idx = 1:num_outer_folds_repeats

        test_indices = outer_folds[outer_fold_idx]

        verbose && println("Outer fold: ", outer_fold_idx, " / ", num_outer_folds_repeats)

        @views X_test = X_predictors[test_indices, :]
        @views Y_test = Y_responses[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X_predictors[train_indices, :]
        @views Y_train = Y_responses[train_indices, :]

        Y_auxiliary_train =
            Y_auxiliary !== nothing ? Y_auxiliary[train_indices, :] : Y_auxiliary

        optimal_num_latent_variables[outer_fold_idx] = optimize_num_latent_variables(
            X_train,
            Y_train,
            max_components,
            num_inner_folds,
            num_inner_folds_repeats,
            gamma,
            observation_weights,
            Y_auxiliary_train,
            center,
            X_tolerance,
            X_loading_weight_tolerance,
            t_squared_norm_tolerance,
            gamma_rel_tol,
            gamma_abs_tol,
            weighted_nmc,
            rng,
            verbose,
        )

        final_model = fit_cppls_light(
            X_train,
            Y_train,
            optimal_num_latent_variables[outer_fold_idx],
            gamma = gamma,
            observation_weights = observation_weights,
            Y_auxiliary = Y_auxiliary_train,
            center = center,
            X_tolerance = X_tolerance,
            X_loading_weight_tolerance = X_loading_weight_tolerance,
            t_squared_norm_tolerance = t_squared_norm_tolerance,
            gamma_rel_tol = gamma_rel_tol,
            gamma_abs_tol = gamma_abs_tol,
        )

        predicted_labels = predictonehot(final_model, predict(final_model, X_test))

        outer_fold_accuracies[outer_fold_idx] =
            1 - nmc(predicted_labels, Y_test, weighted_nmc)

        verbose && println(
            "Accuracy for outer fold: ",
            outer_fold_accuracies[outer_fold_idx],
            "\n",
        )
    end

    outer_fold_accuracies, optimal_num_latent_variables
end

function nested_cv(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real};
    kwargs...,
)
    cv_regression_error("nested_cv")
end

function nested_cv(
    X_predictors::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...,
) where {T,R,V,C,U}
    Y_responses, _ = labels_to_one_hot(labels)
    nested_cv(X_predictors, Y_responses; kwargs...)
end

function nested_cv(X_predictors::AbstractMatrix{<:Real}, labels::AbstractVector; kwargs...)
    if eltype(labels) <: Real
        cv_regression_error("nested_cv")
    else
        Y_responses, _ = labels_to_one_hot(labels)
        nested_cv(X_predictors, Y_responses; kwargs...)
    end
end

"""
    nested_cv_permutation(X_predictors::AbstractMatrix{<:Real}, Y_responses::AbstractMatrix{<:Integer};
        gamma::Union{<:Real, <:NTuple{2,<:Real}, <:AbstractVector{<:Union{<:Real,<:NTuple{2,<:Real}}}}=0.5,
        observation_weights::Union{AbstractVector{<:Real},Nothing}=nothing,
        Y_auxiliary::Union{AbstractMatrix{<:Real},Nothing}=nothing,
        center::Bool=true,
        X_tolerance::Real=1e-12,
        X_loading_weight_tolerance::Real=eps(Float64),
        t_squared_norm_tolerance::Real=1e-10,
        gamma_rel_tol::Real=1e-6,
        gamma_abs_tol::Real=1e-12,
        num_outer_folds::Integer=9,
        num_outer_folds_repeats::Integer=num_outer_folds,
        num_inner_folds::Integer=8,
        num_inner_folds_repeats::Integer=num_inner_folds,
        max_components::Integer=5,
        weighted_nmc::Bool=true,
        num_permutations::Integer=999,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true)

Permutation-based significance test for the nested CV pipeline. Keywords mirror
`nested_cv` but with explicit defaults suited for permutation tests. Parameter
summary:

- `gamma`, `observation_weights`, `Y_auxiliary`, `center`, tolerances,
  `weighted_nmc`: forwarded directly into each nested CV call. Only discriminant
  responses (integer one-hot matrices or categorical vectors) are supported because
  the scoring metric is classification-specific.
- `num_outer_folds`, `num_outer_folds_repeats`, `num_inner_folds`,
  `num_inner_folds_repeats`, `max_components`: control the inner/outer fold
  geometry per permutation.
- `num_permutations`: number of label shuffles (≥ 1).
- `rng`: governs shuffling of both labels and folds; `verbose`: prints progress.

For each permutation, the rows of `Y_responses` are randomly shuffled, then
`nested_cv` is executed with the same hyperparameters, and the mean outer-fold
accuracy is recorded. Returns a vector of length `num_permutations` containing
those mean accuracies so you can compute empirical p-values against the
unpermuted nested-CV accuracy.

# Example
```
julia> using Random

julia> X = rand(MersenneTwister(1), 12, 4);

julia> labels = repeat(["red", "blue", "green"], 4);

julia> Y, classes = labels_to_one_hot(labels);

julia> perms = nested_cv_permutation(X, Y;
                 gamma=0.5,
                 num_outer_folds=3,
                 num_inner_folds=2,
                 max_components=2,
                 num_permutations=2,
                 verbose=false,
                 rng=MersenneTwister(2));
[ Info: Stratum 2 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 3 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 1 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 2 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 3 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 1 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 2 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 3 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 1 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 2 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 3 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 1 (size = 4) not evenly divisible by 3 batches.
[ Info: Stratum 2 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 3 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 1 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 2 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 3 (size = 3) not evenly divisible by 2 batches.
[ Info: Stratum 1 (size = 3) not evenly divisible by 2 batches.

julia> perms ≈ [0.3055555555555556, 0.22222222222222224]
true
```
"""
function nested_cv_permutation(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Integer};
    gamma::Union{<:T1,<:NTuple{2,T1},<:AbstractVector{<:Union{<:T1,<:NTuple{2,T1}}}} = 0.5,
    observation_weights::Union{AbstractVector{T2},Nothing} = nothing,
    Y_auxiliary::Union{AbstractMatrix{T3},Nothing} = nothing,
    center::Bool = true,
    X_tolerance::Real = 1e-12,
    X_loading_weight_tolerance::Real = eps(Float64),
    t_squared_norm_tolerance::Real = 1e-10,
    gamma_rel_tol::Real = 1e-6,
    gamma_abs_tol::Real = 1e-12,
    num_outer_folds::Integer = 9,
    num_outer_folds_repeats::Integer = num_outer_folds,
    num_inner_folds::Integer = 8,
    num_inner_folds_repeats::Integer = num_inner_folds,
    max_components::Integer = 5,
    weighted_nmc::Bool = true,
    num_permutations::Integer = 999,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    verbose::Bool = true,
) where {T1<:Real,T2<:Real,T3<:Real}

    num_outer_folds_repeats ≤ num_outer_folds || throw(
        ArgumentError(
            "The number of outer fold repeats cannot exceed the number of outer folds",
        ),
    )

    num_inner_folds_repeats ≤ num_inner_folds || throw(
        ArgumentError(
            "The number of inner fold repeats cannot exceed the number of inner folds",
        ),
    )

    max_components > 0 ||
        throw(ArgumentError("The number of components must be greater than zero"))

    n_samples = size(X_predictors, 1)
    permutation_accuracies = Vector{Float64}(undef, num_permutations)

    for i = 1:num_permutations
        verbose && println("Permutation: ", i, " / ", num_permutations)

        shuffled_indices = shuffle(1:n_samples)
        shuffled_Y_responses = @view Y_responses[shuffled_indices, :]

        outer_fold_accuracies, _ = nested_cv(
            X_predictors,
            shuffled_Y_responses;
            gamma = gamma,
            observation_weights = observation_weights,
            Y_auxiliary = Y_auxiliary,
            center = center,
            X_tolerance = X_tolerance,
            X_loading_weight_tolerance = X_loading_weight_tolerance,
            t_squared_norm_tolerance = t_squared_norm_tolerance,
            gamma_rel_tol = gamma_rel_tol,
            gamma_abs_tol = gamma_abs_tol,
            num_outer_folds = num_outer_folds,
            num_outer_folds_repeats = num_outer_folds_repeats,
            num_inner_folds = num_inner_folds,
            num_inner_folds_repeats = num_inner_folds_repeats,
            max_components = max_components,
            weighted_nmc = weighted_nmc,
            rng = rng,
            verbose = verbose,
        )

        permutation_accuracies[i] = mean(outer_fold_accuracies)
    end
    permutation_accuracies
end

function nested_cv_permutation(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real};
    kwargs...,
)
    cv_regression_error("nested_cv_permutation")
end

function nested_cv_permutation(
    X_predictors::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...,
) where {T,R,V,C,U}
    Y_responses, _ = labels_to_one_hot(labels)
    nested_cv_permutation(X_predictors, Y_responses; kwargs...)
end

function nested_cv_permutation(
    X_predictors::AbstractMatrix{<:Real},
    labels::AbstractVector;
    kwargs...,
)
    if eltype(labels) <: Real
        cv_regression_error("nested_cv_permutation")
    else
        Y_responses, _ = labels_to_one_hot(labels)
        nested_cv_permutation(X_predictors, Y_responses; kwargs...)
    end
end
