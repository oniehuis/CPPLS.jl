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

    # Each fold must have at least 2 samples per stratum to make variance estimates meaningful.
    # This requires num_batches <= floor(min_stratum_size / 2).
    min_stratum_size = minimum(length, values(strata_groups))
    if num_batches > fld(min_stratum_size, 2)
        throw(
            ArgumentError(
                "Number of batches ($num_batches) is too large for the smallest stratum " *
                "(size = $min_stratum_size). Each fold must have at least 2 samples per " *
                "stratum, so num_batches must be ≤ $(fld(min_stratum_size, 2)).",
            ),
        )
    end

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

function resolve_observation_weights(
    observation_weights::Union{AbstractVector{<:Real},Nothing},
    observation_weights_fn::Union{Function,Nothing},
    labels_train::AbstractVector,
    train_indices::AbstractVector{<:Integer},
    total_samples::Integer,
)
    if observation_weights_fn !== nothing
        observation_weights === nothing || throw(
            ArgumentError(
                "Provide either observation_weights or observation_weights_fn, not both.",
            ),
        )
        weights = observation_weights_fn(labels_train)
        length(weights) == length(labels_train) || throw(
            ArgumentError(
                "observation_weights_fn must return a vector matching the number of training samples.",
            ),
        )
        return weights
    end

    if observation_weights === nothing
        return nothing
    end

    if length(observation_weights) == total_samples
        return observation_weights[train_indices]
    end

    length(observation_weights) == length(train_indices) || throw(
        ArgumentError(
            "Length of observation_weights must match the total sample count or the number of training samples.",
        ),
    )
    observation_weights
end

function with_n_components(spec::CPPLSSpec, n_components::Integer)
    return CPPLSSpec(
        n_components = n_components,
        gamma = spec.gamma,
        center = spec.center,
        X_tolerance = spec.X_tolerance,
        X_loading_weight_tolerance = spec.X_loading_weight_tolerance,
        t_squared_norm_tolerance = spec.t_squared_norm_tolerance,
        gamma_rel_tol = spec.gamma_rel_tol,
        gamma_abs_tol = spec.gamma_abs_tol,
        analysis_mode = spec.analysis_mode,
    )
end

function subset_vector_like(
    values,
    train_indices::AbstractVector{<:Integer},
    n_samples::Integer,
    name::Symbol,
)
    values === nothing && return values
    values isa AbstractVector || return values

    if length(values) == n_samples
        return values[train_indices]
    elseif length(values) == length(train_indices)
        return values
    end

    throw(
        DimensionMismatch(
            "Length of $name must match the total sample count or the number of training samples.",
        ),
    )
end

function subset_matrix_like(
    values,
    train_indices::AbstractVector{<:Integer},
    n_samples::Integer,
    name::Symbol,
)
    values === nothing && return values
    values isa AbstractVector && return subset_vector_like(values, train_indices, n_samples, name)
    values isa AbstractMatrix || return values

    if size(values, 1) == n_samples
        return values[train_indices, :]
    elseif size(values, 1) == length(train_indices)
        return values
    end

    throw(
        DimensionMismatch(
            "Row count of $name must match the total sample count or the number of training samples.",
        ),
    )
end

function subset_fit_kwargs(
    fit_kwargs::NamedTuple,
    train_indices::AbstractVector{<:Integer},
    n_samples::Integer,
)
    isempty(fit_kwargs) && return fit_kwargs

    out_pairs = Pair{Symbol,Any}[]
    for (key, value) in Base.pairs(fit_kwargs)
        adjusted = if key in (:observation_weights, :sample_labels, :sample_classes)
            subset_vector_like(value, train_indices, n_samples, key)
        elseif key in (:Y_aux, :Y_auxiliary)
            subset_matrix_like(value, train_indices, n_samples, key)
        else
            value
        end
        push!(out_pairs, key => adjusted)
    end
    return (; out_pairs...)
end

function ensure_response_labels(
    fit_kwargs::NamedTuple,
    Y::AbstractMatrix{<:Real},
)
    haskey(fit_kwargs, :response_labels) && return fit_kwargs
    labels = string.(1:size(Y, 2))
    return merge(fit_kwargs, (response_labels = labels,))
end

function build_folds(
    n_samples::Integer,
    num_folds::Integer,
    rng::AbstractRNG;
    strata::Union{AbstractVector{<:Integer},Nothing} = nothing,
)
    if num_folds < 1
        throw(ArgumentError("Number of folds must be at least 1."))
    end
    if num_folds > n_samples
        throw(
            ArgumentError(
                "Number of folds ($num_folds) exceeds number of samples ($n_samples).",
            ),
        )
    end

    if strata === nothing
        indices = shuffle(rng, collect(1:n_samples))
        base = fld(n_samples, num_folds)
        extra = n_samples % num_folds
        folds = Vector{Vector{Int}}(undef, num_folds)
        start = 1
        for i in 1:num_folds
            n_take = base + (i <= extra ? 1 : 0)
            stop = start + n_take - 1
            folds[i] = indices[start:stop]
            start = stop + 1
        end
        return folds
    end

    length(strata) == n_samples || throw(
        DimensionMismatch("Length of strata must match the number of samples."),
    )
    return random_batch_indices(strata, num_folds, rng)
end

"""
    cv_classification(; weighted=true)

Return a named tuple with `score_fn`, `predict_fn`, `select_fn`, and `flag_fn`
implementations for CPPLS classification (NMC-based scoring). These assume
one-hot response matrices and CPPLS fits created in `:discriminant` mode.
"""
function cv_classification(; weighted::Bool = true)
    score_fn = (Y_true, Y_pred) -> 1 - nmc(Y_pred, Y_true, weighted)
    predict_fn = (model, X, k) -> predictonehot(model, predict(model, X, k))
    select_fn = argmax
    flag_fn =
        (Y_true, Y_pred) ->
            one_hot_to_labels(Y_pred) .!= one_hot_to_labels(Y_true)
    return (
        score_fn = score_fn,
        predict_fn = predict_fn,
        select_fn = select_fn,
        flag_fn = flag_fn,
    )
end

"""
    CPPLS.optimize_num_latent_variables(
        X_train_full::AbstractMatrix{<:Real},
        Y_train_full::AbstractMatrix{<:Real},
        max_components::Integer,
        num_inner_folds::Integer,
        num_inner_folds_repeats::Integer,
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple,
        score_fn::Function,
        predict_fn::Function,
        select_fn::Function,
        rng::AbstractRNG,
        verbose::Bool;
        strata::Union{AbstractVector{<:Integer},Nothing}=nothing)

Repeated inner cross-validation used inside `nested_cv` to pick the component
count. `score_fn`, `predict_fn`, and `select_fn` define how models are evaluated.
"""
function optimize_num_latent_variables(
    X_train_full::AbstractMatrix{<:Real},
    Y_train_full::AbstractMatrix{<:Real},
    max_components::Integer,
    num_inner_folds::Integer,
    num_inner_folds_repeats::Integer,
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple,
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    rng::AbstractRNG,
    verbose::Bool;
    strata::Union{AbstractVector{<:Integer},Nothing} = nothing,
)

    max_components > 0 ||
        throw(ArgumentError("The number of components must be greater than zero"))
    num_inner_folds_repeats ≤ num_inner_folds || throw(
        ArgumentError(
            "The number of inner fold repeats cannot exceed the number of inner folds",
        ),
    )

    n_samples = size(X_train_full, 1)
    size(Y_train_full, 1) == n_samples || throw(
        DimensionMismatch("Row count mismatch between X_train_full and Y_train_full"),
    )

    inner_folds =
        build_folds(n_samples, num_inner_folds, rng; strata = strata)

    best_num_latent_vars_per_fold = Vector{Int}(undef, num_inner_folds_repeats)

    for inner_fold_idx = 1:num_inner_folds_repeats
        test_indices = inner_folds[inner_fold_idx]

        verbose && println("  Inner fold: ", inner_fold_idx, " / ", num_inner_folds_repeats)

        @views X_validation = X_train_full[test_indices, :]
        @views Y_validation = Y_train_full[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X_train_full[train_indices, :]
        @views Y_train = Y_train_full[train_indices, :]

        fold_kwargs = subset_fit_kwargs(fit_kwargs, train_indices, n_samples)
        if spec.analysis_mode === :discriminant
            fold_kwargs = ensure_response_labels(fold_kwargs, Y_train)
        end
        spec_max = with_n_components(spec, max_components)
        model = fit(spec_max, X_train, Y_train; fold_kwargs...)

        scores = Vector{Float64}(undef, max_components)
        for k in 1:max_components
            Y_pred = predict_fn(model, X_validation, k)
            score = score_fn(Y_validation, Y_pred)
            score isa Real || throw(ArgumentError("score_fn must return a Real"))
            scores[k] = score
        end

        best_k = select_fn(scores)
        (1 ≤ best_k ≤ max_components) || throw(
            ArgumentError("select_fn must return an integer between 1 and $max_components"),
        )
        best_num_latent_vars_per_fold[inner_fold_idx] = best_k
    end

    floor(Int, median(best_num_latent_vars_per_fold))
end

"""
    nested_cv(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple=(;),
        score_fn::Function,
        predict_fn::Function,
        select_fn::Function,
        num_outer_folds::Integer=8,
        num_outer_folds_repeats::Integer=num_outer_folds,
        num_inner_folds::Integer=7,
        num_inner_folds_repeats::Integer=num_inner_folds,
        max_components::Integer=spec.n_components,
        strata::Union{AbstractVector{<:Integer},Nothing}=nothing,
        reshuffle_outer_folds::Bool=false,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true)

Explicit nested cross-validation. The caller supplies `score_fn`, `predict_fn`, and
`select_fn` so the routine is agnostic to regression vs classification. Use
`spec` + `fit_kwargs` to configure CPPLS fitting.

If `spec.analysis_mode == :discriminant` and `response_labels` are not provided in
`fit_kwargs`, default labels `"1"`, `"2"`, … are injected to satisfy the CPPLS
fit metadata requirements.
"""
function nested_cv(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple = (;),
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    num_outer_folds::Integer = 8,
    num_outer_folds_repeats::Integer = num_outer_folds,
    num_inner_folds::Integer = 7,
    num_inner_folds_repeats::Integer = num_inner_folds,
    max_components::Integer = spec.n_components,
    strata::Union{AbstractVector{<:Integer},Nothing} = nothing,
    reshuffle_outer_folds::Bool = false,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    verbose::Bool = true,
)

    num_outer_folds_repeats > 0 ||
        throw(ArgumentError("The number of outer folds must be greater than zero"))
    num_inner_folds_repeats > 0 ||
        throw(ArgumentError("The number of inner folds must be greater than zero"))

    reshuffle_outer_folds || num_outer_folds_repeats ≤ num_outer_folds || throw(
        ArgumentError(
            "The number of outer fold repeats cannot exceed the number of outer folds unless reshuffle_outer_folds=true",
        ),
    )

    num_inner_folds_repeats ≤ num_inner_folds || throw(
        ArgumentError(
            "The number of inner fold repeats cannot exceed the number of inner folds",
        ),
    )

    max_components > 0 ||
        throw(ArgumentError("The number of components must be greater than zero"))

    n_samples = size(X, 1)
    size(Y, 1) == n_samples || throw(
        DimensionMismatch("Row count mismatch between X and Y"),
    )
    strata === nothing || length(strata) == n_samples || throw(
        DimensionMismatch("Length of strata must match the number of samples."),
    )

    outer_fold_scores = Vector{Float64}(undef, num_outer_folds_repeats)
    optimal_num_latent_variables = Vector{Int}(undef, num_outer_folds_repeats)

    outer_folds = reshuffle_outer_folds ? nothing :
                  build_folds(n_samples, num_outer_folds, rng; strata = strata)

    for outer_fold_idx = 1:num_outer_folds_repeats
        folds = reshuffle_outer_folds ?
                build_folds(n_samples, num_outer_folds, rng; strata = strata) :
                outer_folds

        test_indices = reshuffle_outer_folds ? folds[1] : folds[outer_fold_idx]

        verbose && println("Outer fold: ", outer_fold_idx, " / ", num_outer_folds_repeats)

        @views X_test = X[test_indices, :]
        @views Y_test = Y[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X[train_indices, :]
        @views Y_train = Y[train_indices, :]

        fold_kwargs = subset_fit_kwargs(fit_kwargs, train_indices, n_samples)
        inner_strata = strata === nothing ? nothing : strata[train_indices]
        if spec.analysis_mode === :discriminant
            fold_kwargs = ensure_response_labels(fold_kwargs, Y_train)
        end

        optimal_num_latent_variables[outer_fold_idx] = optimize_num_latent_variables(
            X_train,
            Y_train,
            max_components,
            num_inner_folds,
            num_inner_folds_repeats,
            spec,
            fold_kwargs,
            score_fn,
            predict_fn,
            select_fn,
            rng,
            verbose;
            strata = inner_strata,
        )

        spec_k = with_n_components(spec, optimal_num_latent_variables[outer_fold_idx])
        final_model = fit(spec_k, X_train, Y_train; fold_kwargs...)

        Y_pred = predict_fn(final_model, X_test, optimal_num_latent_variables[outer_fold_idx])
        score = score_fn(Y_test, Y_pred)
        score isa Real || throw(ArgumentError("score_fn must return a Real"))
        outer_fold_scores[outer_fold_idx] = score

        verbose && println("Score for outer fold: ", outer_fold_scores[outer_fold_idx], "\n")
    end

    outer_fold_scores, optimal_num_latent_variables
end

"""
    nested_cv_permutation(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple=(;),
        score_fn::Function,
        predict_fn::Function,
        select_fn::Function,
        num_permutations::Integer=999,
        num_outer_folds::Integer=8,
        num_outer_folds_repeats::Integer=num_outer_folds,
        num_inner_folds::Integer=7,
        num_inner_folds_repeats::Integer=num_inner_folds,
        max_components::Integer=spec.n_components,
        strata::Union{AbstractVector{<:Integer},Nothing}=nothing,
        reshuffle_outer_folds::Bool=false,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true)

Permutation-based significance test for the nested CV pipeline. Returns a vector
of mean scores (one per permutation).
"""
function nested_cv_permutation(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple = (;),
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    num_permutations::Integer = 999,
    num_outer_folds::Integer = 8,
    num_outer_folds_repeats::Integer = num_outer_folds,
    num_inner_folds::Integer = 7,
    num_inner_folds_repeats::Integer = num_inner_folds,
    max_components::Integer = spec.n_components,
    strata::Union{AbstractVector{<:Integer},Nothing} = nothing,
    reshuffle_outer_folds::Bool = false,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    verbose::Bool = true,
)

    num_permutations > 0 ||
        throw(ArgumentError("num_permutations must be greater than zero"))

    n_samples = size(X, 1)
    size(Y, 1) == n_samples || throw(
        DimensionMismatch("Row count mismatch between X and Y"),
    )
    strata === nothing || length(strata) == n_samples || throw(
        DimensionMismatch("Length of strata must match the number of samples."),
    )

    permutation_scores = Vector{Float64}(undef, num_permutations)

    for perm_idx = 1:num_permutations
        perm = randperm(rng, n_samples)
        Y_perm = Y[perm, :]
        strata_perm = strata === nothing ? nothing : strata[perm]

        verbose && println("Permutation: ", perm_idx, " / ", num_permutations)

        scores, _ = nested_cv(
            X,
            Y_perm;
            spec = spec,
            fit_kwargs = fit_kwargs,
            score_fn = score_fn,
            predict_fn = predict_fn,
            select_fn = select_fn,
            num_outer_folds = num_outer_folds,
            num_outer_folds_repeats = num_outer_folds_repeats,
            num_inner_folds = num_inner_folds,
            num_inner_folds_repeats = num_inner_folds_repeats,
            max_components = max_components,
            strata = strata_perm,
            reshuffle_outer_folds = reshuffle_outer_folds,
            rng = rng,
            verbose = verbose,
        )

        permutation_scores[perm_idx] = mean(scores)
    end

    permutation_scores
end

"""
    cv_outlier_scan(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple=(;),
        num_outer_folds::Integer=8,
        num_outer_folds_repeats::Integer=10 * num_outer_folds,
        num_inner_folds::Integer=7,
        num_inner_folds_repeats::Integer=num_inner_folds,
        max_components::Integer=spec.n_components,
        reshuffle_outer_folds::Bool=true,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true)

Convenience outlier scan for classification. Returns per-sample test counts and
misclassification counts plus the rate (`n_flagged ./ n_tested`). By default
`reshuffle_outer_folds = true` so samples are repeatedly re-tested across random
outer splits.
"""
function cv_outlier_scan(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple = (;),
    num_outer_folds::Integer = 8,
    num_outer_folds_repeats::Integer = 10 * num_outer_folds,
    num_inner_folds::Integer = 7,
    num_inner_folds_repeats::Integer = num_inner_folds,
    max_components::Integer = spec.n_components,
    reshuffle_outer_folds::Bool = true,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    verbose::Bool = true,
)
    spec.analysis_mode === :discriminant || throw(
        ArgumentError("cv_outlier_scan expects spec.analysis_mode = :discriminant"),
    )

    n_samples = size(X, 1)
    size(Y, 1) == n_samples || throw(
        DimensionMismatch("Row count mismatch between X and Y"),
    )

    cfg = cv_classification()
    strata = one_hot_to_labels(Y)

    n_tested = zeros(Int, n_samples)
    n_flagged = zeros(Int, n_samples)

    reshuffle_outer_folds || num_outer_folds_repeats ≤ num_outer_folds || throw(
        ArgumentError(
            "The number of outer fold repeats cannot exceed the number of outer folds unless reshuffle_outer_folds=true",
        ),
    )

    fixed_folds = reshuffle_outer_folds ? nothing :
                  build_folds(n_samples, num_outer_folds, rng; strata = strata)

    for outer_fold_idx = 1:num_outer_folds_repeats
        folds = reshuffle_outer_folds ?
                build_folds(n_samples, num_outer_folds, rng; strata = strata) :
                fixed_folds
        test_indices = reshuffle_outer_folds ? folds[1] : folds[outer_fold_idx]

        verbose && println("Outer fold: ", outer_fold_idx, " / ", num_outer_folds_repeats)

        @views X_test = X[test_indices, :]
        @views Y_test = Y[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X[train_indices, :]
        @views Y_train = Y[train_indices, :]

        fold_kwargs = subset_fit_kwargs(fit_kwargs, train_indices, n_samples)
        fold_kwargs = ensure_response_labels(fold_kwargs, Y_train)
        inner_strata = strata[train_indices]

        best_k = optimize_num_latent_variables(
            X_train,
            Y_train,
            max_components,
            num_inner_folds,
            num_inner_folds_repeats,
            spec,
            fold_kwargs,
            cfg.score_fn,
            cfg.predict_fn,
            cfg.select_fn,
            rng,
            verbose;
            strata = inner_strata,
        )

        spec_k = with_n_components(spec, best_k)
        final_model = fit(spec_k, X_train, Y_train; fold_kwargs...)

        Y_pred = cfg.predict_fn(final_model, X_test, best_k)
        flags = cfg.flag_fn(Y_test, Y_pred)
        length(flags) == length(test_indices) || throw(
            DimensionMismatch("flag_fn must return one flag per test sample"),
        )

        n_tested[test_indices] .+= 1
        n_flagged[test_indices] .+= flags
    end

    rate = n_flagged ./ max.(1, n_tested)
    (n_tested = n_tested, n_flagged = n_flagged, rate = rate)
end

function cv_outlier_scan(
    X::AbstractMatrix{<:Real},
    labels::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...,
) where {T,R,V,C,U}
    Y, _ = labels_to_one_hot(labels)
    cv_outlier_scan(X, Y; kwargs...)
end

function cv_outlier_scan(
    X::AbstractMatrix{<:Real},
    labels::AbstractVector;
    kwargs...,
)
    if eltype(labels) <: Real
        throw(ArgumentError("cv_outlier_scan expects categorical labels or one-hot Y."))
    end
    Y, _ = labels_to_one_hot(labels)
    cv_outlier_scan(X, Y; kwargs...)
end
