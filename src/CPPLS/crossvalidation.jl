"""
    CPPLS.random_batch_indices(
        strata::AbstractVector{<:Int},
        num_batches::Int, 
        rng::AbstractRNG=Random.GLOBAL_RNG
    )

Construct stratified batches from integer class labels in `strata`. Indices within each
stratum are shuffled with `rng` and then dealt round-robin into `num_batches` disjoint
vectors, which helps preserve class proportions across batches. An `ArgumentError` is
thrown if `num_batches` is less than `1`, exceeds the number of samples, or is too large
for the smallest stratum.

# Examples
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
    strata::AbstractVector{<:Int},
    num_batches::Int,
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    n_samples = length(strata)

    num_batches ≥ 1 || throw(ArgumentError("Number of batches must be at least 1."))
    num_batches ≤ n_samples || throw(ArgumentError(
        "Number of batches ($num_batches) exceeds number of samples ($n_samples)."))

    strata_groups =
        Dict(stratum => findall(==(stratum), strata) for stratum in unique(strata))

    # Each fold must have at least 2 samples per stratum to make variance estimates 
    # meaningful. This requires num_batches ≤ floor(min_stratum_size / 2).
    min_stratum_size = minimum(length, values(strata_groups))
    num_batches ≤ fld(min_stratum_size, 2) || throw(ArgumentError(
        "Number of batches ($num_batches) is too large for the smallest stratum " *
        "(size = $min_stratum_size). Each fold must have at least 2 samples per " *
        "stratum, so num_batches must be ≤ $(fld(min_stratum_size, 2))."))

    batches = [Int[] for _ = 1:num_batches]

    for (stratum, indices) in strata_groups
        shuffled = shuffle(rng, indices)
        n = length(shuffled)
        if !(n % num_batches ≈ 0)
            @info ("Stratum $stratum (size = $n) not evenly divisible by " *
                "$num_batches batches.")
        end
        for (i, idx) in enumerate(shuffled)
            push!(batches[mod1(i, num_batches)], idx)
        end
    end

    batches
end

############################################################################################
# Public Helpers
############################################################################################

"""
    cv_classification(; weighted::Bool=true)

Return a named tuple containing `score_fn`, `predict_fn`, `select_fn`, and `flag_fn`
suited to CPPLS classification with one-hot response matrices. The scoring rule is based
on nearest-mean classification and assumes models were fit in `:discriminant` mode.

This helper is meant to supply the callback interface expected by `nested_cv`,
`nested_cv_permutation`, and `cv_outlier_scan`. In particular, `score_fn`,
`predict_fn`, and `select_fn` can be passed directly to `nested_cv` and
`nested_cv_permutation`, while `flag_fn` is used by `cv_outlier_scan` to count
misclassified samples across repeated outer folds.
"""
function cv_classification(; weighted::Bool=true)
    score_fn = (Y_true, Y_pred) -> 1 - nmc(Y_true, Y_pred, weighted)
    predict_fn = (model, X, k) -> predictonehot(model, X, k)
    select_fn = argmax
    flag_fn = (Y_true, Y_pred) -> one_hot_to_labels(Y_pred) .≠ one_hot_to_labels(Y_true)
    (score_fn=score_fn, predict_fn=predict_fn, select_fn=select_fn, flag_fn=flag_fn)
end

"""
    cv_regression(;
        score_fn=(Y_true, Y_pred) -> sqrt(mean((Y_true .- Y_pred) .^ 2)),
        select_fn=argmin,
    )

Return a named tuple containing `score_fn`, `predict_fn`, and `select_fn` suited to
CPPLS regression. By default, predictions are scored with root mean squared error and
the selected number of components minimizes that loss.

This helper is meant to supply the callback interface expected by `nested_cv` and
`nested_cv_permutation` for regression problems. The returned `predict_fn` extracts the
prediction matrix corresponding to the requested number of components.
"""
function cv_regression(;
    score_fn::Function=(Y_true, Y_pred) -> sqrt(mean((Y_true .- Y_pred) .^ 2)),
    select_fn::Function=argmin,
)
    predict_fn = (model, X, k) -> predict(model, X, k)[:, :, end]
    (score_fn=score_fn, predict_fn=predict_fn, select_fn=select_fn)
end

############################################################################################
# Public API
############################################################################################

"""
    nested_cv(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple=(;),
        obs_weight_fn::Union{Function, Nothing}=nothing,
        score_fn::Function,
        predict_fn::Function,
        select_fn::Function,
        num_outer_folds::Int=8,
        num_outer_folds_repeats::Int=num_outer_folds,
        num_inner_folds::Int=7,
        num_inner_folds_repeats::Int=num_inner_folds,
        max_components::Int=spec.n_components,
        strata::Union{AbstractVector{<:Int}, Nothing}=nothing,
        reshuffle_outer_folds::Bool=false,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true)

Run explicit nested cross-validation for CPPLS. The caller supplies `score_fn`,
`predict_fn`, and `select_fn`, so the routine can be used for either regression or
classification. `spec` and `fit_kwargs` control model fitting, while the return value is
`(outer_fold_scores, optimal_num_latent_variables)`.

When provided, `obs_weight_fn` is called on each training split as
`obs_weight_fn(X_train, Y_train; sample_indices=..., fit_kwargs=..., spec=...)`. The
callback must return fold-local observation weights matching the current training-set
size, or `nothing`. Any fold-local weights returned by the callback are combined
elementwise with fixed `obs_weights` supplied through `fit_kwargs`.

For standard use, these callbacks can be obtained from `cv_classification()` or
`cv_regression()`.

When `spec.analysis_mode == :discriminant`, default `response_labels` are injected if
they are not already present in `fit_kwargs`.
"""
function nested_cv(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple = (;),
    obs_weight_fn::Union{Function, Nothing}=nothing,
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    num_outer_folds::Int=8,
    num_outer_folds_repeats::Int=num_outer_folds,
    num_inner_folds::Int=7,
    num_inner_folds_repeats::Int=num_inner_folds,
    max_components::Int=spec.n_components,
    strata::T1=nothing,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true
) where {
    T1<:Union{AbstractVector{<:Int}, Nothing}
}
    num_outer_folds_repeats > 0 || throw(ArgumentError(
        "The number of outer folds must be greater than zero"))
    num_inner_folds_repeats > 0 || throw(ArgumentError(
        "The number of inner folds must be greater than zero"))

    reshuffle_outer_folds || num_outer_folds_repeats ≤ num_outer_folds || throw(
        ArgumentError("The number of outer fold repeats cannot exceed the number " * 
            "of outer folds unless reshuffle_outer_folds=true"))

    num_inner_folds_repeats ≤ num_inner_folds || throw(ArgumentError(
        "The number of inner fold repeats cannot exceed the number of inner folds"))

    max_components > 0 || throw(ArgumentError(
        "The number of components must be greater than zero"))

    n_samples = size(X, 1)
    size(Y, 1) == n_samples || throw(DimensionMismatch(
        "Row count mismatch between X and Y"))
    isnothing(strata) || length(strata) == n_samples || throw(DimensionMismatch(
        "Length of strata must match the number of samples."))

    outer_fold_scores = Vector{Float64}(undef, num_outer_folds_repeats)
    optimal_num_latent_variables = Vector{Int}(undef, num_outer_folds_repeats)

    outer_folds = reshuffle_outer_folds ? nothing :
                  build_folds(n_samples, num_outer_folds, rng; strata=strata)

    for outer_fold_idx = 1:num_outer_folds_repeats
        folds = reshuffle_outer_folds ?
                build_folds(n_samples, num_outer_folds, rng; strata=strata) :
                outer_folds

        test_indices = reshuffle_outer_folds ? folds[1] : folds[outer_fold_idx]

        verbose && println("Outer fold: ", outer_fold_idx, " / ", num_outer_folds_repeats)

        @views X_test = X[test_indices, :]
        @views Y_test = Y[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X[train_indices, :]
        @views Y_train = Y[train_indices, :]

        base_fold_kwargs = subset_fit_kwargs(fit_kwargs, train_indices, n_samples)
        fold_kwargs = resolve_obs_weights(
            base_fold_kwargs,
            obs_weight_fn,
            X_train,
            Y_train,
            train_indices,
            spec,
        )
        inner_strata = isnothing(strata) ? nothing : strata[train_indices]
        if spec.analysis_mode ≡ :discriminant
            fold_kwargs = ensure_response_labels(fold_kwargs, Y_train)
        end

        optimal_num_latent_variables[outer_fold_idx] = optimize_num_latent_variables(
            X_train, Y_train, max_components, num_inner_folds, num_inner_folds_repeats, 
            spec, base_fold_kwargs, obs_weight_fn, score_fn, predict_fn, select_fn, rng,
            verbose; strata=inner_strata, sample_indices=train_indices)

        spec_k = with_n_components(spec, optimal_num_latent_variables[outer_fold_idx])
        final_model = fit(spec_k, X_train, Y_train; fold_kwargs...)

        Y_pred = predict_fn(final_model, X_test, 
            optimal_num_latent_variables[outer_fold_idx])
        score = score_fn(Y_test, Y_pred)
        score isa Real || throw(ArgumentError("score_fn must return a Real"))
        outer_fold_scores[outer_fold_idx] = score

        verbose && println("Score for outer fold: ", outer_fold_scores[outer_fold_idx], 
            "\n")
    end

    outer_fold_scores, optimal_num_latent_variables
end

"""
    nested_cv_permutation(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple=(;),
        obs_weight_fn::Union{Function, Nothing}=nothing,
        score_fn::Function,
        predict_fn::Function,
        select_fn::Function,
        num_permutations::Int=999,
        num_outer_folds::Int=8,
        num_outer_folds_repeats::Int=num_outer_folds,
        num_inner_folds::Int=7,
        num_inner_folds_repeats::Int=num_inner_folds,
        max_components::Int=spec.n_components,
        strata::Union{AbstractVector{<:Int}, Nothing}=nothing,
        reshuffle_outer_folds::Bool=false,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true)

Run a permutation test around `nested_cv` by repeatedly permuting the rows of `Y` and
recomputing the nested cross-validation score. The result is a vector of mean scores,
one for each permutation. As in `nested_cv`, the required callbacks can be obtained from
`cv_classification()` or `cv_regression()`.
"""
function nested_cv_permutation(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple = (;),
    obs_weight_fn::Union{Function, Nothing}=nothing,
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    num_permutations::Int=999,
    num_outer_folds::Int=8,
    num_outer_folds_repeats::Int=num_outer_folds,
    num_inner_folds::Int=7,
    num_inner_folds_repeats::Int=num_inner_folds,
    max_components::Int=spec.n_components,
    strata::T1=nothing,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
) where {
    T1<:Union{AbstractVector{<:Integer},Nothing}
}
    num_permutations > 0 || throw(ArgumentError(
        "num_permutations must be greater than zero"))

    n_samples = size(X, 1)
    size(Y, 1) == n_samples || throw(DimensionMismatch(
        "Row count mismatch between X and Y"))
    isnothing(strata) || length(strata) == n_samples || throw(DimensionMismatch(
        "Length of strata must match the number of samples."))

    permutation_scores = Vector{Float64}(undef, num_permutations)

    for perm_idx = 1:num_permutations
        perm = randperm(rng, n_samples)
        Y_perm = Y[perm, :]
        strata_perm = isnothing(strata) ? nothing : strata[perm]

        verbose && println("Permutation: ", perm_idx, " / ", num_permutations)

        scores, _ = nested_cv(X, Y_perm; spec=spec, fit_kwargs=fit_kwargs, 
            obs_weight_fn=obs_weight_fn, score_fn=score_fn, predict_fn=predict_fn,
            select_fn=select_fn,
            num_outer_folds=num_outer_folds, 
            num_outer_folds_repeats=num_outer_folds_repeats, 
            num_inner_folds=num_inner_folds, 
            num_inner_folds_repeats=num_inner_folds_repeats, max_components=max_components,
            strata=strata_perm, reshuffle_outer_folds=reshuffle_outer_folds, rng=rng,
            verbose=verbose)

        permutation_scores[perm_idx] = mean(scores)
    end

    permutation_scores
end

"""
    cv_outlier_scan(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple=(;),
        obs_weight_fn::Union{Function, Nothing}=nothing,
        num_outer_folds::Integer=8,
        num_outer_folds_repeats::Integer=10 * num_outer_folds,
        num_inner_folds::Integer=7,
        num_inner_folds_repeats::Integer=num_inner_folds,
        max_components::Integer=spec.n_components,
        reshuffle_outer_folds::Bool=true,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true)

Run repeated nested cross-validation for classification and count how often each sample
is tested and misclassified. The result is a named tuple containing `n_tested`,
`n_flagged`, and `rate = n_flagged ./ n_tested`. By default,
`reshuffle_outer_folds = true` so samples are re-evaluated across different outer splits.
This routine uses the callback bundle returned by `cv_classification()`.
"""
function cv_outlier_scan(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple = (;),
    obs_weight_fn::Union{Function, Nothing}=nothing,
    num_outer_folds::Int=8,
    num_outer_folds_repeats::Int=10 * num_outer_folds,
    num_inner_folds::Int=7,
    num_inner_folds_repeats::Int=num_inner_folds,
    max_components::Int=n_components(spec),
    reshuffle_outer_folds::Bool=true,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true
)
    spec.analysis_mode ≡ :discriminant || throw(ArgumentError(
        "cv_outlier_scan expects spec.analysis_mode = :discriminant"))

    n_samples = size(X, 1)
    size(Y, 1) == n_samples || throw(DimensionMismatch(
        "Row count mismatch between X and Y"))

    cfg = cv_classification()
    strata = one_hot_to_labels(Y)

    n_tested = zeros(Int, n_samples)
    n_flagged = zeros(Int, n_samples)

    reshuffle_outer_folds || num_outer_folds_repeats ≤ num_outer_folds || throw(
        ArgumentError("The number of outer fold repeats cannot exceed the number of " * 
            "outer folds unless reshuffle_outer_folds=true"))

    fixed_folds = reshuffle_outer_folds ? nothing :
                  build_folds(n_samples, num_outer_folds, rng; strata=strata)

    for outer_fold_idx = 1:num_outer_folds_repeats
        folds = reshuffle_outer_folds ?
                build_folds(n_samples, num_outer_folds, rng; strata=strata) :
                fixed_folds
        test_indices = reshuffle_outer_folds ? folds[1] : folds[outer_fold_idx]

        verbose && println("Outer fold: ", outer_fold_idx, " / ", num_outer_folds_repeats)

        @views X_test = X[test_indices, :]
        @views Y_test = Y[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X[train_indices, :]
        @views Y_train = Y[train_indices, :]

        base_fold_kwargs = subset_fit_kwargs(fit_kwargs, train_indices, n_samples)
        fold_kwargs = resolve_obs_weights(
            base_fold_kwargs,
            obs_weight_fn,
            X_train,
            Y_train,
            train_indices,
            spec,
        )
        fold_kwargs = ensure_response_labels(fold_kwargs, Y_train)
        inner_strata = strata[train_indices]

        best_k = optimize_num_latent_variables(X_train, Y_train, max_components, 
            num_inner_folds, num_inner_folds_repeats, spec, base_fold_kwargs,
            obs_weight_fn, cfg.score_fn, cfg.predict_fn, cfg.select_fn, rng, verbose;
            strata=inner_strata, sample_indices=train_indices)

        spec_k = with_n_components(spec, best_k)
        final_model = fit(spec_k, X_train, Y_train; fold_kwargs...)

        Y_pred = cfg.predict_fn(final_model, X_test, best_k)
        flags = cfg.flag_fn(Y_test, Y_pred)
        length(flags) == length(test_indices) || throw(DimensionMismatch(
            "flag_fn must return one flag per test sample"))

        n_tested[test_indices] .+= 1
        n_flagged[test_indices] .+= flags
    end

    rate = n_flagged ./ max.(1, n_tested)
    (n_tested = n_tested, n_flagged = n_flagged, rate = rate)
end

"""
    cv_outlier_scan(
        X::AbstractMatrix{<:Real},
        sample_classes::AbstractCategoricalArray; 
        kwargs...
    )

Convert categorical sample classes to one-hot form and forward to `cv_outlier_scan`.
"""
function cv_outlier_scan(
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...
) where {T,R,V,C,U}

    Y, _ = labels_to_one_hot(sample_classes)
    cv_outlier_scan(X, Y; kwargs...)
end

"""
    cv_outlier_scan(
        X::AbstractMatrix{<:Real}, 
        sample_classes::AbstractVector; 
        kwargs...
    )

Convert non-numeric sample classes to one-hot form and forward to `cv_outlier_scan`.
Numeric vectors are rejected because they are ambiguous with regression targets.
"""
function cv_outlier_scan(
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractVector;
    kwargs...
)
    if eltype(sample_classes) <: Real
        throw(ArgumentError(
            "cv_outlier_scan expects categorical sample_classes or one-hot Y."))
    end

    Y, _ = labels_to_one_hot(sample_classes)
    cv_outlier_scan(X, Y; kwargs...)
end

############################################################################################
# Internal Helpers
############################################################################################

"""
    CPPLS.optimize_num_latent_variables(
        X_train_full::AbstractMatrix{<:Real},
        Y_train_full::AbstractMatrix{<:Real},
        max_components::Int,
        num_inner_folds::Int,
        num_inner_folds_repeats::Int,
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple,
        obs_weight_fn::Union{Function, Nothing},
        score_fn::Function,
        predict_fn::Function,
        select_fn::Function,
        rng::AbstractRNG,
        verbose::Bool;
        strata::Union{AbstractVector{<:Int}, Nothing}=nothing,
        sample_indices::AbstractVector{<:Int}=collect(1:size(X_train_full, 1)))

Run repeated inner cross-validation to select the number of latent variables. For each
inner split, a model is fit with up to `max_components` components, evaluated with
`score_fn` and `predict_fn`, and reduced to a single component count with `select_fn`.
The returned value is the median selected component count across repeats.
"""
function optimize_num_latent_variables(
    X_train_full::AbstractMatrix{<:Real},
    Y_train_full::AbstractMatrix{<:Real},
    max_components::Int,
    num_inner_folds::Int,
    num_inner_folds_repeats::Int,
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple,
    obs_weight_fn::Union{Function, Nothing},
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    rng::AbstractRNG,
    verbose::Bool;
    strata::T1=nothing,
    sample_indices::AbstractVector{<:Int}=collect(1:size(X_train_full, 1)),
) where {
    T1<:Union{AbstractVector{<:Int}, Nothing}
}
    max_components > 0 || throw(ArgumentError(
        "The number of components must be greater than zero"))
    num_inner_folds_repeats ≤ num_inner_folds || throw(ArgumentError(
            "The number of inner fold repeats cannot exceed the number of inner folds"))

    n_samples = size(X_train_full, 1)
    size(Y_train_full, 1) == n_samples || throw(DimensionMismatch(
        "Row count mismatch between X_train_full and Y_train_full"))
    length(sample_indices) == n_samples || throw(DimensionMismatch(
        "Length of sample_indices must match the number of training samples."))

    inner_folds = build_folds(n_samples, num_inner_folds, rng; strata=strata)

    best_num_latent_vars_per_fold = Vector{Int}(undef, num_inner_folds_repeats)

    for inner_fold_idx = 1:num_inner_folds_repeats
        test_indices = inner_folds[inner_fold_idx]

        verbose && println("  Inner fold: ", inner_fold_idx, " / ", num_inner_folds_repeats)

        @views X_validation = X_train_full[test_indices, :]
        @views Y_validation = Y_train_full[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X_train_full[train_indices, :]
        @views Y_train = Y_train_full[train_indices, :]
        fold_sample_indices = sample_indices[train_indices]

        fold_kwargs = subset_fit_kwargs(fit_kwargs, train_indices, n_samples)
        fold_kwargs = resolve_obs_weights(
            fold_kwargs,
            obs_weight_fn,
            X_train,
            Y_train,
            fold_sample_indices,
            spec,
        )
        if spec.analysis_mode ≡ :discriminant
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
        1 ≤ best_k ≤ max_components || throw(ArgumentError(
            "select_fn must return an integer between 1 and $max_components"))
        
        best_num_latent_vars_per_fold[inner_fold_idx] = best_k
    end

    floor(Int, median(best_num_latent_vars_per_fold))
end

"""
    resolve_obs_weights(fit_kwargs, obs_weight_fn, X_train, Y_train, sample_indices, spec)

Return `fit_kwargs` with fold-local observation weights applied. Fixed `obs_weights`
present in `fit_kwargs` are preserved and combined elementwise with weights returned by
`obs_weight_fn`.
"""
function resolve_obs_weights(
    fit_kwargs::NamedTuple,
    obs_weight_fn::Union{Function, Nothing},
    X_train::AbstractMatrix{<:Real},
    Y_train::AbstractMatrix{<:Real},
    sample_indices::AbstractVector{<:Int},
    spec::CPPLSSpec,
)
    isnothing(obs_weight_fn) && return fit_kwargs

    base_weights = haskey(fit_kwargs, :obs_weights) ? fit_kwargs.obs_weights : nothing
    derived_weights = obs_weight_fn(
        X_train,
        Y_train;
        sample_indices=sample_indices,
        fit_kwargs=fit_kwargs,
        spec=spec,
    )
    isnothing(derived_weights) && return fit_kwargs

    checked_weights = validate_obs_weight_output(derived_weights, size(X_train, 1))
    final_weights = isnothing(base_weights) ? checked_weights :
        combine_obs_weights(base_weights, checked_weights)

    merge(fit_kwargs, (; obs_weights=final_weights))
end

function validate_obs_weight_output(weights, n_samples::Int)
    weights isa AbstractVector{<:Real} || throw(ArgumentError(
        "obs_weight_fn must return an AbstractVector of real numbers or nothing."))
    length(weights) == n_samples || throw(DimensionMismatch(
        "obs_weight_fn returned $(length(weights)) weights for $n_samples training samples."))
    all(isfinite, weights) || throw(ArgumentError(
        "obs_weight_fn returned non-finite observation weights."))
    all(≥(0), weights) || throw(ArgumentError(
        "obs_weight_fn returned negative observation weights."))
    any(>(0), weights) || throw(ArgumentError(
        "obs_weight_fn returned only zero observation weights."))
    weights isa Vector{Float64} ? weights : Float64.(weights)
end

function combine_obs_weights(
    base_weights::AbstractVector{<:Real},
    derived_weights::AbstractVector{<:Real},
)
    length(base_weights) == length(derived_weights) || throw(DimensionMismatch(
        "obs_weights and obs_weight_fn output must have the same length."))
    combined = Float64.(base_weights) .* Float64.(derived_weights)
    any(>(0), combined) || throw(ArgumentError(
        "Combining obs_weights with obs_weight_fn output produced only zero weights."))
    combined
end

"""
    build_folds(
        n_samples::Int, 
        num_folds::Int, 
        rng::AbstractRNG; 
        strata::Union{AbstractVector{<:Int}, Nothing}=nothing
    )

Construct cross-validation folds for `n_samples` observations. When `strata` is provided, 
stratified folds are created via `random_batch_indices`; otherwise shuffled contiguous 
folds are returned.
"""
function build_folds(
    n_samples::Int,
    num_folds::Int,
    rng::AbstractRNG;
    strata::T1=nothing,
) where {
    T1<:Union{AbstractVector{<:Int}, Nothing}
}
    num_folds ≥ 1 || throw(ArgumentError("Number of folds must be at least 1."))
    
    num_folds ≤ n_samples || throw( ArgumentError(
        "Number of folds ($num_folds) exceeds number of samples ($n_samples)."))

    if strata ≡ nothing
        indices = shuffle(rng, collect(1:n_samples))
        base = fld(n_samples, num_folds)
        extra = n_samples % num_folds
        folds = Vector{Vector{Int}}(undef, num_folds)
        start = 1
        for i in 1:num_folds
            n_take = base + (i ≤ extra ? 1 : 0)
            stop = start + n_take - 1
            folds[i] = indices[start:stop]
            start = stop + 1
        end
        return folds
    end

    length(strata) == n_samples || throw(DimensionMismatch(
        "Length of strata must match the number of samples."))

    random_batch_indices(strata, num_folds, rng)
end

"""
    subset_fit_kwargs(fit_kwargs::NamedTuple, train_indices, n_samples)

Subset fold-dependent entries in `fit_kwargs` to the current training indices and return
the adjusted `NamedTuple`.
"""
function subset_fit_kwargs(
    fit_kwargs::NamedTuple,
    train_indices::AbstractVector{<:Int},
    n_samples::Int
)
    isempty(fit_kwargs) && return fit_kwargs

    out_pairs = Pair{Symbol, Any}[]
    for (key, value) in Base.pairs(fit_kwargs)
        adjusted = if key in (:obs_weights, :sample_labels, :sample_classes)
            subset_vector_like(value, train_indices, n_samples, key)
        elseif key in (:Y_aux, :Y_auxiliary)
            subset_matrix_like(value, train_indices, n_samples, key)
        else
            value
        end
        push!(out_pairs, key => adjusted)
    end
    (; out_pairs...)
end

"""
    ensure_response_labels(fit_kwargs::NamedTuple, Y::AbstractMatrix{<:Real})

Ensure that discriminant fits have `response_labels` in `fit_kwargs`, injecting default
labels `"1"`, `"2"`, and so on when necessary.
"""
function ensure_response_labels(
    fit_kwargs::NamedTuple,
    Y::AbstractMatrix{<:Real},
)
    haskey(fit_kwargs, :response_labels) && return fit_kwargs
    labels = string.(1:size(Y, 2))
    merge(fit_kwargs, (response_labels = labels,))
end

"""
    subset_vector_like(values, train_indices, n_samples, name)

Return the training subset of a vector-like argument passed through `fit_kwargs`. Values
that already match the training set length are returned unchanged; `nothing` and
non-vectors are passed through.
"""
function subset_vector_like(
    values,
    train_indices::AbstractVector{<:Int},
    n_samples::Int,
    name::Symbol,
)
    isnothing(values) && return values
    values isa AbstractVector || return values

    if length(values) == n_samples
        return values[train_indices]
    elseif length(values) == length(train_indices)
        return values
    else
        throw(DimensionMismatch(
            "Length of $name must match the total sample count or the number of " * 
            "training samples."))
    end
end

"""
    subset_matrix_like(values, train_indices, n_samples, name)

Return the training subset of a matrix-like argument passed through `fit_kwargs`. Values
that already match the training set shape are returned unchanged; `nothing` and
non-matrices are passed through.
"""
function subset_matrix_like(
    values,
    train_indices::AbstractVector{<:Int},
    n_samples::Int,
    name::Symbol,
)
    isnothing(values) && return values
    values isa AbstractVector && return subset_vector_like(values, train_indices, n_samples, 
        name)
    values isa AbstractMatrix || return values

    if size(values, 1) == n_samples
        return values[train_indices, :]
    elseif size(values, 1) == length(train_indices)
        return values
    else
        throw(DimensionMismatch(
            "Row count of $name must match the total sample count or the number " * 
            "of training samples."))
    end
end

"""
    with_n_components(spec::CPPLSSpec, n_components::Int)

Return a copy of `spec` with `n_components` replaced and all other fields preserved.
"""
function with_n_components(spec::CPPLSSpec, n_components::Int)
    CPPLSSpec(
        n_components=n_components, gamma=spec.gamma, center=spec.center,
        X_tolerance=spec.X_tolerance,
        X_loading_weight_tolerance=spec.X_loading_weight_tolerance,
        t_squared_norm_tolerance=spec.t_squared_norm_tolerance,
        gamma_rel_tol=spec.gamma_rel_tol, gamma_abs_tol=spec.gamma_abs_tol,
        analysis_mode=spec.analysis_mode
    )
end
