"""
    random_batch_indices(
        strata::AbstractVector{<:Integer},
        num_batches::Integer, 
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
    strata::T1,
    num_batches::T2,
    rng::AbstractRNG=Random.GLOBAL_RNG
) where {
    T1<:AbstractVector{<:Integer}, 
    T2<:Integer
}
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

The returned callbacks implement the following defaults:
`score_fn(Y_true, Y_pred) = 1 - nmc(Y_true, Y_pred, weighted)`, so larger scores are
better and `weighted=true` applies inverse-frequency class weighting;
`predict_fn(model, X, k) = predictonehot(model, X, k)` to obtain one-hot class
predictions from a discriminant CPPLS fit;
`select_fn = argmax`, so inner cross-validation chooses the component count with the
largest score; and
`flag_fn(Y_true, Y_pred) = one_hot_to_labels(Y_pred) .≠ one_hot_to_labels(Y_true)`,
which returns a per-sample misclassification mask.

This helper is meant to supply the callback interface expected by `nested_cv`,
`nested_cv_permutation`, and `cv_outlier_scan`. In particular, `score_fn`, `predict_fn`, 
and `select_fn` can be passed directly to `nested_cv` and `nested_cv_permutation`, while 
`flag_fn` is used by `cv_outlier_scan` to count misclassified samples across repeated 
outer folds.

See also
[`cv_outlier_scan`](@ref CPPLS.cv_outlier_scan),
[`cv_regression`](@ref CPPLS.cv_regression), 
[`invfreqweights`](@ref CPPLS.invfreqweights),
[`nmc`](@ref CPPLS.nmc), 
[`nested_cv`](@ref CPPLS.nested_cv), 
[`nested_cv_permutation`](@ref CPPLS.nested_cv_permutation)
[`predict`](@ref CPPLS.predict)

```jldoctest
julia> cb = cv_classification();

julia> Y_true = [1 0; 0 1; 0 1];

julia> Y_pred = [0 1; 0 1; 0 1];

julia> cb.score_fn(Y_true, Y_pred)
0.5

julia> cb.flag_fn(Y_true, Y_pred)
3-element BitVector:
 1
 0
 0

julia> cb.select_fn([0.2, 0.6, 0.4])
2
```

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

The returned callbacks implement the following defaults:
`score_fn(Y_true, Y_pred) = sqrt(mean((Y_true .- Y_pred) .^ 2))`, so smaller scores are
better;
`predict_fn(model, X, k) = predict(model, X, k)[:, :, end]`, which extracts the
prediction matrix for the requested component count from the 3-dimensional array returned
by `predict`; and
`select_fn = argmin`, so inner cross-validation chooses the component count with the
smallest score.

This helper is meant to supply the callback interface expected by `nested_cv` and
`nested_cv_permutation` for regression problems. The returned `predict_fn` extracts the
prediction matrix corresponding to the requested number of components.

See also
[`CPPLSFitLight`](@ref CPPLS.CPPLSFitLight),
[`cv_classification`](@ref CPPLS.cv_classification),
[`nested_cv`](@ref CPPLS.nested_cv),
[`nested_cv_permutation`](@ref CPPLS.nested_cv_permutation),
[`predict`](@ref CPPLS.predict)

```jldoctest
julia> cb = cv_regression();

julia> Y_true = reshape([1.0, 2.0], :, 1);

julia> Y_pred = reshape([1.0, 3.0], :, 1);

julia> cb.score_fn(Y_true, Y_pred) ≈ sqrt(0.5)
true

julia> B, X_bar, Y_bar = reshape([2.0], 1, 1, 1), reshape([0.0], 1, 1), reshape([0.5], 1, 1);

julia> model = CPPLSFitLight(B, X_bar, Y_bar, :regression);

julia> X = reshape([1.0, 2.0], :, 1);

julia> cb.predict_fn(model, X, 1)
2×1 Matrix{Float64}:
 2.5
 4.5

julia> cb.select_fn([0.3, 0.2])
2
```
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
    nested_cv(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real};
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple=(;),
        obs_weight_fn::Union{Function, Nothing}=nothing,
        score_fn::Function,
        predict_fn::Function,
        select_fn::Function,
        num_outer_folds::Integer=8,
        num_outer_folds_repeats::Integer=num_outer_folds,
        num_inner_folds::Integer=7,
        num_inner_folds_repeats::Integer=num_inner_folds,
        max_components::Integer=spec.n_components,
        strata::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        reshuffle_outer_folds::Bool=false,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true
    )

Run explicit nested cross-validation for CPPLS. The caller supplies `score_fn`,
`predict_fn`, and `select_fn`, so the routine can be used for either regression or
classification. `spec` and `fit_kwargs` control model fitting, while the return value is
`(outer_fold_scores, optimal_num_latent_variables)`.

When provided, `obs_weight_fn` is called on each training split as
`obs_weight_fn(X_train, Y_train; sample_indices=..., fit_kwargs=..., spec=...)`. The
callback must return fold-local observation weights matching the current training-set
size, or `nothing`. Any fold-local weights returned by the callback are combined
elementwise with fixed `obs_weights` supplied through `fit_kwargs`.

For standard use, `score_fn`, `predict_fn`, and `select_fn` can be obtained from
`cv_classification()` or `cv_regression()`. In contrast, `obs_weight_fn` is an optional
user-supplied callback when fold-specific observation weighting is needed.

When `spec.analysis_mode == :discriminant`, default `response_labels` are injected if
they are not already present in `fit_kwargs`.

Arguments
- `X`: predictor matrix with one row per sample.
- `Y`: response matrix with one row per sample. For classification this is typically a
    one-hot matrix; for regression it contains continuous responses.

Keyword arguments
- `spec`: CPPLS model specification used for every fit. During inner optimization, the
    routine evaluates component counts `1:max_components` by replacing `spec.n_components`
    on temporary copies of `spec`.
- `fit_kwargs`: additional keyword arguments forwarded to `fit`. Entries tied to the
    sample axis, namely `obs_weights`, `sample_labels`, `sample_classes`, `Y_aux`, and
    `Y_auxiliary`, are subset automatically to the current training split.
- `obs_weight_fn`: optional callback for fold-local observation weights. It receives
    `X_train` and `Y_train` for the current training split, and may also inspect
    `sample_indices`, `fit_kwargs`, and `spec` through keyword arguments. It must return
    either `nothing` or an `AbstractVector` of nonnegative finite weights of length
    `size(X_train, 1)`. These weights are multiplied elementwise with any fixed
    `obs_weights` present in `fit_kwargs`.
- `score_fn`: scoring callback with signature `score_fn(Y_true, Y_pred) -> Real`. It is
    called on held-out responses and the predictions produced by `predict_fn` for a single
    candidate component count. The returned scalar is the objective optimized in inner
    cross-validation and reported on the outer folds.
- `predict_fn`: prediction callback with signature `predict_fn(model, X_holdout, k)`. It
    receives a fitted CPPLS model, the held-out predictor rows to evaluate, and the number
    of latent variables `k`. `predict_fn` must return predictions in a representation and 
    shape that `score_fn` can compare directly with `Y_true`.
- `select_fn`: model-selection callback with signature `select_fn(scores) -> Int`. It
    receives the vector of inner-fold scores for `k = 1:max_components` and must return a
    1-based component count in that range. Its optimization direction must match
    `score_fn`, for example `argmax` for larger-is-better scores or `argmin` for
    smaller-is-better losses.
- `num_outer_folds`: number of folds in the outer assessment loop.
- `num_outer_folds_repeats`: number of outer-fold evaluations to run. When
    `reshuffle_outer_folds=false`, this usually matches `num_outer_folds` so each fold in
    one fixed outer partition is evaluated once. Larger values require
    `reshuffle_outer_folds=true`, in which case new outer partitions are drawn between
    repeats.
- `num_inner_folds`: number of folds used inside each outer training split to choose the
    number of latent variables.
- `num_inner_folds_repeats`: number of inner folds evaluated per outer split. This
    cannot exceed `num_inner_folds`.
- `max_components`: largest latent-variable count considered in the inner loop.
    This value defines the search range `1:max_components` independently of the
    `n_components` stored in `spec`. If `max_components > spec.n_components`, the inner
    loop and the final outer-fold fit still evaluate those larger component counts by
    working on temporary copies of `spec` with `n_components` replaced accordingly.
- `strata`: optional integer labels used to build stratified folds. When omitted, folds
    are created by shuffling sample indices without stratification.
- `reshuffle_outer_folds`: if `true`, regenerate the outer folds on each repeat; if
    `false`, build one outer partition and reuse its folds, which is the standard nested
    cross-validation setup. If `true`, perform repeated nested cross-validation by
    drawing a new outer partition on each repeat. This is especially useful for
    diagnostics such as `cv_outlier_scan`, but it can also be used in ordinary nested CV
    when repeated random outer splits are desired. For permutation testing, however, the
    observed run and all permuted runs must use the same `reshuffle_outer_folds` setting
    and the same score aggregation; otherwise the resulting p-value is not comparable.
- `rng`: random-number generator used for fold construction and any reshuffling.
- `verbose`: if `true`, print progress for outer and inner folds.

Returns
- `outer_fold_scores`: one scalar score per outer repeat, obtained by fitting the model
    on the corresponding outer training split with the selected number of components and
    evaluating it with `score_fn` on the outer test split.
- `optimal_num_latent_variables`: one selected component count per outer repeat. Each
    value comes from repeated inner cross-validation on that outer training split.

See also
[`CPPLSSpec`](@ref CPPLS.CPPLSSpec),
[`fit`](@ref CPPLS.fit),
[`calculate_p_value`](@ref CPPLS.calculate_p_value),
[`cv_classification`](@ref CPPLS.cv_classification),
[`cv_outlier_scan`](@ref CPPLS.cv_outlier_scan),
[`cv_regression`](@ref CPPLS.cv_regression), 
[`invfreqweights`](@ref CPPLS.invfreqweights),
[`labels_to_one_hot`](@ref CPPLS.labels_to_one_hot),
[`nested_cv_permutation`](@ref CPPLS.nested_cv_permutation),
[`one_hot_to_labels`](@ref CPPLS.one_hot_to_labels)

```jldoctest
julia> using Random;

julia> X = [0.0 0.0; 0.1 0.2; 0.2 0.1; 0.3 0.2; 0.2 0.4; 0.4 0.3;
            2.0 2.0; 2.1 2.2; 2.2 2.1; 2.3 2.2; 2.2 2.4; 2.4 2.3];

julia> classes = repeat(["A", "B"], inner=6)
12-element Vector{String}:
 "A"
 "A"
 "A"
 "A"
 "A"
 "A"
 "B"
 "B"
 "B"
 "B"
 "B"
 "B"

julia> Y, response_labels = labels_to_one_hot(classes)
([1 0; 1 0; … ; 0 1; 0 1], ["A", "B"])

julia> cb = cv_classification();

julia> spec = CPPLSSpec(n_components=1, gamma=0.5, analysis_mode=:discriminant);

julia> obs_weight_fn = (X_train, Y_train; kwargs...) -> invfreqweights(one_hot_to_labels(Y_train));

julia> scores, best_k = nested_cv(
           X,
           Y;
           spec=spec,
           fit_kwargs=(; response_labels=response_labels),
           obs_weight_fn=obs_weight_fn,
           score_fn=cb.score_fn,
           predict_fn=cb.predict_fn,
           select_fn=cb.select_fn,
           num_outer_folds=3,
           num_outer_folds_repeats=3,
           num_inner_folds=2,
           num_inner_folds_repeats=2,
           max_components=1,
           strata=one_hot_to_labels(Y),
           rng=MersenneTwister(1),
           verbose=false,
       );

julia> length(scores)
3

julia> all(==(1.0), scores)
true

julia> best_k == fill(1, 3)
true
```
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
    num_outer_folds::T1=8,
    num_outer_folds_repeats::T2=num_outer_folds,
    num_inner_folds::T3=7,
    num_inner_folds_repeats::T4=num_inner_folds,
    max_components::T5=spec.n_components,
    strata::T6=nothing,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true
) where {
    T1<:Integer,
    T2<:Integer,
    T3<:Integer,
    T4<:Integer,
    T5<:Integer,
    T6<:Union{AbstractVector{<:Integer}, Nothing}
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
    nested_cv_permutation(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real};
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple=(;),
        obs_weight_fn::Union{Function, Nothing}=nothing,
        score_fn::Function,
        predict_fn::Function,
        select_fn::Function,
        num_permutations::Integer=999,
        num_outer_folds::Integer=8,
        num_outer_folds_repeats::Integer=num_outer_folds,
        num_inner_folds::Integer=7,
        num_inner_folds_repeats::Integer=num_inner_folds,
        max_components::Integer=spec.n_components,
        strata::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        reshuffle_outer_folds::Bool=false,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        verbose::Bool=true
    )

Run a permutation test around `nested_cv` by repeatedly permuting the rows of `Y` and
recomputing the nested cross-validation score. The result is a vector of mean scores,
one for each permutation. Each permutation reruns the full nested-CV pipeline, so the
null distribution reflects outer-fold assessment, inner-loop model selection, and any
fold-local observation weighting supplied through `obs_weight_fn`.

This function takes the same core callbacks as `nested_cv`: `score_fn`, `predict_fn`,
and `select_fn` can be obtained from `cv_classification()` or `cv_regression()`, while
`obs_weight_fn` remains an optional user-supplied callback.

The positional arguments `X` and `Y`, and the shared keyword arguments `spec`,
`fit_kwargs`, `obs_weight_fn`, the callback trio, the fold controls,
`max_components`, `reshuffle_outer_folds`, `rng`, and `verbose` have the same meaning
as in `nested_cv`. Here, `num_permutations` additionally controls how many shuffled
response runs are performed.

When `strata` are provided, the same row permutation applied to `Y` is also applied to
`strata` before nested CV is rerun, so fold construction remains aligned with the
permuted responses.

Returns
- `permutation_scores`: vector whose `i`th entry is `mean(scores)` from the `i`th call to
    `nested_cv` on permuted responses.

To compare these scores with an observed score using `calculate_p_value`, the observed
analysis must use the same `score_fn`, `predict_fn`, `select_fn`, fold settings,
`reshuffle_outer_folds` choice, and score aggregation.

See also
[`CPPLSSpec`](@ref CPPLS.CPPLSSpec),
[`fit`](@ref CPPLS.fit),
[`calculate_p_value`](@ref CPPLS.calculate_p_value),
[`cv_classification`](@ref CPPLS.cv_classification),
[`cv_outlier_scan`](@ref CPPLS.cv_outlier_scan),
[`cv_regression`](@ref CPPLS.cv_regression), 
[`invfreqweights`](@ref CPPLS.invfreqweights),
[`labels_to_one_hot`](@ref CPPLS.labels_to_one_hot),
[`nested_cv`](@ref CPPLS.nested_cv_permutation),
[`one_hot_to_labels`](@ref CPPLS.one_hot_to_labels)

```jldoctest
julia> using Random;

julia> X = [0.0 0.0; 0.1 0.2; 0.2 0.1; 0.3 0.2; 0.2 0.4; 0.4 0.3;
            2.0 2.0; 2.1 2.2; 2.2 2.1; 2.3 2.2; 2.2 2.4; 2.4 2.3];

julia> classes = repeat(["A", "B"], inner=6)
12-element Vector{String}:
 "A"
 "A"
 "A"
 "A"
 "A"
 "A"
 "B"
 "B"
 "B"
 "B"
 "B"
 "B"

julia> Y, response_labels = labels_to_one_hot(classes);

julia> cb = cv_classification();

julia> spec = CPPLSSpec(n_components=1, gamma=0.5, analysis_mode=:discriminant);

julia> obs_weight_fn = (X_train, Y_train; kwargs...) -> invfreqweights(one_hot_to_labels(Y_train));

julia> permutation_scores = nested_cv_permutation(
                     X,
                     Y;
                     spec=spec,
                     fit_kwargs=(; response_labels=response_labels),
                     obs_weight_fn=obs_weight_fn,
                     score_fn=cb.score_fn,
                     predict_fn=cb.predict_fn,
                     select_fn=cb.select_fn,
                     num_permutations=3,
                     num_outer_folds=3,
                     num_outer_folds_repeats=3,
                     num_inner_folds=2,
                     num_inner_folds_repeats=2,
                     max_components=1,
                     strata=one_hot_to_labels(Y),
                     rng=MersenneTwister(1),
                     verbose=false,
             );

julia> length(permutation_scores)
3

julia> all(0.0 .≤ permutation_scores .≤ 1.0)
true
```
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
    num_permutations::T1=999,
    num_outer_folds::T2=8,
    num_outer_folds_repeats::T3=num_outer_folds,
    num_inner_folds::T4=7,
    num_inner_folds_repeats::T5=num_inner_folds,
    max_components::T6=spec.n_components,
    strata::T7=nothing,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true
) where {
    T1<:Integer,
    T2<:Integer,
    T3<:Integer,
    T4<:Integer,
    T5<:Integer,
    T6<:Integer,
    T7<:Union{AbstractVector{<:Integer}, Nothing}
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
    cv_outlier_scan(
        X::AbstractMatrix{<:Real}, 
        Y::AbstractMatrix{<:Real};
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
        verbose::Bool=true
    )

Run repeated nested cross-validation for discriminant analysis and count how often each
sample is tested and misclassified. This is a diagnostic companion to `nested_cv`, not a
replacement for it: the goal is to identify samples that repeatedly fail when held out,
which can be useful when screening for mislabeled, contaminated, atypical, or otherwise
problematic observations.

Unlike `nested_cv`, this routine fixes the cross-validation callbacks internally by using
the bundle returned by `cv_classification()`. On each outer split it selects the number
of latent variables by repeated inner cross-validation, predicts the held-out samples,
and records which of those samples were misclassified. Across repeats, each sample
accumulates a test count and a misclassification count.

This method expects discriminant-analysis settings, so `spec.analysis_mode` must be
`:discriminant` and `Y` must be a one-hot response matrix. Stratification is derived
automatically from `Y` via `one_hot_to_labels(Y)`.

Arguments
- `X`: predictor matrix with one row per sample.
- `Y`: one-hot response matrix with one row per sample and one column per class.

Keyword arguments
- `spec`: CPPLS model specification used for every fit. During inner optimization, the
    routine evaluates component counts `1:max_components` by replacing `spec.n_components`
    on temporary copies of `spec`.
- `fit_kwargs`: additional keyword arguments forwarded to `fit`. Entries tied to the
    sample axis, namely `obs_weights`, `sample_labels`, `sample_classes`, `Y_aux`, and
    `Y_auxiliary`, are subset automatically to the current training split. If
    `response_labels` are not supplied, they are inferred from the number of columns in
    `Y`.
- `obs_weight_fn`: optional callback for fold-local observation weights. It is called as
    `obs_weight_fn(X_train, Y_train; sample_indices=..., fit_kwargs=..., spec=...)` on
    each outer or inner training split and must return either `nothing` or an
    `AbstractVector` of nonnegative finite weights of length `size(X_train, 1)`. These
    weights are multiplied elementwise with any fixed `obs_weights` present in
    `fit_kwargs`.
- `num_outer_folds`: number of folds in each outer partition.
- `num_outer_folds_repeats`: number of outer-fold evaluations to run. With the default
    `reshuffle_outer_folds=true`, new outer partitions are drawn between repeats so a
    sample can be tested multiple times across different train/test splits.
- `num_inner_folds`: number of folds used inside each outer training split to choose the
    number of latent variables.
- `num_inner_folds_repeats`: number of inner folds evaluated per outer split. This
    cannot exceed `num_inner_folds`.
- `max_components`: largest latent-variable count considered in the inner loop. This
    search limit is independent of `spec.n_components`; if it is larger, temporary copies
    of `spec` are used with the required component count.
- `reshuffle_outer_folds`: if `true`, regenerate the outer folds on each repeat; if
    `false`, build one outer partition and reuse its folds. Since outlier scanning is
    usually meant to probe sample stability across many different holdout sets, the
    default is `true`.
- `rng`: random-number generator used for fold construction and reshuffling.
- `verbose`: if `true`, print progress for the outer and inner folds.

Returns
- `n_tested`: integer vector whose `i`th entry counts how often sample `i` appeared in an
    outer test set.
- `n_flagged`: integer vector whose `i`th entry counts how often sample `i` was
    misclassified when it appeared in an outer test set.
- `rate`: vector defined as `n_flagged ./ max.(1, n_tested)`. Larger values indicate
    samples that are more frequently flagged when held out. A sample that was never
    tested receives rate `0.0`.

See also
[`CPPLSSpec`](@ref CPPLS.CPPLSSpec),
[`fit`](@ref CPPLS.fit),
[`cv_classification`](@ref CPPLS.cv_classification),
[`invfreqweights`](@ref CPPLS.invfreqweights),
[`labels_to_one_hot`](@ref CPPLS.labels_to_one_hot),
[`nested_cv`](@ref CPPLS.nested_cv_permutation),
[`one_hot_to_labels`](@ref CPPLS.one_hot_to_labels)

```jldoctest
julia> using Random;

julia> X = [0.0 0.0; 0.1 0.2; 0.2 0.1; 0.3 0.2; 0.2 0.4; 0.4 0.3;
            2.0 2.0; 2.1 2.2; 2.2 2.1; 2.3 2.2; 2.2 2.4; 2.4 2.3];

julia> classes = repeat(["A", "B"], inner=6);

julia> Y, response_labels = labels_to_one_hot(classes);

julia> spec = CPPLSSpec(n_components=1, gamma=0.5, analysis_mode=:discriminant);

julia> obs_weight_fn = (X_train, Y_train; kwargs...) -> invfreqweights(one_hot_to_labels(Y_train));

julia> out = cv_outlier_scan(
           X,
           Y;
           spec=spec,
           fit_kwargs=(; response_labels=response_labels),
           obs_weight_fn=obs_weight_fn,
           num_outer_folds=3,
           num_outer_folds_repeats=3,
           num_inner_folds=2,
           num_inner_folds_repeats=2,
           max_components=1,
           rng=MersenneTwister(1),
           verbose=false,
       );

julia> sum(out.n_tested)
12

julia> all(out.n_flagged .≤ out.n_tested)
true

julia> all(0.0 .≤ out.rate .≤ 1.0)
true
```
"""
function cv_outlier_scan(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple=(;),
    obs_weight_fn::Union{Function, Nothing}=nothing,
    num_outer_folds::T1=8,
    num_outer_folds_repeats::T2=10 * num_outer_folds,
    num_inner_folds::T3=7,
    num_inner_folds_repeats::T4=num_inner_folds,
    max_components::T5=n_components(spec),
    reshuffle_outer_folds::Bool=true,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true
) where {
    T1<:Integer,
    T2<:Integer,
    T3<:Integer,
    T4<:Integer,
    T5<:Integer
}
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
    optimize_num_latent_variables(
        X_train_full::AbstractMatrix{<:Real},
        Y_train_full::AbstractMatrix{<:Real},
        max_components::Integer,
        num_inner_folds::Integer,
        num_inner_folds_repeats::Integer,
        spec::CPPLSSpec,
        fit_kwargs::NamedTuple,
        obs_weight_fn::Union{Function, Nothing},
        score_fn::Function,
        predict_fn::Function,
        select_fn::Function,
        rng::AbstractRNG,
        verbose::Bool;
        strata::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        sample_indices::AbstractVector{<:Integer}=collect(1:size(X_train_full, 1))
    )

Run repeated inner cross-validation to select the number of latent variables. For each
inner split, a model is fit with up to `max_components` components, evaluated with
`score_fn` and `predict_fn`, and reduced to a single component count with `select_fn`.
The returned value is the median selected component count across repeats.
"""
function optimize_num_latent_variables(
    X_train_full::AbstractMatrix{<:Real},
    Y_train_full::AbstractMatrix{<:Real},
    max_components::T1,
    num_inner_folds::T2,
    num_inner_folds_repeats::T3,
    spec::CPPLSSpec,
    fit_kwargs::NamedTuple,
    obs_weight_fn::Union{Function, Nothing},
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    rng::AbstractRNG,
    verbose::Bool;
    strata::T4=nothing,
    sample_indices::T5=collect(1:size(X_train_full, 1))
) where {
    T1<:Integer,
    T2<:Integer,
    T3<:Integer,
    T4<:Union{AbstractVector{<:Integer}, Nothing},
    T5<:AbstractVector{<:Integer}
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
    resolve_obs_weights(
        fit_kwargs::NamedTuple, 
        obs_weight_fn::Union{Function, Nothing},
        X_train::AbstractMatrix{<:Real},
        Y_train::AbstractMatrix{<:Real},
        sample_indices::AbstractVector{<:Integer},
        spec::CPPLSSpec
    )

Return `fit_kwargs` with fold-local observation weights applied. Fixed `obs_weights`
present in `fit_kwargs` are preserved and combined elementwise with weights returned by
`obs_weight_fn`.
"""
function resolve_obs_weights(
    fit_kwargs::NamedTuple,
    obs_weight_fn::Union{Function, Nothing},
    X_train::AbstractMatrix{<:Real},
    Y_train::AbstractMatrix{<:Real},
    sample_indices::AbstractVector{<:Integer},
    spec::CPPLSSpec
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
        n_samples::Integer, 
        num_folds::Integer, 
        rng::AbstractRNG; 
        strata::Union{AbstractVector{<:Integer}, Nothing}=nothing
    )

Construct cross-validation folds for `n_samples` observations. When `strata` is provided, 
stratified folds are created via `random_batch_indices`; otherwise shuffled contiguous 
folds are returned.
"""
function build_folds(
    n_samples::Integer,
    num_folds::Integer,
    rng::AbstractRNG;
    strata::T1=nothing
) where {
    T1<:Union{AbstractVector{<:Integer}, Nothing}
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
    subset_fit_kwargs(
        fit_kwargs::NamedTuple, 
        train_indices::AbstractVector{<:Integer}, 
        n_samples::Integer
    )

Subset fold-dependent entries in `fit_kwargs` to the current training indices and return
the adjusted `NamedTuple`.
"""
function subset_fit_kwargs(
    fit_kwargs::NamedTuple,
    train_indices::AbstractVector{<:Integer},
    n_samples::Integer
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
    Y::AbstractMatrix{<:Real}
)
    haskey(fit_kwargs, :response_labels) && return fit_kwargs
    labels = string.(1:size(Y, 2))
    merge(fit_kwargs, (response_labels = labels,))
end

"""
    subset_vector_like(
        values, 
        train_indices::AbstractVector{<:Integer}, 
        n_samples::Integer, 
        name::Symbol
    )

Return the training subset of a vector-like argument passed through `fit_kwargs`. Values
that already match the training set length are returned unchanged; `nothing` and
non-vectors are passed through.
"""
function subset_vector_like(
    values,
    train_indices::AbstractVector{<:Integer},
    n_samples::Integer,
    name::Symbol
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
    subset_matrix_like(
        values, 
        train_indices::AbstractVector{<:Integer}, 
        n_samples::Integer, name::Symbol
    )

Return the training subset of a matrix-like argument passed through `fit_kwargs`. Values
that already match the training set shape are returned unchanged; `nothing` and
non-matrices are passed through.
"""
function subset_matrix_like(
    values,
    train_indices::AbstractVector{<:Integer},
    n_samples::Integer,
    name::Symbol
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
    with_n_components(spec::CPPLSSpec, n_components::Integer)

Return a copy of `spec` with `n_components` replaced and all other fields preserved.
"""
function with_n_components(spec::CPPLSSpec, n_components::Integer)
    CPPLSSpec(
        n_components=n_components, gamma=spec.gamma, center=spec.center,
        X_tolerance=spec.X_tolerance,
        X_loading_weight_tolerance=spec.X_loading_weight_tolerance,
        t_squared_norm_tolerance=spec.t_squared_norm_tolerance,
        gamma_rel_tol=spec.gamma_rel_tol, gamma_abs_tol=spec.gamma_abs_tol,
        analysis_mode=spec.analysis_mode
    )
end
