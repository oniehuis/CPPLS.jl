# Cross Validation

CPPLS is a supervised method, so it is always at risk of learning structure that is only
accidentally aligned with the response. Cross-validation is used to test whether the
relationship learned during fitting also generalizes to samples that were not used to fit
the model. In CPPLS, cross-validation also serves a second purpose: selecting the number
of latent variables. These two tasks are coupled, because model complexity has a direct
effect on apparent predictive performance.

The package implements explicit nested cross-validation. The outer loop is used for 
performance assessment, whereas the inner loop is used for model selection. This keeps
the choice of the number of latent variables separated from the final evaluation of the
model and reduces optimistic bias.

## How Nested Cross-Validation Is Implemented in CPPLS

[`nested_cv`](@ref) uses disjoint outer folds for performance assessment and disjoint
inner folds for model selection. For each outer repeat, one fold is held out as a test
set and the remaining samples are used for training. Within that outer training set, an
inner cross-validation determines the number of latent variables. A final model is then
fitted on the full outer training set with the selected number of components and applied
to the outer test set.

For each inner fold, CPPLS evaluates all component counts from `1:max_components` and
selects the best one with `select_fn`. The final component count for the current outer
repeat is the median of those inner-fold selections, rounded down to an integer. With the
default selectors, ties are resolved in favor of the smaller component count.

If `strata` are supplied, fold construction is stratified so that class proportions are
approximately preserved across folds. Within a given fold construction, the folds are
non-overlapping. What can change is only whether the outer folds are reused or rebuilt
between repeats: with `reshuffle_outer_folds=false`, the same outer partition is reused,
whereas with `reshuffle_outer_folds=true`, a new outer partition is drawn for each repeat.

In conclusion, the outer scores measure prediction on samples excluded from fitting, while 
the inner loop controls model complexity. For classification, [`cv_classification`](@ref) 
provides an accuracy-like score based on one-hot predictions and normalized 
misclassification cost. For regression, [`cv_regression`](@ref) provides the corresponding 
callbacks, using root mean squared error by default.

## Permutation-Based Significance Assessment

Even a carefully cross-validated supervised model can appear predictive if the response is
weakly structured, the sample size is small, or the predictor matrix is high-dimensional.
To address that question, CPPLS provides [`nested_cv_permutation`](@ref), which places the
entire nested cross-validation workflow inside a permutation test.

The idea is to destroy the real link between predictors and response while keeping the
predictor matrix itself unchanged. CPPLS does that by permuting the rows of `Y`. After
permutation:

1. The same nested CV procedure is rerun.
2. The mean outer-fold score from that permuted run is stored.
3. This is repeated many times to build a null distribution of scores expected when the
    correspondence between `X` and `Y` is random.

This is useful because it tests the whole modeling pipeline, not just a single fitted
model. The null distribution therefore includes the effect of repeated fold splitting,
inner-loop model selection, and final outer-loop evaluation. If the score from the real,
unpermuted data lies well outside what is typical under permutation, the result is more
consistent with genuine predictive structure than with chance alignment. The helper
[`calculate_p_value`](@ref) can then be used to summarize the observed score relative to
the permutation distribution. This comparison is only valid if the score from the real
data is aggregated in exactly the same way, that is, as the mean of the outer-fold
scores.

When `strata` are supplied, the same permutation applied to `Y` is also applied to the
stratification vector before nested CV is rerun. This keeps the stratified fold generation
consistent with the permuted labels during each null-model evaluation.

## DA-Specific Interpretation and Outlier Scanning

For discriminant analysis, ordinary cross-validation implicitly treats the provided class
labels as ground truth. That means the assessment assumes that every sample assigned to a
class is in fact a correct representative of that class. In controlled benchmark datasets
that assumption may be reasonable, but in experimental settings it is not guaranteed.
Samples can be mislabeled, contaminated, technically compromised, or simply atypical in a
way that makes them behave poorly under the fitted model.

For that reason, CPPLS also provides [`cv_outlier_scan`](@ref). This routine is not a
replacement for nested CV; instead, it is a diagnostic extension for discriminant
analysis. It repeatedly:

1. builds an outer training/test split,
2. selects the number of latent variables within the training data,
3. predicts the held-out samples, and
4. records which test samples were misclassified.

Across repeated outer folds, each sample accumulates two counts:

1. `n_tested`: how many times that sample appeared in an outer test set,
2. `n_flagged`: how many of those appearances led to a wrong class assignment.

The ratio `n_flagged ./ n_tested` is returned as `rate`. Samples with elevated rates are
not automatically mislabeled, but they are candidates for closer inspection. In practice,
such samples may indicate possible class-assignment problems, outlying biology, unusual
measurement behavior, or incompatibility between the sample and the fitted latent
structure.

This makes `cv_outlier_scan` especially useful in experimental DA applications, where the
goal is often not only to estimate classification performance but also to identify samples
that deserve follow-up quality control.

# Example

We again use the synthetic discriminant-analysis dataset introduced on the [Fit](@ref)
page. The goal here is to estimate predictive performance with nested cross-validation,
compare that performance against a permutation-based null distribution, and then inspect
which samples are most often misclassified across repeated outer folds.

To keep the documentation example reasonably fast, we use a fixed `gamma=0.5` and allow at
most two latent variables. For a real analysis, those settings should be chosen more 
carefully.

In this example, class balancing is applied through `obs_weight_fn`, which is recomputed
inside each training fold. This is preferable to precomputing inverse-frequency weights on
the full dataset before cross-validation. If you instead want fixed sample-specific
weights, pass them through `fit_kwargs=(; obs_weights=...)` and they will simply be
subsetted to each training split.

The example uses the encoding helpers 
[`labels_to_one_hot`](@ref labels_to_one_hot(::AbstractVector)) and
[`one_hot_to_labels`](@ref). The first converts a vector of class labels into the one-hot
response matrix expected by the fitting and cross-validation routines, while the second
converts such a matrix back to ordinary class labels. That back-conversion is used below
when stratified folds or class-frequency weights need access to the class labels again.

The packages loaded below play different roles: `CPPLS` provides the modeling and
cross-validation functions, `JLD2` reads the example dataset from disk, `Random`
provides a reproducible RNG, `Statistics` provides summary functions such as `mean`, and
`CairoMakie` is used to draw and save the histogram. In a normal Julia environment,
packages such as `CPPLS`, `JLD2`, and `CairoMakie` must be installed before running the
example; the Julia Pkg documentation explains how to install registered packages in the
[Getting Started](https://pkgdocs.julialang.org/v1/getting-started/#Basic-Usage)
section.

```@example crossvalidation
using CPPLS
using JLD2
using Random
using Statistics
using CairoMakie

sample_labels, X, classes, Y_aux = load(
    CPPLS.dataset("synthetic_cppls_da_dataset.jld2"),
    "sample_labels",
    "X",
    "classes",
    "Y_aux"
)

Y, response_labels = labels_to_one_hot(classes)
cfg = cv_classification()

spec = CPPLSSpec(
    n_components=2,
    gamma=0.5,
    analysis_mode=:discriminant,
)

rng = MersenneTwister(12345)
obs_weight_fn = (X_train, Y_train; kwargs...) -> invfreqweights(one_hot_to_labels(Y_train))

fit_kwargs = (
    Y_aux=Y_aux,
    sample_labels=sample_labels,
    response_labels=response_labels,
)

scores, best_components = nested_cv(
    X,
    Y;
    spec=spec,
    fit_kwargs=fit_kwargs,
    obs_weight_fn=obs_weight_fn,
    score_fn=cfg.score_fn,
    predict_fn=cfg.predict_fn,
    select_fn=cfg.select_fn,
    num_outer_folds=5,
    num_outer_folds_repeats=5,
    num_inner_folds=4,
    num_inner_folds_repeats=4,
    max_components=2,
    strata=one_hot_to_labels(Y),
    rng=rng,
    verbose=false,
)

observed_accuracy = mean(scores)
```

The returned `scores` vector contains the outer-fold accuracies, and the `best_components` 
vector contains the component counts selected inside the outer repeats. The value 
`observed_accuracy` is the mean nested-CV score on the real labels.

That mean accuracy is not especially impressive in absolute terms, because it is only
moderately better than random guessing. This raises the question of whether the result is
nonetheless significantly better than chance. To answer that, we compare it with a null
distribution obtained from permuted data in which the correspondence between predictors
and class labels has been broken. For a fair comparison, the permutation procedure uses
exactly the same settings as `nested_cv`, here with a total of `999` permutations.

```@example crossvalidation
permutation_scores = nested_cv_permutation(
    X,
    Y;
    spec=spec,
    fit_kwargs=fit_kwargs,
    obs_weight_fn=obs_weight_fn,
    score_fn=cfg.score_fn,
    predict_fn=cfg.predict_fn,
    select_fn=cfg.select_fn,
    num_permutations=999,
    num_outer_folds=5,
    num_outer_folds_repeats=5,
    num_inner_folds=4,
    num_inner_folds_repeats=4,
    max_components=2,
    strata=one_hot_to_labels(Y),
    rng=MersenneTwister(12345),
    verbose=false,
)
```

The `permutation_scores` vector contains the mean outer-fold accuracies for each of the
`999` permutations. Let us visualize that distribution.

```@example crossvalidation
f = Figure(; size=(900, 600))
ax = Axis(
    f[1, 1], 
    title="Model accuracy null distribution",
    xlabel="Mean outer-fold accuracy",
    ylabel="Count across permutations"
)
hist!(ax, permutation_scores, bins=20)
save("accuracy_hist.svg", f)
nothing
```

![](accuracy_hist.svg)

As we can see, the permutation accuracies are mostly distributed around and below `0.5`, 
with only few being as large as or larger than the observed accuracy from the real data. 
This shows that even a modest mean accuracy can still be statistically significant. We can 
quantify that more formally with [`calculate_p_value`](@ref), using the permutation scores 
and the observed accuracy. Because classification uses an accuracy-like score here,
`calculate_p_value` is called with `tail=:upper`, corresponding to a one-sided
upper-tail test in which the p-value is the probability, under the null model, of
observing a score at least as large as the observed one.

```@example crossvalidation
p_value = calculate_p_value(permutation_scores, observed_accuracy; tail=:upper)
```

In this example, the resulting p-value indicates statistical significance at an
`alpha = 0.05` threshold.

Note that the assessment above was comparatively inexpensive because `X` is small and we
used a fixed `gamma` value. With datasets containing hundreds of samples and thousands of
traits, especially when `gamma` is optimized across a dense grid such as `gamma=0:0.01:1`, 
the computation becomes much more demanding. In that situation it can make sense to 
distribute the permutation runs. For example, one could run `20` separate calls to 
[`nested_cv_permutation`](@ref) on `20` different nodes, then concatenate the stored 
`permutation_scores` vectors before passing them to [`calculate_p_value`](@ref).

!!! warning
    When permutation runs are distributed across multiple jobs or nodes, each run should
    be started with a different RNG seed. Reusing the same seed can lead to overlapping
    permutation sequences and therefore to a biased null distribution.

When the focus shifts from global performance to potentially problematic samples,
[`cv_outlier_scan`](@ref) can be used as a follow-up diagnostic.

```@example crossvalidation
outlier_scan = cv_outlier_scan(
    X,
    classes;
    spec=spec,
    fit_kwargs=(; Y_aux=Y_aux, sample_labels=sample_labels),
    obs_weight_fn=obs_weight_fn,
    num_outer_folds=5,
    num_outer_folds_repeats=500,
    num_inner_folds=4,
    num_inner_folds_repeats=4,
    max_components=2,
    rng=MersenneTwister(54321),
    verbose=false,
)
```

`outlier_scan` is a named tuple with the fields `n_tested`, `n_flagged`, and `rate`.
`n_tested` specifies how often a given sample was evaluated, `n_flagged` specifies how
often that sample was misclassified, and `rate` is the ratio of the two. Most users will
primarily be interested in `rate`. Let us inspect the samples with the largest rates,
rounded to three decimal places and sorted in descending order.

```@example crossvalidation
suspect_idx = sortperm(outlier_scan.rate, rev=true)[1:10]
rate = round.(outlier_scan.rate[suspect_idx]; digits=3)
for (j, i) in enumerate(suspect_idx)
    println("Sample: ", sample_labels[i], "; error rate: ", rate[j])
end
```

In the current example, the overall model accuracy is fairly low, so several samples show
non-negligible flagging rates. None of these samples is actually mislabeled. This
illustrates an important point: these rates do not constitute formal proof that a sample
is mislabeled. They are instead a practical way to prioritize follow-up inspection of
samples that repeatedly fail when they are held out, which is often exactly the situation
in which class-assignment problems or unusual sample behavior become visible.

The functions discussed above are documented in full below.

# API

```@docs
CPPLS.calculate_p_value
CPPLS.cv_classification
CPPLS.cv_regression
CPPLS.cv_outlier_scan
CPPLS.nested_cv
CPPLS.nested_cv_permutation
CPPLS.nmc
```
