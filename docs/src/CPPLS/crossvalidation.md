# Cross Validation

CPPLS is a supervised method, so it is always at risk of learning structure that is only
accidentally aligned with the response. Cross-validation is used to test whether the
relationship learned during fitting also generalizes to samples that were not used to fit
the model. In CPPLS, cross-validation also serves a second purpose: selecting the number
of latent variables. These two tasks are coupled, because model complexity has a direct
effect on apparent predictive performance.

The package therefore implements explicit nested cross-validation. The outer loop is used
for performance assessment, whereas the inner loop is used for model selection. This keeps
the choice of the number of latent variables separated from the final evaluation of the
model and reduces optimistic bias.

## How Nested Cross-Validation Is Implemented in CPPLS

[`nested_cv`](@ref) evaluates a CPPLS workflow in the following way:

1. The samples are split into outer folds.
2. For each outer repeat, one fold is held out as the test set and the remaining samples
    are used as the training set.
3. Within that outer training set, an inner cross-validation is run to select the number
    of latent variables.
4. A final model is fitted on the full outer training set with the selected number of
    latent variables.
5. That model is then applied to the outer test set, and the resulting score is stored.

The implementation is explicit rather than implicit. In particular:

1. Fold construction is handled by `build_folds`.
2. If `strata` are supplied, folds are created with `random_batch_indices`, which shuffles
    the samples within each stratum and then distributes them round-robin across folds.
    This helps maintain class proportions across folds in discriminant-analysis workflows.
3. The inner-loop model selection is handled by `optimize_num_latent_variables`.
4. For each inner fold, CPPLS fits one model with up to `max_components` latent variables,
    evaluates every component count from `1:max_components`, and reduces those scores to a
    single best component count by `select_fn`.
5. The final choice returned by the inner loop is the median of the per-fold best
    component counts, rounded down to an integer.

This design matters because it reflects how CPPLS is actually used in the package. The
outer score measures prediction on samples excluded from fitting, while the inner loop
decides how complex the model is allowed to be. For classification, the helper
[`cv_classification`](@ref) supplies a score function based on one-hot predictions and
normalized misclassification cost. For regression, [`cv_regression`](@ref) provides the
corresponding callbacks, using root mean squared error by default.

By default, the outer folds can either be reused or reshuffled between repeats. Reusing
the outer folds yields a fixed repeated evaluation over a predefined partition. Enabling
`reshuffle_outer_folds=true` instead rebuilds the outer split on each repeat, which gives
broader coverage of the samples across repeated assessments.

## Why Use a Permutation Procedure?

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
the permutation distribution.

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

To keep the documentation example reasonably fast, we use a fixed `gamma=0.5`, allow at
most two latent variables, and run only a small number of folds and permutations. For a
real analysis, those settings should be chosen more carefully.

In this example, class balancing is applied through `obs_weight_fn`, which is recomputed
inside each training fold. This is preferable to precomputing inverse-frequency weights on
the full dataset before cross-validation. If you instead want fixed sample-specific
weights, pass them through `fit_kwargs=(; obs_weights=...)` and they will simply be
subsetted to each training split.

```@example crossvalidation
using CPPLS
using JLD2
using Random
using Statistics

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
obs_weight_fn = (X_train, Y_train; kwargs...) ->
    invfreqweights(one_hot_to_labels(Y_train))

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

p_value = calculate_p_value(permutation_scores, observed_accuracy; tail=:upper)

(
    observed_accuracy=observed_accuracy,
    best_components=best_components,
    permutation_mean=mean(permutation_scores),
    p_value=p_value,
)
```

The returned `best_components` vector contains the component count selected inside each
outer repeat. The `observed_accuracy` is the mean nested-CV score on the real labels,
whereas `permutation_scores` summarize the same workflow after the correspondence between
predictors and class labels has been broken. The empirical p-value therefore answers a
pipeline-level question: is the observed accuracy unusual compared with what the same
analysis achieves under random label assignments? Because classification uses an
accuracy-like score here, `calculate_p_value` is called with `tail=:upper` so that large
observed scores are treated as stronger evidence against the null model.

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

suspect_idx = sortperm(outlier_scan.rate, rev=true)[1:5]

(
    sample=sample_labels[suspect_idx],
    class=classes[suspect_idx],
    tested=outlier_scan.n_tested[suspect_idx],
    flagged=outlier_scan.n_flagged[suspect_idx],
    rate=round.(outlier_scan.rate[suspect_idx]; digits=3),
)
```

These rates are not formal proof that a sample is mislabeled. They are a practical way to
prioritize inspection of samples that repeatedly fail when they are held out, which is
often exactly the situation in which class-assignment problems or unusual sample behavior
become visible.

```@docs
CPPLS.calculate_p_value
CPPLS.cv_classification
CPPLS.cv_regression
CPPLS.cv_outlier_scan
CPPLS.nested_cv
CPPLS.nested_cv_permutation
```
