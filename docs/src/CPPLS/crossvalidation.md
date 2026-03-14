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


```@docs
CPPLS.calculate_p_value
CPPLS.cv_classification
CPPLS.cv_regression
CPPLS.cv_outlier_scan
CPPLS.nested_cv
CPPLS.nested_cv_permutation
```
