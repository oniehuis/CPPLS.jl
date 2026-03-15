"""
    nmc(
        Y_true_one_hot::AbstractMatrix{<:Integer}, 
        Y_pred_one_hot::AbstractMatrix{<:Integer},
        weighted::Bool
    ) -> Float64

Compute the normalized misclassification cost between true and predicted one-hot label
matrices. The inputs `Y_true_one_hot` and `Y_pred_one_hot` are `(n_samples × n_classes)`
one-hot matrices of identical shape containing the ground-truth and predicted labels,
respectively. If `weighted` is `false`, the function returns the plain misclassification 
rate, computed as the `mean` of the entry-wise inequality indicator. When `weighted` is 
`true`, class weights inversely proportional to class prevalence are applied so that rare 
classes contribute equally. Returns a `Float64` between 0 and 1.

See also
[`cv_classification`](@ref CPPLS.cv_classification), 
[`nested_cv`](@ref CPPLS.nested_cv), 
[`nested_cv_permutation`](@ref CPPLS.nested_cv_permutation)

# Example
```jldoctest
julia> Y_true = [1 0 0; 0 1 0; 0 1 0];

julia> Y_pred = [1 0 0; 0 0 1; 0 1 0];

julia> nmc(Y_true, Y_pred, false) ≈ 0.2222222222222222
true

julia> nmc(Y_true, Y_pred, true) ≈ 0.25
true
```
"""
function nmc(
    Y_true_one_hot::AbstractMatrix{<:Integer},
    Y_pred_one_hot::AbstractMatrix{<:Integer},
    weighted::Bool
)

    size(Y_true_one_hot) == size(Y_pred_one_hot) || throw(
        DimensionMismatch("Input matrices must have the same dimensions"))

    n_samples = size(Y_true_one_hot, 1)
    n_samples > 0 || throw(ArgumentError(
        "Cannot compute weighted NMC: input has zero samples"))

    !weighted && return mean(Y_true_one_hot .≠ Y_pred_one_hot)

    true_labels = one_hot_to_labels(Y_true_one_hot)
    pred_labels = one_hot_to_labels(Y_pred_one_hot)

    sample_weights = invfreqweights(true_labels)
    errors = true_labels .≠ pred_labels

    weighted_error = sum(sample_weights[errors])
    clamp(weighted_error, 0.0, 1.0)
end


"""
    calculate_p_value(
        null_scores::AbstractVector{<:Real},
        observed_score::Real;
        tail::Symbol=:upper
    )

Compute a one-sided empirical p-value for an `observed_score` relative to a null
distribution of scores. The input `null_scores` is the vector of scores from null-model
or reference runs, typically obtained from label-shuffled permutations, and
`observed_score` is the score achieved by the model fit to the original data. The
argument `tail` selects the direction of the one-sided test and must be either `:upper`
or `:lower`. In both cases, a `+1` correction is applied to the numerator and
denominator to account for the observed score itself in the empirical null ranking.
With `tail=:upper`, the p-value is the fraction of null scores greater than or
numerically equal to the observed score, corresponding to a one-sided upper-tail test
appropriate when larger scores indicate stronger evidence against the null. With
`tail=:lower`, the comparison is reversed, corresponding to a one-sided lower-tail test
appropriate when smaller scores indicate stronger evidence against the null.

See also
[`nested_cv`](@ref CPPLS.nested_cv),
[`nested_cv_permutation`](@ref CPPLS.nested_cv_permutation)

# Example
```jldoctest
julia> calculate_p_value([0.4, 0.5, 0.55, 0.6], 0.58) ≈ 0.4
true

julia> calculate_p_value([0.4, 0.5, 0.55, 0.6], 0.58; tail=:lower) ≈ 0.8
true
```
"""
function calculate_p_value(
    null_scores::AbstractVector{<:Real},
    observed_score::Real;
    tail::Symbol=:upper,
)
    tail in (:upper, :lower) || throw(ArgumentError(
        "tail must be :upper or :lower, got $tail"))

    count_fn = if tail ≡ :upper
        x -> x ≥ observed_score || x ≈ observed_score
    else
        x -> x ≤ observed_score || x ≈ observed_score
    end

    (count(count_fn, null_scores) + 1) / (length(null_scores) + 1)
end
