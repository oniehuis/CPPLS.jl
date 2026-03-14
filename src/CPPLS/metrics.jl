"""
    CPPLS.nmc(Y_true_one_hot::AbstractMatrix{<:Integer}, 
        Y_pred_one_hot::AbstractMatrix{<:Integer}, weighted::Bool)

Compute the normalized misclassification cost between true and predicted one-hot label 
matrices. If `weighted` is `false`, the function returns the plain misclassification rate 
(`mean` of entry-wise inequality). When `true`, class weights inversely proportional to 
their prevalence are applied, so rare classes contribute equally.

Arguments
- `Y_true_one_hot`: `(n_samples × n_classes)` ground truth one-hot labels.
- `Y_pred_one_hot`: predicted one-hot labels of the same shape.
- `weighted`: toggle class-balanced weighting.

Returns a `Float64` between 0 and 1.

# Example
```
julia> Y_true = [1 0 0; 0 1 0; 0 1 0];

julia> Y_pred = [1 0 0; 0 0 1; 0 1 0];

julia> CPPLS.nmc(Y_true, Y_pred, false) ≈ 0.2222222222222222
true

julia> CPPLS.nmc(Y_true, Y_pred, true) ≈ 0.25
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
        permutation_scores::AbstractVector{<:Real},
        observed_score::Float64;
        tail::Symbol=:upper,
    )

Compute an empirical p-value from permutation scores. With `tail=:upper`, the p-value is
the fraction of permutation scores greater than or numerically equal to the observed
score, divided by `length(permutation_scores) + 1` to include the observed model in the
denominator. This is appropriate for accuracy-like metrics where larger values are better.
With `tail=:lower`, the comparison is reversed, which is appropriate for error-like
metrics where smaller values are better.

Arguments
- `permutation_scores`: vector of scores from label-shuffled runs.
- `observed_score`: score achieved by the model fit to the original data.
- `tail`: comparison direction, either `:upper` or `:lower`.

# Example
```
julia> calculate_p_value([0.4, 0.5, 0.55, 0.6], 0.58) ≈ 0.4
true

julia> calculate_p_value([0.4, 0.5, 0.55, 0.6], 0.58; tail=:lower) ≈ 0.6
true
```
"""
function calculate_p_value(
    permutation_scores::AbstractVector{<:Real},
    observed_score::Float64;
    tail::Symbol=:upper,
)
    tail in (:upper, :lower) || throw(ArgumentError(
        "tail must be :upper or :lower, got $tail"))

    count_fn = if tail === :upper
        x -> x ≥ observed_score || x ≈ observed_score
    else
        x -> x ≤ observed_score || x ≈ observed_score
    end

    count(count_fn, permutation_scores) / (length(permutation_scores) + 1)
end
