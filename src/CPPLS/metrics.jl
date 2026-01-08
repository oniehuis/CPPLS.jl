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
    weighted::Bool,
)

    size(Y_true_one_hot) == size(Y_pred_one_hot) ||
        throw(DimensionMismatch("Input matrices must have the same dimensions"))

    n_samples = size(Y_true_one_hot, 1)
    n_samples > 0 || error("Cannot compute weighted NMC: input has zero samples")

    !weighted && return mean(Y_true_one_hot .≠ Y_pred_one_hot)

    true_labels = one_hot_to_labels(Y_true_one_hot)
    pred_labels = one_hot_to_labels(Y_pred_one_hot)

    class_counts = countmap(true_labels)
    inv_freqs = Dict(k => n_samples / v for (k, v) in class_counts)
    total_weight = sum(inv_freqs[k] * class_counts[k] for k in keys(inv_freqs))
    class_weights = Dict(k => inv_freqs[k] / total_weight for k in keys(inv_freqs))

    sample_weights = getindex.(Ref(class_weights), true_labels)
    errors = true_labels .≠ pred_labels

    weighted_error = sum(sample_weights[errors])
    clamp(weighted_error, 0.0, 1.0)
end


"""
    calculate_p_value(permutation_accuracies::AbstractVector{<:Real},
                      model_accuracy::Float64)

Compute an empirical p-value from permutation test accuracies. Counts how many permutation 
accuracies are less than or numerically equal to the observed `model_accuracy`, divides by 
`length(permutation_accuracies) + 1` to include the observed model in the denominator.

Arguments
- `permutation_accuracies`: vector of accuracies from label-shuffled runs.
- `model_accuracy`: accuracy achieved by the true model.

# Example
```
julia> calculate_p_value([0.4, 0.5, 0.55, 0.6], 0.58) ≈ 0.6
true
```
"""
function calculate_p_value(
    permutation_accuracies::AbstractVector{<:Real},
    model_accuracy::Float64,
)

    (
        count(x -> x ≤ model_accuracy || x ≈ model_accuracy, permutation_accuracies) /
        (length(permutation_accuracies) + 1)
    )
end
