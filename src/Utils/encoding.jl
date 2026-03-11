"""
    labels_to_one_hot(label_indices::AbstractVector{<:Integer}, n_labels::Integer)

Convert 1-based integer label indices to a dense one-hot encoded matrix with `n_labels`
columns. This method assumes the set of classes is already known and returns only the
encoded matrix.

# Examples
```
julia> labels_to_one_hot([1, 3, 2], 3)
3×3 Matrix{Int64}:
 1  0  0
 0  0  1
 0  1  0
```
"""
function labels_to_one_hot(label_indices::AbstractVector{<:Integer}, n_labels::Integer)
    n_samples = length(label_indices)
    one_hot = zeros(Int, n_samples, n_labels)
    @inbounds for i = 1:n_samples
        one_hot[i, label_indices[i]] = 1
    end
    one_hot
end


"""
    labels_to_one_hot(labels::AbstractVector)

Encode arbitrary labels such as strings, integers, or symbols into a one-hot matrix,
automatically determining the label ordering. The result is a tuple containing the
encoded matrix and the ordered labels, so predictions can be mapped back to the original
domain.

# Examples
```
julia> matrix, classes = labels_to_one_hot(["cat", "dog", "cat"])
([1 0; 0 1; 1 0], ["cat", "dog"])
```
"""
function labels_to_one_hot(labels::AbstractVector)
    unique_labels = sort(collect(Set(labels)))  # consistent label order
    label_to_index = Dict(label => i for (i, label) in enumerate(unique_labels))

    num_classes = length(unique_labels)
    num_samples = length(labels)
    one_hot = zeros(Int, num_samples, num_classes)

    @inbounds for (i, label) in enumerate(labels)
        idx = label_to_index[label]
        one_hot[i, idx] = 1
    end

    one_hot, unique_labels
end


"""
    one_hot_to_labels(one_hot_matrix::AbstractMatrix{<:Integer})

Decode one-hot rows back into label indices by selecting the column of the maximum entry
in each row. This works with integer-valued matrices whose rows encode a single selected
label.

# Examples
```
julia> one_hot_to_labels([1 0 0; 0 1 0; 0 0 1])
3-element Vector{Int64}:
 1
 2
 3
```
"""
one_hot_to_labels(one_hot_matrix::AbstractMatrix{<:Integer}) =
    [argmax(row) for row in eachrow(one_hot_matrix)]
