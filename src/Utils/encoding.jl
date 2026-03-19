"""
    onehot(label_indices::AbstractVector{<:Integer}, n_labels::Integer)

Convert 1-based integer label indices to a dense one-hot encoded matrix with `n_labels`
columns. Each entry in `label_indices` is interpreted as the target column of the `1`
in that sample's one-hot row, so the integer values are used directly as class indices
rather than being renumbered from the observed data. The argument `n_labels` specifies
the full size of the known class space and therefore the number of output columns, even
if some classes do not appear in `label_indices`. This method returns only the encoded
matrix.

See also
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses)

# Examples
```jldoctest
julia> onehot([4, 3, 2], 5)
3×5 Matrix{Int64}:
 0  0  0  1  0
 0  0  1  0  0
 0  1  0  0  0
```
"""
function onehot(label_indices::AbstractVector{<:Integer}, n_labels::Integer)
    n_labels ≥ 0 || throw(ArgumentError("n_labels must be nonnegative, got $n_labels"))
    all(≥(1), label_indices) || throw(ArgumentError(
        "label_indices must contain only positive 1-based class indices"))
    isempty(label_indices) || maximum(label_indices) ≤ n_labels || throw(ArgumentError(
        "n_labels must be at least maximum(label_indices) " * 
        "= $(maximum(label_indices)), got $n_labels"))

    n_samples = length(label_indices)
    one_hot = zeros(Int, n_samples, n_labels)
    @inbounds for i = 1:n_samples
        one_hot[i, label_indices[i]] = 1
    end
    one_hot
end

"""
    onehot(labels::AbstractVector)

Encode arbitrary labels such as strings, integers, or symbols into a one-hot matrix.
The function removes duplicate labels, sorts the unique labels using Julia's default
`sort` order for their type, and uses that sorted sequence as the column order of the
encoded matrix. The result is a tuple containing the encoded matrix and the ordered
labels, so predictions can be mapped back to the original domain.

See also
[`nestedcv`](@ref CPPLS.nestedcv),
[`nestedcvperm`](@ref CPPLS.nestedcvperm),
[`sampleclasses`](@ref CPPLS.sampleclasses)

# Examples
```jldoctest
julia> matrix, classes = onehot(["dog", "cat", "cat"])
([0 1; 1 0; 1 0], ["cat", "dog"])
```
"""
function onehot(labels::AbstractVector)
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
    sampleclasses(one_hot_matrix::AbstractMatrix{<:Integer})

Decode one-hot rows back into label indices by returning the column index selected in
each row. An `ArgumentError` is thrown if `one_hot_matrix` contains values other than
`0` and `1`, or if any row contains either no `1` entry or more than one `1`.

See also
[`invfreqweights`](@ref CPPLS.invfreqweights),
[`onehot`](@ref CPPLS.onehot(::AbstractVector)),
[`nestedcv`](@ref CPPLS.nestedcv),
[`nestedcvperm`](@ref CPPLS.nestedcvperm)

# Examples
```jldoctest
julia> sampleclasses([1 0 0; 0 1 0; 0 0 1])
3-element Vector{Int64}:
 1
 2
 3
```
"""
function sampleclasses(one_hot_matrix::AbstractMatrix{<:Integer})
    all(value -> value == 0 || value == 1, one_hot_matrix) || throw(ArgumentError(
        "one_hot_matrix must contain only 0/1 entries"))

    row_sums = vec(sum(one_hot_matrix, dims=2))
    all(==(1), row_sums) || throw(ArgumentError(
        "each row of one_hot_matrix must contain exactly one 1"))

    [argmax(row) for row in eachrow(one_hot_matrix)]
end
