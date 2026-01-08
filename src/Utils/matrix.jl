"""
    find_invariant_and_variant_columns(M::AbstractMatrix)

Scan each column of `M` and split them into two index vectors: columns whose entries are all 
identical (`invariant_columns`) and columns that contain any variation (`variant_columns`). 
Useful for removing zero-variance predictors before fitting.

# Example
```
julia> invariant, variant = find_invariant_and_variant_columns([1 2 2; 1 3 4])
([1], [2, 3])
```
"""
function find_invariant_and_variant_columns(M::AbstractMatrix)
    invariant_columns = Int[]
    variant_columns = Int[]

    for j in axes(M, 2)
        col = @view M[:, j]
        if all(x -> x == col[1], col)
            push!(invariant_columns, j)
        else
            push!(variant_columns, j)
        end
    end

    invariant_columns, variant_columns
end
