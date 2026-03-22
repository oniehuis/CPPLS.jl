"""
    intervalize(values::AbstractVector{<:Real})

Convert a sequence of numeric values into adjacent `(lo, hi)` intervals by pairing
consecutive entries. This is useful for building gamma intervals from a range. If
`values` has length `1`, a single `(x, x)` interval is returned. An empty collection
throws an `ArgumentError`.

See also
[`CPPLSModel`](@ref CPPLS.CPPLSModel),
[`fit`](@ref CPPLS.fit)

# Examples
```jldoctest
julia> intervalize(0:0.5:1) == [(0.0, 0.5), (0.5, 1.0)]
true
```
"""
function intervalize(values::AbstractVector{<:Real})
    n = length(values)
    n > 0 || throw(ArgumentError("values must contain at least one element."))
    n == 1 && return [(values[1], values[1])]

    intervals = Vector{Tuple{eltype(values), eltype(values)}}(undef, n - 1)
    @inbounds for i in 1:(n - 1)
        intervals[i] = (values[i], values[i + 1])
    end
    
    intervals
end

"""
    invfreqweights(samples::AbstractVector)

Return normalized inverse-frequency weights for `samples`. Each observation receives
weight `1 / count(samples[i])`, and the resulting weights are rescaled to sum to `1`.

See also
[`fit`](@ref CPPLS.fit)

# Examples
```jldoctest
julia> invfreqweights(["a", "b", "b"]) ≈ [0.5, 0.25, 0.25]
true
```
"""
function invfreqweights(samples::AbstractVector)
    countof = Dict{eltype(samples), Int}()
    for sample in samples
        countof[sample] = get(countof, sample, 0) + 1
    end
    w = [1 / countof[sample] for sample in samples]
    w ./ sum(w)
end
