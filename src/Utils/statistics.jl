"""
    CPPLS.fisherztrack(
        X::AbstractArray{<:Real, 3}, 
        scores::AbstractVector; 
        weights=:mean
    )

Interpret `X` as a three-dimensional array of shape `n × axis1 × axis2`, where `n`
matches the length of `scores`. For each combination of `axis1` and `axis2`, the
corresponding length-`n` slice is correlated with `scores`. These correlations are
Fisher z-transformed, averaged within each `axis1`, and inverse-transformed again. When
`weights = :mean`, the slice means are used as weights; when `weights = :none`, all
slices contribute equally. The result is a vector of length `size(X, 2)` containing one
smoothed correlation per `axis1`.

# Examples
```
julia> X = reshape(Float64[1, 2, 3, 4, 2, 3, 4, 5, 3, 5, 7, 9], 4, 3, 1);

julia> scores = [1.0, 2.0, 3.0, 4.0];

julia> fisherztrack(X, scores) ≈ [1.0, 1.0, 1.0]
true
```
"""
function fisherztrack(
    X::AbstractArray{<:Real,3},
    scores::AbstractVector{<:Real};
    weights::Symbol = :mean,
)

    n_samples, n_axis₁, n_axis₂ = size(X)
    length(scores) == n_samples ||
        throw(ArgumentError("scores length must equal size(X, 1)."))
    (weights === :mean || weights === :none) ||
        throw(ArgumentError("weights must be :mean or :none."))

    ρ = Vector{Float64}(undef, n_axis₁)

    lo = nextfloat(-1.0)
    hi = prevfloat(1.0)

    @inbounds for a₁ = 1:n_axis₁
        rs = Vector{Float64}(undef, n_axis₂)
        ws = ones(Float64, n_axis₂)

        for a₂ = 1:n_axis₂
            xs = @view X[:, a₁, a₂]
            rs[a₂] = robustcor(xs, scores)
            if weights === :mean
                ws[a₂] = mean(xs)
            end
        end

        zs = atanh.(clamp.(rs, lo, hi))
        z̄ = sum(ws .* zs) / (sum(ws) + eps(Float64))
        ρ[a₁] = tanh(z̄)
    end

    ρ
end

"""
    CPPLS.intervalize(values::AbstractVector{<:Real})

Convert a sequence of numeric values into adjacent `(lo, hi)` intervals by pairing
consecutive entries. This is useful for building gamma intervals from a range. If
`values` has length `1`, a single `(x, x)` interval is returned. An empty collection
throws an `ArgumentError`.
"""
function intervalize(values::AbstractVector{<:Real})
    n = length(values)
    n > 0 || throw(ArgumentError("values must contain at least one element."))
    n == 1 && return [(values[1], values[1])]

    intervals = Vector{Tuple{eltype(values),eltype(values)}}(undef, n - 1)
    @inbounds for i in 1:(n - 1)
        intervals[i] = (values[i], values[i + 1])
    end
    
    intervals
end

"""
    CPPLS.robustcor(x::AbstractVector, y::AbstractVector)

Return the Pearson correlation between `x` and `y`, falling back to `0.0` when either
input is constant or when the computed value is not finite. This helper is used in
diagnostic calculations that should remain well-defined for degenerate inputs.

# Examples
```
julia> CPPLS.robustcor([1, 2, 3], [3, 2, 1])
-1.0

julia> CPPLS.robustcor([1, 1, 1], [2, 3, 4])
0.0
```
"""
@inline function robustcor(x::AbstractVector, y::AbstractVector)
    (std(x) == 0 || std(y) == 0) && return 0.0
    c = cor(x, y)
    isfinite(c) ? c : 0.0
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
