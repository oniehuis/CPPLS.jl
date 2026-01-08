"""
    fisherztrack(X::AbstractArray{<:Real, 3}, scores::AbstractVector; weights=:mean)

Interpret `X` as a three-dimensional array (a “tensor”) of shape `n × axis₁ × axis₂`,
where `n` matches the length of `scores`. For every combination of `axis₁` and `axis₂`, 
the function extracts the length-`n` slice (a single “track”) and correlates it with 
`scores`. Those correlations are Fisher z-transformed to stabilize variance, optionally 
weighted by the slice means (when `weights = :mean`), averaged for each `axis₁`, and 
finally inverse-transformed. The result is one smoothed correlation per `axis₁`, 
summarizing all its `axis₂` slices.

Arguments
- `X`: 3-D tensor whose first axis matches the observation axis of `scores`.
- `scores`: response to correlate with every slice in `X`.
- `weights`: choose `:mean` to weight by the slice means or `:none` for equal weights.

Returns a vector of correlations with length `size(X, 2)`.

# Example
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
    intervalize(values::AbstractVector{<:Real})

Convert a sequence of numeric values into adjacent `(lo, hi)` intervals by pairing
consecutive entries. This is useful for building a vector of γ-intervals from a range. For 
example, `intervalize(0:0.1:0.4)` returns `[(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4)]`.

If `values` has length 1, a single `(x, x)` interval is returned. An empty
collection throws an `ArgumentError`.
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

Robust correlation helper used inside projection diagnostics. Returns the Pearson 
correlation between `x` and `y`, falling back to `0.0` when either input is constant or 
when the computed value is not finite (e.g. `NaN` or `Inf`).

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
    separationaxis(Xscores::AbstractMatrix, Y::AbstractMatrix; method::Symbol=:centroid, 
    positive_class::Integer=1)

Given `Xscores` (rows = samples, columns = latent components) and a binary one-hot label 
matrix `Y` with two columns, this helper finds the line in score space that best separates 
the classes. It returns `(direction, scores)` where `direction` is a unit vector and 
`scores = Xscores * direction` are the signed projections, flipped if necessary so that the 
`positive_class` has larger values.

When `Xscores` has multiple columns, choose `method = :centroid` to use the difference of 
class means or `method = :lda` to use Fisher’s linear discriminant (pooled covariance). If 
`Xscores` has only one column, the function just selects the orientation that makes the 
`positive_class` larger on average.

# Examples
```
julia> X = [1.0 0.0; 2.0 1.0; 0.0 1.0; 1.0 2.0];

julia> Y = [1 0; 1 0; 0 1; 0 1];

julia> direction, scores = separationaxis(X, Y; method=:centroid);
julia> direction ≈ [0.7071067811865474, -0.7071067811865474]
true

julia> scores ≈ [0.7071067811865474, 0.7071067811865474, -0.7071067811865474, -0.7071067811865474]
true
```
"""
function separationaxis(
    Xscores::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real};
    method::Symbol = :centroid,
    positive_class::T = 1,
) where {T<:Integer}

    n, p = size(Xscores)
    p ≥ 1 || throw(ArgumentError("Expecting at least one score axis (got $p)."))
    size(Y, 1) == n || throw(ArgumentError("Row count mismatch between Xscores and Y."))
    size(Y, 2) == 2 || throw(ArgumentError("Binary only; for K > 2 use one-vs-rest."))

    all((Y .== 0) .| (Y .== 1)) || throw(ArgumentError("Y must be strictly one-hot (0/1)."))
    all(sum(Y, dims = 2) .== 1) ||
        throw(ArgumentError("Each row of Y must have exactly one '1'."))

    idx₁ = findall(Y[:, 1] .== 1)
    idx₂ = findall(Y[:, 2] .== 1)
    (!isempty(idx₁) && !isempty(idx₂)) ||
        throw(ArgumentError("Both classes must be present."))

    if p == 1
        s = vec(@view Xscores[:, 1])
        direction = ones(eltype(Xscores), 1)
        scores = copy(s)

        m₁, m₂ = mean(scores[idx₁]), mean(scores[idx₂])
        want = positive_class == 1 ? m₁ : m₂
        other = positive_class == 1 ? m₂ : m₁
        if want < other
            direction .= -direction
            scores .= -scores
        end

        return direction, scores

    else
        X₁ = @view Xscores[idx₁, :]
        X₂ = @view Xscores[idx₂, :]
        μ₁ = vec(mean(X₁, dims = 1))
        μ₂ = vec(mean(X₂, dims = 1))

        direction = if method === :centroid
            μ₁ - μ₂
        elseif method === :lda
            S₁ = cov(X₁; corrected = true)
            S₂ = cov(X₂; corrected = true)
            (S₁ + S₂) \ (μ₁ - μ₂)
        else
            throw(ArgumentError("method must be :centroid or :lda"))
        end

        if !any(isfinite, direction) || norm(direction) == 0
            throw(
                ArgumentError(
                    "Separation axis is undefined (zero vector). Check class means.",
                ),
            )
        end

        direction ./= (norm(direction) + eps(eltype(direction)))
        scores = Xscores * direction

        m₁, m₂ = mean(scores[idx₁]), mean(scores[idx₂])
        want = positive_class == 1 ? m₁ : m₂
        other = positive_class == 1 ? m₂ : m₁
        if want < other
            direction .= -direction
            scores .= -scores
        end

        return direction, scores
    end
end
