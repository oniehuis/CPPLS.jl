"""
    predict(cppls::CPPLS, X::AbstractMatrix{<:Real},
        n_components::Integer=size(cppls.regression_coefficients, 3)) -> Array{Float64, 3}

Generate predictions from a fitted CPPLS model for a given input matrix `X`.

# Arguments
- `cppls`: A fitted CPPLS model, containing regression coefficients and mean values of 
  predictors and responses.
- `X`: A matrix of predictor variables of size `(n_samples_X, n_features)`.
- `n_components` (optional): Number of CPPLS components to use for prediction. Defaults to 
  the full number trained in the model. Must not exceed the number available.

# Returns
- A 3-dimensional array of shape `(n_samples_X, n_targets_Y, n_components)`:
  - `n_samples_X`: Number of input samples (rows of `X`)
- `n_targets_Y`: Number of target variables in the CPPLS model
- `n_components`: Number of components used for prediction
  Each `[:,:,i]` slice corresponds to predictions using the first `i` components.

# Example
```
julia> coeffs = reshape(Float64[0.5, 1.0], 2, 1, 1);  # two predictors, one target

julia> X_mean = zeros(1, 2); Y_mean = reshape([0.0], 1, 1);

julia> model = CPPLSLight(coeffs, X_mean, Y_mean, :regression);

julia> Xnew = [1.0 2.0; 3.0 4.0];

julia> predict(model, Xnew) ≈ [2.5; 5.5]
true
```
"""
function predict(
    cppls::AbstractCPPLS,
    X::AbstractMatrix{<:Real},
    n_components::T = size(cppls.regression_coefficients, 3),
) where {T<:Integer}

    n_samples_X = size(X, 1)
    n_targets_Y = size(cppls.Y_means, 2)

    if n_components > size(cppls.regression_coefficients, 3)
        throw(
            DimensionMismatch("n_components exceeds the number of components in the model"),
        )
    end

    X_centered = X .- cppls.X_means
    fitted_values = similar(X, n_samples_X, n_targets_Y, n_components)

    for i = 1:n_components
        @views fitted_values[:, :, i] .=
            (X_centered * cppls.regression_coefficients[:, :, i] .+ cppls.Y_means)
    end

    fitted_values
end

"""
    predictonehot(cppls::AbstractCPPLS, predictions::AbstractArray{<:Real, 3}) -> Matrix{Int}

Convert a 3D array of predictions from a CPPLS model into a one-hot encoded 2D matrix, 
assigning each sample to the class with the highest summed prediction across components, 
after adjusting for overcounted means.

# Arguments
- `cppls`: A fitted CPPLS model object containing the mean response vector (`Y_means`).
- `predictions`: A 3D array of predictions with dimensions `(n_samples_X, n_targets_Y, 
  n_components)`. 
  Typically, this represents predicted values for multiple samples, targets, and components.

# Returns
- A 2D matrix of size `(n_samples_X, n_targets_Y)` where each row is a one-hot encoded 
  vector indicating the target class with the highest summed prediction across components.

# Details
- Sums predictions across all components for each sample and class.
- Adjusts the summed predictions by subtracting `(n_components - 1)` times the mean 
  response, to correct for repeated addition of the mean in each component.
- For each sample, finds the class index with the highest adjusted prediction.
- Converts the predicted class indices to a one-hot encoded matrix.

# Example
```
julia> coeffs = reshape(Float64[1, -1, 0.5, -0.5], 2, 2, 1);  # two predictors, two classes

julia> X_mean = zeros(1, 2); Y_mean = reshape([0.0 0.0], 1, 2);

julia> model = CPPLSLight(coeffs, X_mean, Y_mean, :regression);

julia> Xnew = [2.0 1.0; 0.5 3.0];

julia> raw = predict(model, Xnew);  # size 2×2×1

julia> raw ≈ [1.0 0.5; -2.5 -1.25]
true

julia> predictonehot(model, raw) ≈ [1 0; 0 1]
true
```
"""
function predictonehot(cppls::AbstractCPPLS, predictions::AbstractArray{<:Real,3})
    n_components = size(predictions, 3)
    n_classes = size(predictions, 2)

    Y_pred_sum = sum(predictions, dims = 3)[:, :, 1]
    Y_pred_final = Y_pred_sum .- (n_components - 1) .* cppls.Y_means

    predicted_class_indices = argmax.(eachrow(Y_pred_final))

    labels_to_one_hot(predicted_class_indices, n_classes)
end

"""
    project(cppls::AbstractCPPLS, X::AbstractMatrix{<:Real}) -> AbstractMatrix

Compute latent component scores by projecting new predictors `X` with a fitted CPPLS model.

# Arguments
- `cppls`: Any CPPLS model (e.g., `CPPLS` or `CPPLSLight`) providing `X_means` and
  `projection`.
- `X`: Predictor matrix shaped like the training data (`n_samples × n_features`).

# Returns
- Matrix of size `(n_samples, n_components)` containing the component scores.

# Details
- Centers `X` by subtracting `cppls.X_means`, then multiplies by the projection matrix.

# Example
```
julia> struct DemoCPPLS <: CPPLS.AbstractCPPLS
           projection::Matrix{Float64}
           X_means::Matrix{Float64}
       end

julia> proj = reshape([1.0, 0.5], 2, 1)
2×1 Matrix{Float64}:
 1.0
 0.5

julia> demo = DemoCPPLS(proj, reshape([0.5, 0.5], 1, :));

julia> project(demo, [1.0 2.0; 3.0 4.0]) ≈ [1.25; 4.25]
true
```
In practice, `demo` would be the `CPPLS` object returned by `fit_cppls`, which already
contains the appropriate projection matrix and predictor means.
"""
project(cppls::AbstractCPPLS, X::AbstractMatrix{<:Real}) =
    (X .- cppls.X_means) * cppls.projection

"""
    decision_line(cppls::CPPLS; dims=(1, 2), n_components=maximum(dims))

Return the discriminant hyperplane restricted to the selected score dimensions as a
tuple `(xs, ys, intercept, normal)`. Use `xs`/`ys` to draw the line directly on a score
plot; `intercept` and `normal` describe the underlying equation `normal⋅scores + intercept = 0`.

# Keywords
- `dims`: Two component indices that define the score plane. Defaults to the first two
  components shown in typical score plots.
- `n_components`: Number of latent components used in the classifier. By default matches
  the largest index in `dims`, ensuring the decision line reflects the same submodel
  whose scores are plotted.

# Notes
- Works for discriminant CPPLS fits with exactly two response classes. The prediction step
  is linear in the latent scores, so fitting `normal⋅scores + intercept = (Ŷ₁ - Ŷ₂)` via
  least squares recovers the separating line in the requested score plane.
- The returned `(xs, ys)` span the range of the training scores along `dims`. Offset a small
  margin so the boundary extends slightly beyond the cloud of points.
"""
function decision_line(
    cppls::CPPLS;
    dims::NTuple{2,Int} = (1, 2),
    n_components::Integer = maximum(dims),
)
    cppls.analysis_mode === :discriminant ||
        throw(ArgumentError("decision_line is only defined for discriminant CPPLS models"))
    size(cppls.regression_coefficients, 2) == 2 ||
        throw(ArgumentError("decision_line currently supports exactly two classes"))

    n_components ≥ maximum(dims) ||
        throw(ArgumentError("n_components must be at least as large as the requested dims"))
    n_components ≤ size(cppls.regression_coefficients, 3) ||
        throw(ArgumentError("n_components exceeds the fitted CPPLS model"))
    all(d -> 1 ≤ d ≤ size(cppls.X_scores, 2), dims) ||
        throw(ArgumentError("dims must refer to stored score columns"))

    scores = cppls.X_scores[:, collect(dims)]
    predictions = cppls.fitted_values[:, :, n_components]
    class_diff = predictions[:, 1] .- predictions[:, 2]

    A = hcat(ones(size(scores, 1)), scores)
    β = A \ class_diff
    intercept = β[1]
    normal = β[2:3]

    function padded_range(values)
        lo, hi = extrema(values)
        Δ = hi - lo
        pad = Δ ≈ 0 ? 1.0 : 0.05 * Δ
        lo - pad, hi + pad
    end

    x_lo, x_hi = padded_range(scores[:, 1])
    y_lo, y_hi = padded_range(scores[:, 2])
    xs = Float64[x_lo, x_hi]

    ys = if isapprox(normal[2], 0.0; atol = eps(Float64))
        fill(-intercept / normal[1], 2)
    else
        (-intercept .- normal[1] .* xs) ./ normal[2]
    end

    if isapprox(normal[2], 0.0; atol = eps(Float64))
        ys = Float64[y_lo, y_hi]
        xs .= -intercept / normal[1]
    end

    (xs = xs, ys = ys, intercept = intercept, normal = normal)
end
