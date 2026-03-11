"""
    predict(model::AbstractCPPLSFit, X::AbstractMatrix{<:Real},
        n_components::Integer=size(model.B, 3)) -> Array{Float64, 3}

Generate predictions from a fitted CPPLS model for the rows of `X`. The result is a
`(n_samples, n_targets, n_components)` array where `[:, :, i]` contains predictions 
using the first `i` components. A `DimensionMismatch` is thrown if `n_components`
exceeds the number stored in the model.

# Examples
```
julia> coeffs = reshape(Float64[0.5, 1.0], 2, 1, 1);  # two predictors, one target

julia> X_bar = zeros(1, 2); Y_bar = reshape([0.0], 1, 1);

julia> model = CPPLSFitLight(coeffs, X_bar, Y_bar, :regression);

julia> Xnew = [1.0 2.0; 3.0 4.0];

julia> predict(model, Xnew) ≈ [2.5; 5.5]
true
```
"""
function predict(
    model::AbstractCPPLSFit,
    X::AbstractMatrix{<:Real},
    n_components::Int=size(model.B, 3)
)
    n_samples_X = size(X, 1)
    n_targets_Y = size(model.Y_bar, 2)

    n_components ≤ size(model.B, 3) ||throw(DimensionMismatch(
        "n_components exceeds the number of components in the model"))

    X_centered = X .- model.X_bar
    Y_hat = similar(X, n_samples_X, n_targets_Y, n_components)

    for i = 1:n_components
        @views Y_hat[:, :, i] .= (X_centered * model.B[:, :, i] .+ model.Y_bar)
    end

    Y_hat
end

"""
    predictonehot(
        model::AbstractCPPLSFit,
        X::AbstractMatrix{<:Real},
        n_components::Integer=size(model.B, 3),
    ) -> Matrix{Int}

Generate one-hot encoded class predictions from a fitted CPPLS model and predictors `X`.
This calls `predict`, sums predictions across components, corrects for repeated mean
addition, and assigns each sample to the highest-scoring class.

# Examples
```
julia> coeffs = reshape(Float64[1, -1, 0.5, -0.5], 2, 2, 1);  # two predictors, two classes

julia> X_bar = zeros(1, 2); Y_bar = reshape([0.0 0.0], 1, 2);

julia> model = CPPLSFitLight(coeffs, X_bar, Y_bar, :regression);

julia> Xnew = [2.0 1.0; 0.5 3.0];

julia> predictonehot(model, Xnew) ≈ [1 0; 0 1]
true
```
"""
function predictonehot(
    model::AbstractCPPLSFit,
    X::AbstractMatrix{<:Real},
    n_components::Int=size(model.B, 3)
)
    predictions_to_onehot(model, predict(model, X, n_components))
end

"""
    predictions_to_onehot(
        model::AbstractCPPLSFit,
        predictions::AbstractArray{<:Real, 3},
    ) -> Matrix{Int}

Convert a 3D prediction tensor (as returned by `predict`) into a one-hot encoded matrix.
Predictions are summed across components and corrected for repeated mean addition before
selecting the highest-scoring class for each sample.

# Examples
```
julia> coeffs = reshape(Float64[1, -1, 0.5, -0.5], 2, 2, 1);  # two predictors, two classes

julia> X_bar = zeros(1, 2); Y_bar = reshape([0.0 0.0], 1, 2);

julia> model = CPPLSFitLight(coeffs, X_bar, Y_bar, :regression);

julia> Xnew = [2.0 1.0; 0.5 3.0];

julia> raw = predict(model, Xnew);

julia> predictions_to_onehot(model, raw) ≈ [1 0; 0 1]
true
```
"""
function predictions_to_onehot(
  model::AbstractCPPLSFit, 
  predictions::AbstractArray{<:Real,3}
)
    n_components = size(predictions, 3)
    n_classes = size(predictions, 2)

    Y_pred_sum = sum(predictions, dims = 3)[:, :, 1]
    Y_pred_final = Y_pred_sum .- (n_components - 1) .* model.Y_bar

    predicted_class_indices = argmax.(eachrow(Y_pred_final))

    labels_to_one_hot(predicted_class_indices, n_classes)
end

"""
    predictsampleclasses(
        model::CPPLSFit,
        X::AbstractMatrix{<:Real},
        n_components::Integer=size(model.B, 3),
    ) -> AbstractVector

Generate predicted class labels from a discriminant CPPLS model and predictors `X`.
The returned vector follows the ordering in `response_labels`.
"""
function predictsampleclasses(
    model::CPPLSFit,
    X::AbstractMatrix{<:Real},
    n_components::Int=size(model.B, 3)
)
    predictions_to_sampleclasses(model, predict(model, X, n_components))
end

"""
    predictions_to_sampleclasses(
        model::CPPLSFit,
        predictions::AbstractArray{<:Real, 3},
    ) -> AbstractVector

Convert a 3D prediction tensor (as returned by `predict`) into class labels using the
stored `response_labels` ordering.
"""
function predictions_to_sampleclasses(
    model::CPPLSFit,
    predictions::AbstractArray{<:Real,3}
)
    model.analysis_mode ≡ :discriminant || throw(ArgumentError(
        "predictions_to_sampleclasses is only defined for discriminant CPPLS models"))

    isempty(model.response_labels) && throw(ArgumentError(
        "response_labels must be provided to map predictions to class labels"))

    n_classes = size(predictions, 2)
    length(model.response_labels) == n_classes || throw(DimensionMismatch(
        "response_labels must have length $n_classes, " * 
        "got $(length(model.response_labels))"))

    class_indices = one_hot_to_labels(predictions_to_onehot(model, predictions))
    model.response_labels[class_indices]
end

"""
    project(model::AbstractCPPLSFit, X::AbstractMatrix{<:Real}) -> AbstractMatrix

Compute latent component scores by projecting new predictors `X` with a fitted CPPLS
model. The predictors are centered using `model.X_bar` and multiplied by `model.R`,
returning an `(n_samples, n_components)` score matrix.

# Examples
```
julia> struct DemoCPPLS <: CPPLS.AbstractCPPLSFit
           R::Matrix{Float64}
           X_bar::Matrix{Float64}
       end

julia> proj = reshape([1.0, 0.5], 2, 1)
2×1 Matrix{Float64}:
 1.0
 0.5

julia> demo = DemoCPPLS(proj, reshape([0.5, 0.5], 1, :));

julia> project(demo, [1.0 2.0; 3.0 4.0]) ≈ [1.25; 4.25]
true
```
In practice, `demo` would be the `CPPLSFit` object returned by `fit_cppls`, which already
contains the appropriate R matrix and predictor means.
"""
project(model::AbstractCPPLSFit, X::AbstractMatrix{<:Real}) =
    (X .- model.X_bar) * model.R
