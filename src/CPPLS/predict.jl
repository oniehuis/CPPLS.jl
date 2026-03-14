"""
    predict(
        model::AbstractCPPLSFit, 
        X::AbstractMatrix{<:Real},
        n_components::Integer=size(model.B, 3)
    ) -> Array{Float64, 3}

Generate predictions from a fitted CPPLS model for the rows of `X`. The result is a
`(n_samples, n_targets, n_components)` array where `[:, :, i]` contains predictions 
using the first `i` components. A `DimensionMismatch` is thrown if `n_components`
exceeds the number stored in the model.

See also
[`AbstractCPPLSFit`](@ref CPPLS.AbstractCPPLSFit), 
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predictonehot`](@ref CPPLS.predictonehot), 
[`predictions_to_onehot`](@ref CPPLS.predictions_to_onehot), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses), 
[`predictions_to_sampleclasses`](@ref CPPLS.predictions_to_sampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(n_components=2, gamma=0.5, analysis_mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> Ynew = predict(model, Xnew);

julia> size(Ynew)
(2, 2, 2)
```
"""
function predict(
    model::AbstractCPPLSFit,
    X::AbstractMatrix{<:Real},
    n_components::Integer=size(model.B, 3)
)
    n_samples_X = size(X, 1)
    n_targets_Y = size(model.Y_bar, 2)

    n_components ≤ size(model.B, 3) || throw(DimensionMismatch(
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

See also
[`AbstractCPPLSFit`](@ref CPPLS.AbstractCPPLSFit), 
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`predictions_to_onehot`](@ref CPPLS.predictions_to_onehot), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses), 
[`predictions_to_sampleclasses`](@ref CPPLS.predictions_to_sampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(n_components=2, gamma=0.5, analysis_mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> predictonehot(model, Xnew) ≈ [1 0; 0 1]
true
```
"""
function predictonehot(
    model::AbstractCPPLSFit,
    X::AbstractMatrix{<:Real},
    n_components::Integer=size(model.B, 3)
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

See also
[`AbstractCPPLSFit`](@ref CPPLS.AbstractCPPLSFit), 
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`predictonehot`](@ref CPPLS.predictonehot), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses), 
[`predictions_to_sampleclasses`](@ref CPPLS.predictions_to_sampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(n_components=2, gamma=0.5, analysis_mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> raw = predict(model, Xnew);

julia> predictions_to_onehot(model, raw) ≈ [1 0; 0 1]
true
```
"""
function predictions_to_onehot(
  model::AbstractCPPLSFit, 
  predictions::AbstractArray{<:Real, 3}
)
    n_components = size(predictions, 3)
    n_classes = size(predictions, 2)

    Y_pred_sum = sum(predictions, dims=3)[:, :, 1]
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

See also
[`AbstractCPPLSFit`](@ref CPPLS.AbstractCPPLSFit), 
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`predictonehot`](@ref CPPLS.predictonehot), 
[`predictions_to_onehot`](@ref CPPLS.predictions_to_onehot), 
[`predictions_to_sampleclasses`](@ref CPPLS.predictions_to_sampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(n_components=2, gamma=0.5, analysis_mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> predictsampleclasses(model, Xnew) == ["major", "minor"]
true
```
"""
function predictsampleclasses(
    model::CPPLSFit,
    X::AbstractMatrix{<:Real},
    n_components::Integer=size(model.B, 3)
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

See also
[`AbstractCPPLSFit`](@ref CPPLS.AbstractCPPLSFit), 
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`predictonehot`](@ref CPPLS.predictonehot), 
[`predictions_to_onehot`](@ref CPPLS.predictions_to_onehot), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(n_components=2, gamma=0.5, analysis_mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> raw = predict(model, Xnew);

julia> predictions_to_sampleclasses(model, raw) == ["major", "minor"]
true
```
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
    project(model::CPPLSFit, X::AbstractMatrix{<:Real}) -> AbstractMatrix

Compute latent component scores by projecting new predictors `X` with a fitted CPPLS
model that stores a projection matrix `R` and predictor means `X_bar`.
In normal usage this means a full `CPPLSFit`. The predictors are centered using
`model.X_bar` and multiplied by `model.R`, returning an
`(n_samples, n_components)` score matrix.

Here, `model.X_bar` is the predictor mean vector stored during fitting and used to center
new data in the same way as the training data, while `model.R` is the projection matrix
that maps centered predictors into the latent component score space.

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(n_components=2, gamma=0.5, analysis_mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> xscores = project(model, Xnew);

julia> size(xscores)
(2, 2)
```
"""
project(model::CPPLSFit, X::AbstractMatrix{<:Real}) = (X .- model.X_bar) * model.R
