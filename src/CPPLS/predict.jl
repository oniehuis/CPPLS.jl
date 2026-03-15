"""
    predict(
        model::AbstractCPPLSFit, 
        X::AbstractMatrix{<:Real},
        n_components::Integer=size(regression_coefficients(model), 3)
    ) -> Array{Float64, 3}

Predict the response `Y` for each sample in `X` using the fitted model. Here, 
`n_components` is the number of latent CPPLS components used to form the prediction.
The result is a 3-dimensional array of size `(n_samples, n_targets, n_components)`:
the first dimension indexes samples, the second indexes response variables, and the
third indexes the number of components used. In particular, `[:, :, i]` contains the
prediction matrix obtained using the first `i` components. A `DimensionMismatch` is
thrown if `n_components` exceeds the number of components stored in the model.

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
    n_components::Integer=size(regression_coefficients(model), 3)
)
    n_samples_X = size(X, 1)
    n_targets_Y = size(Y_bar(model), 2)

    n_components ≤ size(regression_coefficients(model), 3) || throw(DimensionMismatch(
        "n_components exceeds the number of components in the model"))

    X_centered = X .- X_bar(model)
    Y_hat = similar(X, n_samples_X, n_targets_Y, n_components)

    for i = 1:n_components
        @views Y_hat[:, :, i] .= 
            (X_centered * regression_coefficients(model)[:, :, i] .+ Y_bar(model))
    end

    Y_hat
end

"""
    predictonehot(
        model::AbstractCPPLSFit,
        X::AbstractMatrix{<:Real},
        n_components::Integer=size(regression_coefficients(model), 3)
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

julia> predictonehot(model, Xnew) == [1 0; 0 1]
true
```
"""
function predictonehot(
    model::AbstractCPPLSFit,
    X::AbstractMatrix{<:Real},
    n_components::Integer=size(regression_coefficients(model), 3)
)
    predictions_to_onehot(model, predict(model, X, n_components))
end

"""
    predictions_to_onehot(
        model::AbstractCPPLSFit,
        predictions::AbstractArray{<:Real, 3}
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
[`labels_to_one_hot`](@ref CPPLS.labels_to_one_hot)

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
    Y_pred_final = Y_pred_sum .- (n_components - 1) .* Y_bar(model)

    predicted_class_indices = argmax.(eachrow(Y_pred_final))

    labels_to_one_hot(predicted_class_indices, n_classes)
end

"""
    predictsampleclasses(
        model::CPPLSFit,
        X::AbstractMatrix{<:Real},
        n_components::Integer=size(regression_coefficients(model), 3)
    ) -> AbstractVector

Generate predicted class labels from a discriminant CPPLS model and predictors `X`.
The returned vector follows the ordering in `response_labels`.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`predictonehot`](@ref CPPLS.predictonehot), 
[`predictions_to_onehot`](@ref CPPLS.predictions_to_onehot), 
[`predictions_to_sampleclasses`](@ref CPPLS.predictions_to_sampleclasses)
[`regression_coefficients`](@ref CPPLS.regression_coefficients), 

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
    n_components::Integer=size(regression_coefficients(model), 3)
)
    predictions_to_sampleclasses(model, predict(model, X, n_components))
end

"""
    predictions_to_sampleclasses(
        model::CPPLSFit,
        predictions::AbstractArray{<:Real, 3}
    ) -> AbstractVector

Convert a 3D prediction tensor (as returned by `predict`) into class labels using the 
stored `response_labels` ordering.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`analysis_mode`](@ref CPPLS.analysis_mode)
[`one_hot_to_labels`](@ref CPPLS.one_hot_to_labels)
[`predict`](@ref CPPLS.predict), 
[`predictions_to_onehot`](@ref CPPLS.predictions_to_onehot), 
[`predictonehot`](@ref CPPLS.predictonehot), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses)
[`response_labels`](@ref CPPLS.response_labels)

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
    analysis_mode(model) ≡ :discriminant || throw(ArgumentError(
        "predictions_to_sampleclasses is only defined for discriminant CPPLS models"))

    isempty(response_labels(model)) && throw(ArgumentError(
        "response_labels must be provided to map predictions to class labels"))

    n_classes = size(predictions, 2)
    length(response_labels(model)) == n_classes || throw(DimensionMismatch(
        "response_labels must have length $n_classes, " * 
        "got $(length(response_labels(model)))"))

    class_indices = one_hot_to_labels(predictions_to_onehot(model, predictions))
    response_labels(model)[class_indices]
end

"""
    project(model::CPPLSFit, X::AbstractMatrix{<:Real}) -> AbstractMatrix

Compute latent component X scores by projecting new predictors `X` with a CPPLSFit
model. The predictors are centered using `X_bar`(@ref CPPLS.X_bar^) and multiplied by 
`projection_matrix`(@ref CPPLS.projection_matrix), returning an 
`(n_samples, n_components)` X score matrix.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`projection_matrix`](@ref CPPLS.projection_matrix),
[`X_bar`](@ref CPPLS.X_bar)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(n_components=2, gamma=0.5, analysis_mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> Xscores = project(model, Xnew);

julia> size(Xscores)
(2, 2)
```
"""
project(model::CPPLSFit, X::AbstractMatrix{<:Real}) = 
    (X .- X_bar(model)) * projection_matrix(model)
