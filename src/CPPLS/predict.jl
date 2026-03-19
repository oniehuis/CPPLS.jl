"""
    predict(
        model::AbstractCPPLSFit, 
        X::AbstractMatrix{<:Real},
        ncomponents::Integer=size(regression_coefficients(model), 3)
    ) -> Array{Float64, 3}

Predict the response `Y` for each sample in `X` using the fitted model. Here, 
`ncomponents` is the number of latent CPPLS components used to form the prediction.
The result is a 3-dimensional array of size `(n_samples, n_targets, ncomponents)`:
the first dimension indexes samples, the second indexes response variables, and the
third indexes the number of components used. In particular, `[:, :, i]` contains the
prediction matrix obtained using the first `i` components. A `DimensionMismatch` is
thrown if `ncomponents` exceeds the number of components stored in the model.

See also
[`AbstractCPPLSFit`](@ref CPPLS.AbstractCPPLSFit),
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predictonehot`](@ref CPPLS.predictonehot), 
[`onehot`](@ref CPPLS.onehot), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(ncomponents=2, gamma=0.5, mode=:discriminant);

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
    ncomponents::Integer=size(regression_coefficients(model), 3)
)
    n_samples_X = size(X, 1)
    n_targets_Y = size(ybar(model), 2)

    ncomponents ≤ size(regression_coefficients(model), 3) || throw(DimensionMismatch(
        "ncomponents exceeds the number of components in the model"))

    X_centered = X .- xbar(model)
    Y_hat = similar(X, n_samples_X, n_targets_Y, ncomponents)

    for i = 1:ncomponents
        @views Y_hat[:, :, i] .= 
            (X_centered * regression_coefficients(model)[:, :, i] .+ ybar(model))
    end

    Y_hat
end

"""
    predictonehot(
        model::AbstractCPPLSFit,
        X::AbstractMatrix{<:Real},
        ncomponents::Integer=size(regression_coefficients(model), 3)
    ) -> Matrix{Int}

Generate one-hot encoded class predictions from a fitted CPPLS model and predictors `X`.
This calls `predict`, sums predictions across components, corrects for repeated mean
addition, and assigns each sample to the highest-scoring class.

See also
[`AbstractCPPLSFit`](@ref CPPLS.AbstractCPPLSFit), 
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`onehot`](@ref CPPLS.onehot), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(ncomponents=2, gamma=0.5, mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> predictonehot(model, Xnew) == [1 0; 0 1]
true
```
"""
function predictonehot(
    model::AbstractCPPLSFit,
    X::AbstractMatrix{<:Real},
    ncomponents::Integer=size(regression_coefficients(model), 3)
)
    onehot(model, predict(model, X, ncomponents))
end

"""
    onehot(
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
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses)
[`onehot`](@ref CPPLS.onehot)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(ncomponents=2, gamma=0.5, mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> raw = predict(model, Xnew);

julia> onehot(model, raw) ≈ [1 0; 0 1]
true
```
"""
function onehot(
  model::AbstractCPPLSFit, 
  predictions::AbstractArray{<:Real, 3}
)
    ncomponents = size(predictions, 3)
    n_classes = size(predictions, 2)

    Y_pred_sum = sum(predictions, dims=3)[:, :, 1]
    Y_pred_final = Y_pred_sum .- (ncomponents - 1) .* ybar(model)

    predicted_class_indices = argmax.(eachrow(Y_pred_final))

    onehot(predicted_class_indices, n_classes)
end

"""
    predictsampleclasses(
        model::CPPLSFit,
        X::AbstractMatrix{<:Real},
        ncomponents::Integer=size(regression_coefficients(model), 3)
    ) -> AbstractVector

Generate predicted class labels from a discriminant CPPLS model and predictors `X`.
The returned vector follows the ordering in `responselabels`.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`predictonehot`](@ref CPPLS.predictonehot), 
[`onehot`](@ref CPPLS.onehot), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses)
[`regression_coefficients`](@ref CPPLS.regression_coefficients), 

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(ncomponents=2, gamma=0.5, mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> predictsampleclasses(model, Xnew) == ["major", "minor"]
true
```
"""
function predictsampleclasses(
    model::CPPLSFit,
    X::AbstractMatrix{<:Real},
    ncomponents::Integer=size(regression_coefficients(model), 3)
)
    predictsampleclasses(model, predict(model, X, ncomponents))
end

"""
    predictsampleclasses(
        model::CPPLSFit,
        predictions::AbstractArray{<:Real, 3}
    ) -> AbstractVector

Convert a 3D prediction tensor (as returned by `predict`) into class labels using the 
stored `responselabels` ordering.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`mode`](@ref CPPLS.mode)
[`sampleclasses`](@ref CPPLS.sampleclasses)
[`predict`](@ref CPPLS.predict), 
[`onehot`](@ref CPPLS.onehot), 
[`predictonehot`](@ref CPPLS.predictonehot), 
[`predictsampleclasses`](@ref CPPLS.predictsampleclasses)
[`responselabels`](@ref CPPLS.responselabels)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(ncomponents=2, gamma=0.5, mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> raw = predict(model, Xnew);

julia> predictsampleclasses(model, raw) == ["major", "minor"]
true
```
"""
function predictsampleclasses(
    model::CPPLSFit,
    predictions::AbstractArray{<:Real,3}
)
    mode(model) ≡ :discriminant || throw(ArgumentError(
        "predictsampleclasses is only defined for discriminant CPPLS models"))

    isempty(responselabels(model)) && throw(ArgumentError(
        "responselabels must be provided to map predictions to class labels"))

    n_classes = size(predictions, 2)
    length(responselabels(model)) == n_classes || throw(DimensionMismatch(
        "responselabels must have length $n_classes, " * 
        "got $(length(responselabels(model)))"))

    class_indices = sampleclasses(onehot(model, predictions))
    responselabels(model)[class_indices]
end

"""
    project(model::CPPLSFit, X::AbstractMatrix{<:Real}) -> AbstractMatrix

Compute latent component X scores by projecting new predictors `X` with a CPPLSFit
model. The predictors are centered using `xbar`(@ref CPPLS.xbar) and multiplied by 
`projectionmatrix`(@ref CPPLS.projectionmatrix), returning an 
`(n_samples, ncomponents)` X score matrix.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`projectionmatrix`](@ref CPPLS.projectionmatrix),
[`xbar`](@ref CPPLS.xbar)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> spec = CPPLSSpec(ncomponents=2, gamma=0.5, mode=:discriminant);

julia> model = fit(spec, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> xscores = project(model, Xnew);

julia> size(xscores)
(2, 2)
```
"""
project(model::CPPLSFit, X::AbstractMatrix{<:Real}) = 
    (X .- xbar(model)) * projectionmatrix(model)
