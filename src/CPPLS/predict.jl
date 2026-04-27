"""
    predict(
        mf::AbstractCPPLSFit, 
        X::AbstractMatrix{<:Real},
        ncomponents::Integer=size(coefall(mf), 3)
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
[`onehot`](@ref CPPLS.onehot), 
[`onehot`](@ref CPPLS.onehot), 
[`sampleclasses`](@ref CPPLS.sampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> classes = categorical(classes);

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, analysis_mode=:discriminant);

julia> mf = fit(m, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> Ynew = predict(mf, Xnew);

julia> size(Ynew)
(2, 2, 2)
```
"""
function predict(
    m::AbstractCPPLSFit,
    X::AbstractMatrix{<:Real},
    ncomponents::Integer=size(coefall(m), 3)
)
    nrow_X = size(X, 1)
    ncol_Y = length(ystd(m))

    ncomponents ≤ size(coefall(m), 3) || throw(DimensionMismatch(
        "ncomponents exceeds the number of components in the model"))

    X_norm = (X .- xmean(m)') ./ xstd(m)'
    
    Y_hat = similar(X, nrow_X, ncol_Y, ncomponents)

    for i = 1:ncomponents
        @views Y_hat[:, :, i] .= (X_norm * coefall(m)[:, :, i]) .* ystd(m)'
    end

    Y_hat
end

"""
    onehot(
        m::AbstractCPPLSFit,
        X::AbstractMatrix{<:Real},
        ncomponents::Integer=size(coefall(m), 3)
    ) -> Matrix{Int}

Generate one-hot encoded class predictions from a fitted CPPLS model and predictors `X`.
This calls `predict`, sums predictions across components, and assigns each sample to the
highest-scoring class.

See also
[`AbstractCPPLSFit`](@ref CPPLS.AbstractCPPLSFit), 
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`onehot`](@ref CPPLS.onehot), 
[`sampleclasses`](@ref CPPLS.sampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> classes = categorical(classes);

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, analysis_mode=:discriminant);

julia> mf = fit(m, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> onehot(mf, Xnew) == [1 0; 0 1]
true
```
"""
function onehot(
    mf::AbstractCPPLSFit,
    X::AbstractMatrix{<:Real},
    ncomponents::Integer=size(coefall(mf), 3)
)
    onehot(mf, predict(mf, X, ncomponents))
end

"""
    onehot(
        mf::AbstractCPPLSFit,
        predictions::AbstractArray{<:Real, 3}
    ) -> Matrix{Int}

Convert a 3D prediction tensor (as returned by `predict`) into a one-hot encoded matrix.
Predictions are summed across components before selecting the highest-scoring class for
each sample.

See also
[`AbstractCPPLSFit`](@ref CPPLS.AbstractCPPLSFit), 
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`onehot`](@ref CPPLS.onehot), 
[`sampleclasses`](@ref CPPLS.sampleclasses)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> classes = categorical(classes);

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, analysis_mode=:discriminant);

julia> mf = fit(m, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> raw = predict(mf, Xnew);

julia> onehot(mf, raw) ≈ [1 0; 0 1]
true
```
"""
function onehot(
  mf::AbstractCPPLSFit, 
  predictions::AbstractArray{<:Real, 3}
)
    size(predictions, 3) > 0 || throw(ArgumentError(
        "predictions must contain at least one component slice"))

    n_classes = size(predictions, 2)

    Y_pred_final = sum(predictions, dims=3)[:, :, 1]

    predicted_class_indices = argmax.(eachrow(Y_pred_final))

    onehot(predicted_class_indices, n_classes)
end

function class_response_columns(mf::CPPLSFit)
    cols = class_response_columns(sampleclasses(mf), responselabels(mf))
    if !isnothing(cols)
        return cols
    end

    Ytrain = fitted(mf) + residuals(mf)
    is_one_hot_matrix(Ytrain) && return collect(1:size(Ytrain, 2))

    throw(ArgumentError(
        "This fitted model does not define class-response columns. Pass categorical " *
        "labels to `fit`, or provide `sampleclasses` plus matching `responselabels` " *
        "for the class-indicator part of a custom response matrix."
    ))
end

function onehot(
    mf::CPPLSFit,
    predictions::AbstractArray{<:Real, 3}
)
    size(predictions, 3) > 0 || throw(ArgumentError(
        "predictions must contain at least one component slice"))

    classcols = class_response_columns(mf)
    predicted_scores = @views sum(predictions[:, classcols, :]; dims=3)[:, :, 1]
    predicted_class_indices = argmax.(eachrow(predicted_scores))
    onehot(predicted_class_indices, length(classcols))
end

"""
    predictclasses(
        mf::CPPLSFit,
        X::AbstractMatrix{<:Real},
        ncomponents::Integer=size(coefall(mf), 3)
    ) -> AbstractVector

Generate predicted class labels from a discriminant CPPLS model and predictors `X`.
The returned vector follows the ordering in `responselabels`.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`predict`](@ref CPPLS.predict), 
[`onehot`](@ref CPPLS.onehot),
[`coefall`](@ref CPPLS.coefall), 

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> classes = categorical(classes);

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, analysis_mode=:discriminant);

julia> mf = fit(m, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> predictclasses(mf, Xnew) == ["major", "minor"]
true
```
"""
function predictclasses(
    mf::CPPLSFit,
    X::AbstractMatrix{<:Real},
    ncomponents::Integer=size(coefall(mf), 3)
)
    predictclasses(mf, predict(mf, X, ncomponents))
end

"""
    predictclasses(
        mf::CPPLSFit,
        predictions::AbstractArray{<:Real, 3}
    ) -> AbstractVector

Convert a 3D prediction tensor (as returned by `predict`) into class labels using the 
stored `responselabels` ordering.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`analysis_mode`](@ref CPPLS.analysis_mode)
[`predictclasses`](@ref CPPLS.predictclasses)
[`predict`](@ref CPPLS.predict), 
[`onehot`](@ref CPPLS.onehot), 
[`onehot`](@ref CPPLS.onehot), 
[`sampleclasses`](@ref CPPLS.sampleclasses)
[`responselabels`](@ref CPPLS.responselabels)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> classes = categorical(classes);

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, analysis_mode=:discriminant);

julia> mf = fit(m, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> raw = predict(mf, Xnew);

julia> predictclasses(mf, raw) == ["major", "minor"]
true
```
"""
function predictclasses(
    mf::CPPLSFit,
    predictions::AbstractArray{<:Real,3}
)
    analysis_mode(mf) ≡ :discriminant || throw(ArgumentError(
        "predictclasses is only defined for discriminant CPPLS models"))

    isempty(responselabels(mf)) && throw(ArgumentError(
        "responselabels must be provided to map predictions to class labels"))

    classcols = class_response_columns(mf)
    classlabels = responselabels(mf)[classcols]
    classlabels[sampleclasses(onehot(mf, predictions))]
end

"""
    project(mf::CPPLSFit, X::AbstractMatrix{<:Real}) -> AbstractMatrix

Compute latent component X scores by projecting new predictors `X` with a CPPLSFit
model. The predictors are centered and then multiplied by 
`projectionmatrix`(@ref CPPLS.projectionmatrix), returning an 
`(n_samples, ncomponents)` X score matrix.

See also
[`CPPLSFit`](@ref CPPLS.CPPLSFit),
[`projectionmatrix`](@ref CPPLS.projectionmatrix)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> classes = categorical(classes);

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, analysis_mode=:discriminant);

julia> mf = fit(m, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> xscores = project(mf, Xnew);

julia> size(xscores)
(2, 2)
```
"""
project(mf::CPPLSFit, X::AbstractMatrix{<:Real}) = 
    ((X .- xmean(mf)') ./ xstd(mf)') * projectionmatrix(mf)
