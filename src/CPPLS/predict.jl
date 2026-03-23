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

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant);

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
    n_samples_X = size(X, 1)
    n_targets_Y = length(ymean(m))

    ncomponents ≤ size(coefall(m), 3) || throw(DimensionMismatch(
        "ncomponents exceeds the number of components in the model"))

    X_norm = (X .- xmean(m)') ./ xstd(m)'
    
    Y_hat = similar(X, n_samples_X, n_targets_Y, ncomponents)

    for i = 1:ncomponents
        @views Y_hat[:, :, i] .= (X_norm * coefall(m)[:, :, i]) .* ystd(m)' .+ ymean(m)'
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
This calls `predict`, sums predictions across components, corrects for repeated mean
addition, and assigns each sample to the highest-scoring class.

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

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant);

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
Predictions are summed across components and corrected for repeated mean addition before
selecting the highest-scoring class for each sample.

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

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant);

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
    ncomponents = size(predictions, 3)
    n_classes = size(predictions, 2)

    Y_pred_sum = sum(predictions, dims=3)[:, :, 1]
    Y_pred_final = Y_pred_sum .- (ncomponents - 1) .* ymean(mf)'

    predicted_class_indices = argmax.(eachrow(Y_pred_final))

    onehot(predicted_class_indices, n_classes)
end

"""
    sampleclasses(
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

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant);

julia> mf = fit(m, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> sampleclasses(mf, Xnew) == ["major", "minor"]
true
```
"""
function sampleclasses(
    mf::CPPLSFit,
    X::AbstractMatrix{<:Real},
    ncomponents::Integer=size(coefall(mf), 3)
)
    sampleclasses(mf, predict(mf, X, ncomponents))
end

"""
    sampleclasses(
        mf::CPPLSFit,
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
[`onehot`](@ref CPPLS.onehot), 
[`sampleclasses`](@ref CPPLS.sampleclasses)
[`responselabels`](@ref CPPLS.responselabels)

# Examples
```jldoctest
julia> using CPPLS; using JLD2; using Random;

julia> X, classes = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"), "X", "classes");

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant);

julia> mf = fit(m, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> raw = predict(mf, Xnew);

julia> sampleclasses(mf, raw) == ["major", "minor"]
true
```
"""
function sampleclasses(
    mf::CPPLSFit,
    predictions::AbstractArray{<:Real,3}
)
    mode(mf) ≡ :discriminant || throw(ArgumentError(
        "sampleclasses is only defined for discriminant CPPLS models"))

    isempty(responselabels(mf)) && throw(ArgumentError(
        "responselabels must be provided to map predictions to class labels"))

    n_classes = size(predictions, 2)
    length(responselabels(mf)) == n_classes || throw(DimensionMismatch(
        "responselabels must have length $n_classes, " * 
        "got $(length(responselabels(mf)))"))

    class_indices = sampleclasses(onehot(mf, predictions))
    responselabels(mf)[class_indices]
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

julia> m = CPPLSModel(ncomponents=2, gamma=0.5, mode=:discriminant);

julia> mf = fit(m, X, classes);

julia> Xnew = randn(MersenneTwister(1234), 2, size(X, 2));

julia> xscores = project(mf, Xnew);

julia> size(xscores)
(2, 2)
```
"""
project(mf::CPPLSFit, X::AbstractMatrix{<:Real}) = 
    ((X .- xmean(mf)') ./ xstd(mf)') * projectionmatrix(mf)
