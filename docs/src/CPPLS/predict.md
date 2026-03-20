# Projection and Prediction

After a CPPLS model has been fitted with [`fit`](@ref), it can be applied in two main
ways. [`project`](@ref) maps new samples into the latent score space defined by the
model, whereas [`predict`](@ref) generates predicted responses from new predictor
values.

For discriminant models, CPPLS also provides helpers for turning raw prediction
arrays into class assignments. [`onehot`](@ref) and
[`sampleclasses`](@ref CPPLS.sampleclasses(::CPPLS.CPPLSFit, ::AbstractArray{<:Real,3})) perform those conversions when you already
have the output of [`predict`](@ref). For convenience, [`onehot`](@ref) returns 
one-hot encoded class predictions directly, and [`sampleclasses`](@ref CPPLS.sampleclasses(::CPPLS.CPPLSFit, ::AbstractMatrix{<:Real}, ::Integer)) returns 
predicted class labels.

## Example

The example below reuses the synthetic discriminant-analysis dataset introduced on the
[Fit](@ref) page. We hold out one sample from each class, fit a CPPLS-DA model on the
remaining observations, and then use the fitted model to:

1. project the held-out samples into the latent space, and
2. predict their class membership.

We start by loading the synthetic data and splitting them into a training set and a
hold-out set.

The packages loaded below serve different purposes: `CPPLS` provides the modeling,
projection, and prediction functions, while `JLD2` reads the example dataset from disk.
In a normal Julia environment, both packages must be installed before running the
example; the Julia Pkg documentation explains how to install registered packages in the
[Getting Started](https://pkgdocs.julialang.org/v1/getting-started/#Basic-Usage)
section.

```@example project
using CPPLS
using JLD2
using CairoMakie

# Get custom colors
orange, blue = Makie.wong_colors()[2], Makie.wong_colors()[1]

samplelabels, X, classes, Y_aux = load(
    CPPLS.dataset("synthetic_cppls_da_dataset.jld2"),
    "sample_labels",
    "X",
    "classes",
    "Y_aux"
)

holdout_idx = [findlast(==("minor"), classes), findlast(==("major"), classes)]
train_idx = setdiff(collect(axes(X, 1)), holdout_idx)

X_train = X[train_idx, :]
classes_train = classes[train_idx]
Y_aux_train = Y_aux[train_idx, :]
labels_train = samplelabels[train_idx]

X_holdout = X[holdout_idx, :]
classes_holdout = classes[holdout_idx]
labels_holdout = samplelabels[holdout_idx]
plot_classes_holdout = ["projected $class" for class in classes_holdout]
nothing # hide
```

We next fit a discriminant model with two latent components and allow `gamma` to be 
selected during fitting.

```@example project
spec = CPPLSSpec(
    ncomponents=2,
    gamma=intervalize(0:0.25:1),
    mode=:discriminant
)

model = fit(
    spec,
    X_train,
    classes_train;
    obs_weights=invfreqweights(classes_train),
    Y_aux=Y_aux_train,
    samplelabels=labels_train
)
nothing # hide
```

We can now apply the fitted model to the held-out samples. We first use [`project`](@ref) 
to obtain latent scores and then plot the held-out samples together with the training 
samples in the fitted score space. The returned `heldout_scores` matrix has one row per 
held-out sample and one column per latent component.

```@example project
heldout_scores = project(model, X_holdout)

projected_plt = scoreplot(
    vcat(labels_train, labels_holdout),
    vcat(classes_train, plot_classes_holdout),
    vcat(xscores(model), heldout_scores);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores",
    group_order=["minor", "projected minor", "major", "projected major"],
    group_marker=Dict(
        "minor" => (; color=orange), 
        "projected minor" => (; color=orange, marker=:x, markersize=16, strokecolor=:black, strokewidth=1),
        "major" => (; color=blue),
        "projected major" => (; color=blue, marker=:x, markersize=16, strokecolor=:black, strokewidth=1)
    ),
    default_marker=(; markersize=14)
)
save("projected.svg", projected_plt)
nothing # hide
```

![](projected.svg)

The two projected samples fall near the clusters of the classes from which they were
held out. That visual impression suggests that the model should classify them as
`minor` and `major`, respectively, but prediction lets us check that conclusion more
directly.

The true class labels of the held-out samples are:

```@example project
classes_holdout
```

Let us now see what the model predicts:

```@example project
heldout_predictions = predict(model, X_holdout)
sampleclasses(model, heldout_predictions)
```

As we can see, the predicted labels match the classes from which the samples were
drawn. In this example, `heldout_predictions` is a three-dimensional array whose
third dimension indexes the number of components used in the prediction.

Instead of calling [`predict`](@ref) and [`sampleclasses`](@ref)
successively, we could have used the convenience wrapper
[`sampleclasses`](@ref). More generally, [`onehot`](@ref) and
[`sampleclasses`](@ref) collapse the full prediction tensor into class
assignments, which is often more convenient in discriminant-analysis workflows.

```@example project
sampleclasses(model, X_holdout)
```

## API

```@docs
CPPLS.onehot
CPPLS.onehot(::CPPLS.AbstractCPPLSFit, ::AbstractArray{<:Real, 3})
CPPLS.onehot(::CPPLS.AbstractCPPLSFit, ::AbstractMatrix{<:Real}, ::Integer)
CPPLS.predict
CPPLS.project
CPPLS.sampleclasses
CPPLS.sampleclasses(::CPPLS.CPPLSFit, ::AbstractArray{<:Real,3})
CPPLS.sampleclasses(::CPPLS.CPPLSFit, ::AbstractMatrix{<:Real}, ::Integer)
```
