# Projection and Prediction

After a CPPLS model has been fitted with [`fit`](@ref), it can be applied in two main
ways. [`project`](@ref) maps new samples into the latent score space defined by the
model, whereas [`predict`](@ref) generates predicted responses from new predictor
values.

For discriminant models, CPPLS also provides helpers for turning raw prediction
arrays into class assignments. [`predictions_to_onehot`](@ref) and
[`predictions_to_sampleclasses`](@ref) perform those conversions when you already
have the output of [`predict`](@ref). For convenience,
[`predictonehot`](@ref) returns one-hot encoded class predictions directly, and
[`predictsampleclasses`](@ref) returns predicted class labels.

## Example

The example below reuses the synthetic discriminant-analysis dataset introduced on the
fit page. We hold out one sample from each class, fit a CPPLS-DA model on the
remaining observations, and then use the fitted model to:

1. project the held-out samples into the latent space, and
2. predict their class membership.

This illustrates the distinction between projection and prediction. Projection returns
latent coordinates, whereas prediction returns responses or class assignments derived
from the fitted model.

```@example project
using CPPLS
using JLD2

sample_labels, X, classes, Y_aux = load(
    CPPLS.dataset("synthetic_cppls_da_dataset.jld2"),
    "sample_labels",
    "X",
    "classes",
    "Y_aux"
)

holdout_idx = [
    findfirst(==("minor"), classes),
    findfirst(==("major"), classes),
]
train_idx = setdiff(collect(axes(X, 1)), holdout_idx)

X_train = X[train_idx, :]
classes_train = classes[train_idx]
Y_aux_train = Y_aux[train_idx, :]
labels_train = sample_labels[train_idx]

X_holdout = X[holdout_idx, :]
classes_holdout = classes[holdout_idx]
labels_holdout = sample_labels[holdout_idx]
nothing # hide
```

We fit a weighted discriminant model with two latent components and allow `gamma` to
be selected during fitting.

```@example project
spec = CPPLSSpec(
    n_components=2,
    gamma=0:0.01:1,
    analysis_mode=:discriminant
)

model = fit(
    spec,
    X_train,
    classes_train;
    obs_weights=invfreqweights(classes_train),
    Y_aux=Y_aux_train,
    sample_labels=labels_train
)
nothing # hide
```

We can now apply the fitted model to the held-out samples. [`project`](@ref) returns
their latent coordinates, [`predict`](@ref) returns the full prediction tensor, and
the classification helpers convert those predictions into class assignments.

```@example project
heldout_scores = project(model, X_holdout)
heldout_predictions = predict(model, X_holdout)
heldout_onehot = predictonehot(model, X_holdout)
heldout_labels = predictsampleclasses(model, X_holdout)
nothing # hide
```

In this example, `heldout_scores` is a matrix with one row per held-out sample and one
column per latent component. `heldout_predictions` is a three-dimensional array whose
third dimension indexes the number of components used in the prediction. The helper
functions `predictonehot` and `predictsampleclasses` collapse that information into
class assignments that are often more convenient in discriminant-analysis workflows.

The true held-out labels are stored in `classes_holdout`, while the predicted labels
are stored in `heldout_labels`, so they can be compared directly if desired.

## API

```@docs
CPPLS.predict
CPPLS.predictonehot
CPPLS.predictsampleclasses
CPPLS.predictions_to_onehot
CPPLS.predictions_to_sampleclasses
CPPLS.project
```
