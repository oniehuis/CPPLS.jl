# Projection and Prediction

After a CPPLS model has been fitted with [`fit`](@ref), it can be applied in two main
ways. [`project`](@ref) maps new samples into the latent score space defined by the
model, whereas [`predict`](@ref) generates predicted responses from new predictor
values.

For discriminant models, CPPLS also provides helpers for turning raw prediction
arrays into class assignments. [`predictions_to_onehot`](@ref) and
[`predictions_to_sampleclasses`](@ref) perform those conversions when you already
have the output of [`predict`](@ref). For convenience, [`predictonehot`](@ref) returns 
one-hot encoded class predictions directly, and [`predictsampleclasses`](@ref) returns 
predicted class labels.

## Example

The example below reuses the synthetic discriminant-analysis dataset introduced on the
[Fit](@ref) page. We hold out one sample from each class, fit a CPPLS-DA model on the
remaining observations, and then use the fitted model to:

1. project the held-out samples into the latent space, and
2. predict their class membership.

We start by loading the synthetic data and splitting them into a training set and a
hold-out set.

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

holdout_idx = [findlast(==("minor"), classes), findlast(==("major"), classes)]
train_idx = setdiff(collect(axes(X, 1)), holdout_idx)

X_train = X[train_idx, :]
classes_train = classes[train_idx]
Y_aux_train = Y_aux[train_idx, :]
labels_train = sample_labels[train_idx]

X_holdout = X[holdout_idx, :]
classes_holdout = classes[holdout_idx]
labels_holdout = sample_labels[holdout_idx]
plot_classes_holdout = ["projected $class" for class in classes_holdout]
nothing # hide
```

We next fit a discriminant model with two latent components and allow `gamma` to be 
selected during fitting.

```@example project
spec = CPPLSSpec(
    n_components=2,
    gamma=0:0.1:1,
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

We can now apply the fitted model to the held-out samples. We first use [`project`](@ref) 
to obtain latent scores and then plot the held-out samples together with the training 
samples in the fitted score space. The returned `heldout_scores` matrix has one row per 
held-out sample and one column per latent component.

```@example project
heldout_scores = project(model, X_holdout)

fig_1 = scoreplot(
    vcat(labels_train, labels_holdout),
    vcat(classes_train, plot_classes_holdout),
    vcat(X_scores(model), heldout_scores);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores",
    default_marker=(; markersize=14)
)
save("fig_1.svg", fig_1)
nothing # hide
```

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
predictions_to_sampleclasses(model, heldout_predictions)
```

As we can see, the predicted labels match the classes from which the samples were
drawn. In this example, `heldout_predictions` is a three-dimensional array whose
third dimension indexes the number of components used in the prediction.

Instead of calling [`predict`](@ref) and [`predictions_to_sampleclasses`](@ref)
successively, we could have used the convenience wrapper
[`predictsampleclasses`](@ref). More generally, [`predictonehot`](@ref) and
[`predictsampleclasses`](@ref) collapse the full prediction tensor into class
assignments, which is often more convenient in discriminant-analysis workflows.

```@example project
predictsampleclasses(model, X_holdout)
```

## API

```@docs
CPPLS.predict
CPPLS.predictonehot
CPPLS.predictsampleclasses
CPPLS.predictions_to_onehot
CPPLS.predictions_to_sampleclasses
CPPLS.project
```
