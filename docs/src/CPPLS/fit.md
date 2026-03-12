# Fit

Model fitting in `CPPLS` is performed through `StatsAPI.fit` together with a
[`CPPLSSpec`](@ref). The same interface is used for both regression and discriminant
analysis. In both settings, the fit can optionally incorporate `obs_weights` and
`Y_aux`. Observation weights control how strongly individual samples contribute to the
model, whereas `Y_aux` supplies auxiliary response information that can shape the
supervised projection without changing the primary prediction target. In discriminant
analysis, `obs_weights` are especially useful when classes are imbalanced. Together
with `gamma`, these arguments are among the main levers for tailoring a CPPLS model to
the structure of a particular dataset.

The following example illustrates the effect of observation weights and auxiliary responses
using a discriminant-analysis workflow.

## Example: Discriminant Analysis with Weights and `Y_aux`

In this example, we use a synthetic dataset that ships with the package. The dataset
contains 100 samples: 10 belong to the minority class `minor` and 90 to the majority
class `major`. Accordingly, the predictor matrix has 100 rows and 14 columns,
representing 14 measured traits. The classes differ across these traits, but the data
also include structured variation captured by `Y_aux`. Because that auxiliary signal
is partly correlated with class membership, it can influence the orientation of the
latent axes even though it is not itself the primary classification target.

The dataset was constructed so that a plain PCA score plot is dominated by nuisance
structure, whereas CPPLS-DA recovers the class contrast more clearly. Because the
dataset combines class imbalance with auxiliary structure that is partly aligned with
class membership, it is well suited for illustrating how weighting and auxiliary
responses alter the fitted score space. Observation weights reduce the influence of
the majority class, and `Y_aux` helps show how auxiliary structure is represented in
the latent space. Besides `CPPLS`, the example uses the packages `JLD2` to load the 
dataset from disk, `MultivariateStats` to compute the PCA, `CategoricalArrays` to handle 
categorical class and bin labels, and `CairoMakie` to render static figures.

```@example fit_da
using CPPLS
using CategoricalArrays
using JLD2
using MultivariateStats
using CairoMakie

sample_labels, X, classes, Y_aux = load(
    CPPLS.dataset("synthetic_cppls_da_dataset.jld2"),
    "sample_labels",
    "X",
    "classes",
    "Y_aux"
)

backend = :makie  # use Makie to render all score plots
figure_kwargs = (; size=(900, 600))  # scoreplot dimensions
nothing # hide
```

We start with a Principal Component Analysis (PCA) to inspect the separation of the two
classes based on variance alone.

```@example fit_da
M = fit(PCA, X'; maxoutdim=2)
scores_pca = predict(M, X')'
pca_fig = scoreplot(
    sample_labels,
    classes,
    scores_pca;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="PCA Scores",
    xlabel="PC1", 
    ylabel="PC2"
)
save("pca.svg", pca_fig)
nothing # hide
```

![](pca.svg)

The PCA score plot provides a baseline. In this synthetic dataset, the first two
principal components are not optimized for class discrimination, so a substantial part
of the visible structure reflects nuisance variation rather than the class contrast of
interest.

We next fit a CPPLS-DA model without observation weights or auxiliary responses. We use a
fixed `gamma=0.5` throughout this example rather than estimating `gamma`, so that the
main differences between the fits come from the inclusion or omission of observation
weights and `Y_aux`.

```@example fit_da
spec = CPPLSSpec(
    n_components=2,
    gamma=0.5,
    analysis_mode=:discriminant,
)

m_plain = fit(
    spec,
    X,
    classes;
    sample_labels=sample_labels,
)

plain_fig = scoreplot(
    m_plain;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA without weights or Y_aux"
)
save("cppls_da_plain.svg", plain_fig)
nothing # hide
```

![](cppls_da_plain.svg)

Fitting CPPLS-DA directly to the class labels yields a more class-oriented score space.
Because the classes are imbalanced, however, the majority class exerts more influence
on the latent variables than the minority class.

We now add inverse-frequency observation weights. This gives the two classes the same
total influence on the fitted model and reduces the bias introduced by unequal class
sizes.

```@example fit_da
m_weighted = fit(
    spec,
    X,
    classes;
    obs_weights=invfreqweights(classes),
    sample_labels=sample_labels
)

weighted_fig = scoreplot(
    m_weighted;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA with inverse-frequency weights"
)
save("cppls_da_weighted.svg", weighted_fig)
nothing # hide
```

![](cppls_da_weighted.svg)

Applying inverse-frequency weights makes the discriminant structure more symmetric. In
this example, the main effect is not necessarily a larger distance between the groups,
but rather a recentering and slight rotation of the first latent variable so that the
class contrast is less biased by class prevalence.

At this point, the class separation already looks convincing. In this dataset,
however, it is driven not only by class-related variation but also by structured
variation associated with `Y_aux`. To make that visible, we plot the same fitted
scores again, but now color the samples by three coarse bins derived from `Y_aux`.

```@example fit_da
aux = vec(Y_aux[:, 1])
aux_bins = categorical(ifelse.(aux .< -0.5, "low", ifelse.(aux .> 0.5, "high", "mid")))

weighted_aux_fig2 = scoreplot(
    sample_labels,
    aux_bins,
    X_scores(m_weighted)[:, 1:2];  # X scores of the first two LV of all samples
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="Weighted CPPLS-DA colored by auxiliary bins",
    group_order = ["low", "mid", "high"],
    group_marker = Dict(
        "low"  => (; color = :lightgray),
        "mid"  => (; color = :gray),
        "high" => (; color = :darkgray),
    )
)
save("cppls_da_weighted_colored.svg", weighted_aux_fig2)
nothing # hide
```

![](cppls_da_weighted_colored.svg)

The three colors are not randomly distributed across the score plot. Instead, they are
ordered roughly along the first latent dimension. This indicates that auxiliary signal
correlated with class membership has leaked into the apparent class separation. To
counteract that, we next add `Y_aux` in addition to the observation weights. This
changes the supervised optimization and allows the latent space to represent
structured auxiliary variation explicitly, rather than forcing all supervised signal
into the class contrast alone.

```@example fit_da
m_weighted_yaux = fit(
    spec,
    X,
    classes;
    obs_weights=invfreqweights(classes),
    Y_aux=Y_aux,
    sample_labels=sample_labels
)

weighted_yaux_fig = scoreplot(
    m_weighted_yaux;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA with weights and Y_aux"
)
save("cppls_da_weighted_aux.svg", weighted_yaux_fig)
nothing # hide
```

![](cppls_da_weighted_aux.svg)

To inspect the effect of `Y_aux`, we again color the score plot by the same coarse
auxiliary bins rather than by the class labels.

```@example fit_da
weighted_yaux_aux_fig3 = scoreplot(
    sample_labels,
    aux_bins,
    X_scores(m_weighted_yaux)[:, 1:2];  # X scores of the first two LV of all samples
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="Weighted CPPLS-DA + Y_aux colored by auxiliary bins",
    group_order = ["low", "mid", "high"],
    group_marker = Dict(
        "low"  => (; color = :lightgray),
        "mid"  => (; color = :gray),
        "high" => (; color = :darkgray),
    )
)
save("cppls_da_weighted_aux_colored.svg", weighted_yaux_aux_fig3)
nothing # hide
```

![](cppls_da_weighted_aux_colored.svg)

With `Y_aux` included, the auxiliary structure is much less pronounced in the fitted
scores, as indicated by the three `Y_aux` color groups being distributed more evenly
across the plot. The visible class separation may not increase much, but it is now
more likely to reflect information that is genuinely related to class membership
rather than variation carried by a correlated covariate.

```@docs
StatsAPI.fit
```
