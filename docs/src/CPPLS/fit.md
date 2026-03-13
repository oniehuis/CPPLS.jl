# Fit

Model fitting in `CPPLS` is performed through `StatsAPI.fit` together with a
[`CPPLSSpec`](@ref). The same interface is used for both regression and discriminant
analysis. In both settings, the fit can optionally incorporate `obs_weights` and
`Y_aux`. Observation weights control how strongly individual samples contribute to the
model, whereas auxiliary response information can shape the supervised projection without 
changing the primary prediction target. In discriminant analysis (DA), observation weights 
are especially useful when classes are imbalanced. Together with `gamma`, these arguments 
are among the main levers for tailoring a CPPLS model to the structure of a particular 
dataset.

The following example illustrates the effect of observation weights and auxiliary
responses in a discriminant-analysis workflow.

## Example: DA with Observation Weights and Auxiliary Responses

In this example, we use a synthetic dataset that ships with the package. The dataset
contains 100 samples: 10 belong to the minority class `minor` and 90 to the majority
class `major`. Accordingly, the predictor matrix has 100 rows and 14 columns,
representing 14 measured traits. The classes differ across these traits, but the data
also include structured variation captured by auxiliary responses. Because that auxiliary 
signal is partly correlated with class membership, it can influence the orientation of the
latent axes even though it is not itself the primary classification target.

The dataset was constructed so that a plain PCA score plot is dominated by nuisance
structure, whereas CPPLS-DA recovers the class contrast more clearly. Because the
dataset combines class imbalance with auxiliary structure that is partly aligned with
class membership, it is well suited for illustrating how observation weighting and
auxiliary responses alter the fitted score space. Observation weights reduce the
influence of the majority class, and auxiliary responses help reveal how auxiliary
structure is represented in the latent space.

In addition to `CPPLS`, the example uses `JLD2` to load the dataset from disk,
`MultivariateStats` to compute the PCA baseline, `Colors` to convert the auxiliary
variable into grayscale values, `Statistics` to orient the latent variables
consistently across plots, and `CairoMakie` to render static figures.

```@example fit_da
using CPPLS
using JLD2
using MultivariateStats
using Statistics
using CairoMakie
using Colors

sample_labels, X, classes, Y_aux = load(
    CPPLS.dataset("synthetic_cppls_da_dataset.jld2"),
    "sample_labels",
    "X",
    "classes",
    "Y_aux"
)

backend = :makie  # use Makie to render all score plots
figure_kwargs = (; size=(900, 600))  # score plot dimensions

function orient_scores(scores, classes; reference_class="major")
    oriented = copy(scores)
    reference_idx = classes .== reference_class
    for lv in axes(oriented, 2)
        if mean(oriented[reference_idx, lv]) < 0
            oriented[:, lv] .*= -1
        end
    end
    oriented
end
nothing # hide
```

We start with a Principal Component Analysis (PCA) to inspect the separation of the two
classes based on variance alone.

```@example fit_da
M = fit(PCA, X'; maxoutdim=2)
scores_pca = permutedims(predict(M, X'))

fig_1 = scoreplot(
    sample_labels,
    classes,
    scores_pca;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="PCA scores",
    xlabel="Principal Component 1", 
    ylabel="Principal Component 2",
    default_marker=(; markersize=14)
)
save("fig_1.svg", fig_1)
nothing # hide
```

![](fig_1.svg)

The PCA score plot provides a baseline. In this synthetic dataset, the first two principal 
components are not optimized for class discrimination, so a substantial part of the visible 
structure reflects nuisance variation rather than the class contrast of interest.

We next fit a CPPLS-DA model without observation weights or auxiliary responses. We use a
fixed `gamma=0.5` throughout this example rather than estimating `gamma`, so that the
main differences between the fits come from the inclusion or omission of observation 
weights and auxiliary responses. For visual comparability across plots, we orient each 
latent variable so that the mean score of class `major` is positive; this only fixes the 
sign indeterminacy of the latent variables and does not change the fit itself.

```@example fit_da
spec = CPPLSSpec(
    n_components=2,
    gamma=0.5,
    analysis_mode=:discriminant
)

m_plain = fit(
    spec,
    X,
    classes;
    sample_labels=sample_labels
)

scores_plain = orient_scores(X_scores(m_plain)[:, 1:2], classes)

fig_2 = scoreplot(
    sample_labels,
    classes,
    scores_plain;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA scores from an unweighted model without auxiliary responses",
    default_marker=(; markersize=14)
)
save("fig_2.svg", fig_2)
nothing # hide
```

![](fig_2.svg)

Fitting CPPLS-DA directly to the class labels yields a more class-oriented score
space. Because the classes are imbalanced, however, the majority class exerts more
influence on the latent variables than the minority class.

We now add inverse-frequency observation weights. This gives the two classes the same
total influence on the fitted model and reduces the bias introduced by unequal class
sizes.

```@example fit_da
m_weighted = fit(
    spec,
    X,
    classes;
    obs_weights=invfreqweights(classes),  # give both classes equal total weight
    sample_labels=sample_labels
)

scores_weighted = orient_scores(X_scores(m_weighted)[:, 1:2], classes)

fig_3 = scoreplot(
    sample_labels,
    classes,
    scores_weighted;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA scores from an inverse-frequency-weighted model",
    default_marker=(; markersize=14)
)
save("fig_3.svg", fig_3)
nothing # hide
```

![](fig_3.svg)

Applying inverse-frequency weights makes the discriminant structure more symmetric. In
this example, the main effect is not a larger distance between the groups, but rather
a recentering and slight rotation of the latent space so that the class contrast is
less biased by class prevalence.

At this point, the class separation already looks convincing. In this dataset,
however, it is driven not only by class-related variation but also by structured
variation associated with auxiliary responses. To address that, we include `Y_aux`
in addition to the observation weights. This changes how the latent variables are
estimated. Instead of forcing all supervised structure into the class contrast, the
model can also represent variation associated with auxiliary responses.

```@example fit_da
m_weighted_yaux = fit(
    spec,
    X,
    classes;
    obs_weights=invfreqweights(classes),  # give both classes equal total weight
    Y_aux=Y_aux,                          # model auxiliary responses explicitly
    sample_labels=sample_labels
)

scores_weighted_yaux = orient_scores(X_scores(m_weighted_yaux)[:, 1:2], classes)

fig_4 = scoreplot(
    sample_labels,
    classes,
    scores_weighted_yaux;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA scores from an inverse-frequency-weighted model with auxiliary responses",
    default_marker=(; markersize=14)
)
save("fig_4.svg", fig_4)
nothing # hide
```

![](fig_4.svg)

The visible class separation may not have increased much, but it is now more likely
to reflect information that is genuinely related to class membership rather than
variation carried by a correlated covariate. To make that difference easier to see,
we plot the last two score sets again, now shading each point by its auxiliary
response value.

```@example fit_da
aux = vec(Y_aux[:, 1])

aux_min, aux_max = extrema(aux)
point_colors = Gray.(0.1 .+ 0.8 .* ((aux .- aux_min) ./ (aux_max - aux_min)))

fig_5 = scoreplot(
    sample_labels,
    fill("samples", length(sample_labels)),
    scores_weighted;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA scores from an inverse-frequency-weighted model " *
          "shaded by auxiliary values",
    show_legend=false,
    default_scatter=(; color=point_colors),
    default_marker=(; markersize=14)
)
save("fig_5.svg", fig_5)

fig_6 = scoreplot(
    sample_labels,
    fill("samples", length(sample_labels)),
    scores_weighted_yaux;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA scores from an inverse-frequency-weighted model with auxiliary responses " *
          "shaded by auxiliary values",
    show_legend=false,
    default_scatter=(; color=point_colors),
    default_marker=(; markersize=14)
)
save("fig_6.svg", fig_6)
nothing # hide
```

![](fig_5.svg)

![](fig_6.svg)

In the first shaded score plot, fitted without auxiliary response information, the
grayscale values are not randomly distributed across the score space. Instead, they
are arranged roughly along the first latent dimension. This indicates that
auxiliary signal correlated with class membership has leaked into the apparent class
separation.

With the auxiliary response information included, as shown in the second plot, the
auxiliary structure is much less pronounced in the fitted scores, as indicated by
the grayscale values being more evenly distributed across the plot.

Overall, the example shows how observation weights can rebalance class influence and
how auxiliary responses can help separate auxiliary structure from the signal that is
genuinely relevant to group membership.

## API

The example above illustrates a discriminant-analysis workflow, but the same fitting
entry point is used more generally throughout `CPPLS`. The reference below documents
the `fit` interface itself.

```@docs
StatsAPI.fit
```
