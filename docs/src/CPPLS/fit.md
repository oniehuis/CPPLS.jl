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

The `gamma` argument supports three distinct workflows. A fixed scalar such as `0.5`
uses the same value for every latent variable. A grid such as `0:0.01:1` evaluates a
set of fixed candidate values and keeps the one with the largest leading canonical
correlation for each latent variable. An interval specification such as
`[(0.1, 0.3), (0.7, 0.9)]` performs one bounded search inside each closed interval,
so both endpoints are included in the search, and then keeps the best interval-wise
optimum. The package ships with the helper function
[`intervalize`](@ref), which converts a grid into a list of adjacent intervals,
for example `intervalize(0:0.01:1)`.

That distinction matters for diagnostics. The stored values returned by
`gamma_search_gammas(model, lv)` and `gamma_search_rhos(model, lv)` contain one point per
candidate supplied to `gamma`. For a grid, that gives a sampled view of the objective
curve across gamma. For `intervalize(...)`, it gives one selected optimum per
interval, which is useful for comparing intervals but is not itself a dense landscape
trace. If the goal is to visualize the shape of the gamma objective, prefer a grid. If
the goal is robust optimization over subranges, prefer intervals.

If observation weights or auxiliary responses are part of the intended analysis, they
should be specified before `gamma` is selected, because both can change the supervised
objective and therefore the `gamma` values that are most useful.

## Example

In this example, we use a synthetic dataset that ships with the package. The dataset
contains 100 samples: 10 belong to the minority class `minor` and 90 to the majority
class `major`. Accordingly, the predictor matrix has 100 rows and 14 columns,
representing 14 measured traits. The dataset is useful for illustrating three modeling
choices that interact in practice: inverse-frequency observation weighting, the
auxiliary response block `Y_aux`, and the selection of `gamma` once those ingredients
are in place.

The dataset was constructed so that a plain PCA score plot is dominated by nuisance
structure, whereas CPPLS-DA recovers the class contrast more clearly. That makes it a
useful benchmark for showing how observation weighting and auxiliary responses change the
fitted score space and how `gamma` should then be chosen for that full model.

In addition to `CPPLS`, the example uses `JLD2` to load the dataset from disk,
`MultivariateStats` to compute the PCA baseline, `Colors` to convert the auxiliary
variable into grayscale values in the later auxiliary-response plots, `Statistics` to
orient the latent variables consistently across plots, and `CairoMakie` to render
static figures. The Julia Pkg documentation explains how to install registered packages in the
[Getting Started](https://pkgdocs.julialang.org/v1/getting-started/#Basic-Usage)
section.

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

For comparison, we first fit a Principal Component Analysis (PCA) baseline to inspect the
separation of the two classes based on variance alone.

```@example fit_da
M = fit(PCA, X'; maxoutdim=2)
scores_pca = permutedims(predict(M, X'))

pca_plt = scoreplot(
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
save("pca.svg", pca_plt)
nothing # hide
```

![](pca.svg)

In this synthetic dataset, the first two principal components are not optimized for
class discrimination, so a substantial part of the visible structure reflects nuisance
variation rather than the class contrast of interest.

### Observation Weights and Auxiliary Responses

In a practical DA analysis, class imbalance and auxiliary supervision should be handled
before `gamma` is tuned, because both affect the supervised structure that `gamma`
balances. We therefore begin with a simple working value, `gamma=0.5`, and first add the
other modeling ingredients that are already known to matter in this dataset. For visual
comparability across plots, we orient each latent variable so that the mean score of
class `major` is positive; this only fixes the sign indeterminacy of the latent variables
and does not change the fit itself.

We first add inverse-frequency observation weights. This gives the two classes the same
total influence on the fitted model and reduces the bias introduced by unequal class
sizes.

```@example fit_da
class_weights = invfreqweights(classes)

spec = CPPLSSpec(
    n_components=2,
    gamma=0.5,
    analysis_mode=:discriminant
)

m_weighted = fit(
    spec,
    X,
    classes;
    obs_weights=class_weights,
    sample_labels=sample_labels
)

scores_weighted = orient_scores(X_scores(m_weighted)[:, 1:2], classes)

cppls_weighted_plt = scoreplot(
    sample_labels,
    classes,
    scores_weighted;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA scores from an inverse-frequency-weighted model",
    default_marker=(; markersize=14)
)
save("cppls_weighted.svg", cppls_weighted_plt)
nothing # hide
```

![](cppls_weighted.svg)

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
    obs_weights=class_weights,
    Y_aux=Y_aux,                          # model auxiliary responses explicitly
    sample_labels=sample_labels
)

scores_weighted_yaux = orient_scores(X_scores(m_weighted_yaux)[:, 1:2], classes)

cppls_weighted_yaux_plt = scoreplot(
    sample_labels,
    classes,
    scores_weighted_yaux;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA scores from an inverse-frequency-weighted model with auxiliary responses",
    default_marker=(; markersize=14)
)
save("cppls_weighted_yaux.svg", cppls_weighted_yaux_plt)
nothing # hide
```

![](cppls_weighted_yaux.svg)

The visible class separation may not have increased, but it is now more likely to reflect 
information that is genuinely related to class membership rather than variation carried 
by a correlated covariate. To make that difference easier to see, we plot the last two 
score sets again, now shading each point by its auxiliary response value.

```@example fit_da
aux = vec(Y_aux[:, 1])

aux_min, aux_max = extrema(aux)
point_colors = Gray.(0.1 .+ 0.8 .* ((aux .- aux_min) ./ (aux_max - aux_min)))

cppls_weighted_shaded_plt = scoreplot(
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
save("cppls_weighted_shaded.svg", cppls_weighted_shaded_plt)

cppls_weighted_yaux_shaded_plt = scoreplot(
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
save("cppls_weighted_yaux_shaded.svg", cppls_weighted_yaux_shaded_plt)
nothing # hide
```

![](cppls_weighted_shaded.svg)

![](cppls_weighted_yaux_shaded.svg)

In the first shaded score plot, fitted without auxiliary response information, the
grayscale values are not randomly distributed across the score space. Instead, they
are arranged roughly along the first latent dimension. This indicates that
auxiliary signal correlated with class membership has leaked into the apparent class
separation.

With the auxiliary response information included, as shown in the second plot, the
auxiliary structure is much less pronounced in the fitted scores, as indicated by
the grayscale values being more evenly distributed across the plot.

### Choosing Gamma for the Weighted + Auxiliary Model

With observation weights and `Y_aux` in place, we can now choose `gamma` for the model we
actually want to interpret. We first inspect the objective landscape on a dense grid and
then use interval-based optimization to obtain a practical two-component fit.

```@example fit_da
weighted_yaux_grid_spec = CPPLSSpec(
    n_components=1,
    gamma=0:0.001:1,
    analysis_mode=:discriminant
)

weighted_yaux_grid_model = fit(
    weighted_yaux_grid_spec,
    X,
    classes;
    obs_weights=class_weights,
    Y_aux=Y_aux
)

weighted_yaux_grid_gammas = gamma_search_gammas(weighted_yaux_grid_model, 1)
weighted_yaux_grid_rhos = gamma_search_rhos(weighted_yaux_grid_model, 1)
selected_weighted_yaux_grid_gamma = gamma(weighted_yaux_grid_model)[1]

println("Best gamma with obs_weights and Y_aux: ", selected_weighted_yaux_grid_gamma)
i = findfirst(==(selected_weighted_yaux_grid_gamma), weighted_yaux_grid_gammas)
println("Associated rho^2: ", weighted_yaux_grid_rhos[i])

weighted_yaux_gamma_fig = Figure(size=(900, 450))
weighted_yaux_gamma_ax = Axis(
    weighted_yaux_gamma_fig[1, 1],
    xlabel="Gamma",
    ylabel="Leading Squared Canonical Correlation",
    title="Weighted + auxiliary objective landscape over gamma"
)

lines!(weighted_yaux_gamma_ax, weighted_yaux_grid_gammas, weighted_yaux_grid_rhos;
    color=:darkgreen, linewidth=3)
vlines!(weighted_yaux_gamma_ax, [selected_weighted_yaux_grid_gamma];
    color=:black, linestyle=:dash)

save("gamma_weighted_yaux_grid.svg", weighted_yaux_gamma_fig)
nothing # hide
```

![](gamma_weighted_yaux_grid.svg)

To turn that landscape into an actual optimization step, we next search over adjacent
intervals. The grey curve below is the dense landscape just plotted, the red points are
the best values retained from each interval, and the dashed line marks the overall
winner.

```@example fit_da
weighted_yaux_interval_spec = CPPLSSpec(
    n_components=1,
    gamma=intervalize(0:0.05:1),
    analysis_mode=:discriminant
)

weighted_yaux_interval_model = fit(
    weighted_yaux_interval_spec,
    X,
    classes;
    obs_weights=class_weights,
    Y_aux=Y_aux
)

weighted_yaux_interval_gammas = gamma_search_gammas(weighted_yaux_interval_model, 1)
weighted_yaux_interval_rhos = gamma_search_rhos(weighted_yaux_interval_model, 1)
selected_weighted_yaux_gamma = gamma(weighted_yaux_interval_model)[1]

println("Interval-optimized gamma with obs_weights and Y_aux: ", selected_weighted_yaux_gamma)
i = findfirst(==(selected_weighted_yaux_gamma), weighted_yaux_interval_gammas)
println("Associated rho^2: ", weighted_yaux_interval_rhos[i])

weighted_yaux_interval_fig = Figure(size=(900, 450))
weighted_yaux_interval_ax = Axis(
    weighted_yaux_interval_fig[1, 1],
    xlabel="Gamma",
    ylabel="Leading Squared Canonical Correlation",
    title="Interval-wise gamma optimization with weights and auxiliary responses"
)

lines!(weighted_yaux_interval_ax, weighted_yaux_grid_gammas, weighted_yaux_grid_rhos;
    color=:grey70, linewidth=3)
scatter!(weighted_yaux_interval_ax, weighted_yaux_interval_gammas, weighted_yaux_interval_rhos;
    color=:firebrick, markersize=10)
vlines!(weighted_yaux_interval_ax, [selected_weighted_yaux_gamma];
    color=:firebrick, linestyle=:dash)

save("gamma_weighted_yaux_intervals.svg", weighted_yaux_interval_fig)
nothing # hide
```

![](gamma_weighted_yaux_intervals.svg)

Finally, we fit the two-component discriminant model with interval-optimized `gamma`
while keeping both inverse-frequency weights and `Y_aux`. This is the most favorable DA
setup examined in this example because weighting, auxiliary supervision, and gamma
selection are all aligned with the same objective.

```@example fit_da
weighted_yaux_best_spec = CPPLSSpec(
    n_components=2,
    gamma=intervalize(0:0.05:1),
    analysis_mode=:discriminant
)

m_weighted_yaux_best = fit(
    weighted_yaux_best_spec,
    X,
    classes;
    obs_weights=class_weights,
    Y_aux=Y_aux,
    sample_labels=sample_labels
)

selected_weighted_yaux_rhos = [
    gamma_search_rhos(m_weighted_yaux_best, lv)[findfirst(
        ==(gamma(m_weighted_yaux_best)[lv]),
        gamma_search_gammas(m_weighted_yaux_best, lv),
    )]
    for lv in 1:2
]

println("Selected gammas for the two latent variables: ",
    round.(gamma(m_weighted_yaux_best), digits=3))
println("Associated rho^2: ", round.(selected_weighted_yaux_rhos, digits=6))

scores_weighted_yaux_best = orient_scores(X_scores(m_weighted_yaux_best)[:, 1:2], classes)

cppls_weighted_yaux_best_plt = scoreplot(
    sample_labels,
    classes,
    scores_weighted_yaux_best;
    backend=backend,
    figure_kwargs=figure_kwargs,
    title="CPPLS-DA scores with weights, Y_aux, and optimized gamma",
    default_marker=(; markersize=14)
)
save("cppls_weighted_yaux_best.svg", cppls_weighted_yaux_best_plt)
nothing # hide
```

![](cppls_weighted_yaux_best.svg)

Overall, the example shows how observation weights can rebalance class influence, how
auxiliary responses can help separate auxiliary structure from the signal that is
genuinely relevant to group membership, and how `gamma` should be chosen only after
those ingredients are in place.

## API

The example above illustrates a discriminant-analysis workflow, but the same fitting
entry point is used more generally throughout `CPPLS`. The reference below documents
the `fit` interface itself.

```@docs
StatsAPI.fit
```
