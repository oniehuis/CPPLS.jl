# Fit Examples

This page collects longer fitting workflows that complement the reference material in
[Fit](fit.md). The examples below show how the same `fit` interface can be used for
discriminant analysis and regression, and how observation weights, additional responses,
and gamma selection affect the resulting models.

## Discriminant-Analysis Workflow

In this example, we use a synthetic dataset that ships with the package. The dataset
contains 100 samples: 10 belong to the minority class `minor` and 90 to the majority
class `major`. Accordingly, the predictor matrix has 100 rows and 14 columns,
representing 14 measured traits. The dataset is useful for illustrating three modeling
choices that interact in practice: inverse-frequency observation weighting, the
additional response, and the selection of gamma once those ingredients are in place.

The dataset was constructed so that a plain PCA score plot is dominated by nuisance
structure, whereas CPPLS-DA recovers the class contrast more clearly. That makes it a
useful benchmark for showing how observation weighting and additional responses change the
fitted score space and how gamma should then be chosen for that full model.

In addition to `CPPLS`, the example uses [JLD2](https://github.com/JuliaIO/JLD2.jl) to load
the dataset from disk,
[MultivariateStats](https://github.com/JuliaStats/MultivariateStats.jl) to compute the PCA
baseline, [Colors](https://github.com/JuliaGraphics/Colors.jl) to convert the additional
variable into grayscale values in the later additional-response plots,
[Statistics](https://github.com/JuliaStats/Statistics.jl) to orient the latent variables
consistently across plots, and [CairoMakie](https://docs.makie.org) to render static
figures. The Julia Pkg documentation explains how to install registered packages in the
[Getting Started](https://pkgdocs.julialang.org/v1/getting-started/#Basic-Usage)
section.

```@example fit_da
using CPPLS
using JLD2
using MultivariateStats
using Statistics
using CairoMakie
using Colors

# Get custom colors
orange, blue = Makie.wong_colors()[2], Makie.wong_colors()[1]

# Load example data from file
data = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"))


# Extract data of interest
sample_labels = data["sample_labels"]
X             = data["X"]
classes       = data["classes"]
Yadd_raw      = data["Y_add"]

# Keep the additional response on its original scale. In CPPLS, additional variables enter
# through predictor-response correlations, so ordinary centering or scaling of Yadd is
# not the main mechanism for controlling its influence.
Yadd = Yadd_raw

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
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="PCA scores",
    xlabel="Principal Component 1",
    ylabel="Principal Component 2",
    group_order=["minor", "major"],
    group_marker=Dict("minor" => (; color=orange), "major" => (; color=blue)),
    default_marker=(; markersize=14)
)
save("pca.svg", pca_plt)
nothing # hide
```

![](pca.svg)

In this synthetic dataset, the first two principal components are not optimized for
class discrimination, so a substantial part of the visible structure reflects nuisance
variation rather than the class contrast of interest.

### Observation Weights and Additional Responses

To illustrate the impact of class imbalance and additional response information on latent
variable extraction, we show how adding each of these factors changes the X scores of the
fitted model. In this demonstration, we use a fixed gamma value of 0.5 so that only the
parameter of interest varies. In a real-world scenario, you would typically decide whether
to include observation weights and additional responses before the main analysis, and then
optimize gamma in a model that already includes these parameters.

We start with a plain model in which neither observation weights nor additional response
information is considered.

```@example fit_da
m = CPPLSModel(
    ncomponents=2,
    gamma=0.5,
    analysis_mode=:discriminant,
    center_X=true,
    scale_X=true
)

m_plain = fit(
    m,
    X,
    classes;
    samplelabels=sample_labels
)

cppls_plain_plt = scoreplot(
    sample_labels,
    classes,
    xscores(m_plain, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores from an unweighted model",
    group_order=["minor", "major"],
    group_marker=Dict("minor" => (; color=orange), "major" => (; color=blue)),
    default_marker=(; markersize=14)
)
save("cppls_plain.svg", cppls_plain_plt)
nothing # hide
```

![](cppls_plain.svg)

Even in this comparatively simple model, the samples belonging to the two classes are
noticeably better separated from each other than in the PCA score plot.

Next, we add inverse-frequency observation weights by specifying the `obs_weights` keyword
argument in the [`fit`](@ref) function. To calculate the inverse-frequency weights for
samples in each class, we use the function [`invfreqweights`](@ref). This adjustment gives
the two classes equal total influence on the fitted model and thus removes the class size
imbalance.

```@example fit_da
class_weights = invfreqweights(classes)

m_weighted = fit(
    m,
    X,
    classes;
    obs_weights=class_weights,   # <- added parameter
    samplelabels=sample_labels
)

cppls_weighted_plt = scoreplot(
    sample_labels,
    classes,
    xscores(m_weighted, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores from an inverse-frequency-weighted model",
    group_order=["minor", "major"],
    group_marker=Dict("minor" => (; color=orange), "major" => (; color=blue)),
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

At this point, the class separation already looks convincing. In this dataset, however, it
is driven not only by class-related variation but also by variation associated with
additional responses. For example, suppose the samples in the two groups were analyzed using
two different instruments, and the choice of instrument is correlated with group
membership. If most majority-class samples were measured with instrument A and most
minority-class samples with instrument B, then instrument effects could confound the class
separation.

If such additional information is available, we can provide it to the model so that it can
account for this confounding structure. This is done using the optional keyword argument
`Yadd` in the [`fit`](@ref) function. Supplying additional response information changes
how the latent variables are estimated: instead of forcing all supervised structure into
the class contrast, the model can also explicitly model variation associated with the
additional responses. This helps ensure that the separation between the classes is not
driven by confounding factors, but rather by the traits of true interest.

```@example fit_da
m_weighted_yadd = fit(
    m,
    X,
    classes;
    obs_weights=class_weights,
    Yadd=Yadd,                   # <- added parameter
    samplelabels=sample_labels
)

cppls_weighted_yadd_plt = scoreplot(
    sample_labels,
    classes,
    xscores(m_weighted_yadd, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores from an inverse-frequency-weighted model with additional responses",
    group_order=["minor", "major"],
    group_marker=Dict("minor" => (; color=orange), "major" => (; color=blue)),
    default_marker=(; markersize=14)
)
save("cppls_weighted_yadd.svg", cppls_weighted_yadd_plt)
nothing # hide
```

![](cppls_weighted_yadd.svg)

The visible class separation may not have increased, but it is now more likely to reflect
information that is genuinely related to class membership rather than variation carried
by a correlated covariate. To make that difference easier to see, we plot the last two
score sets again, now shading each point by its additional response value.

```@example fit_da
# Convert additional-response matrix to a vector
add = vec(Yadd[:, 1])

# Generate colors for each Yadd value within range Gray(0.1) and Gray(0.8)
add_min, add_max = extrema(add)
point_colors = Gray.(0.1 .+ 0.8 .* ((add .- add_min) ./ (add_max - add_min)))

cppls_weighted_shaded_plt = scoreplot(
    sample_labels,
    fill("samples", length(sample_labels)),
    xscores(m_weighted, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores from an inverse-frequency-weighted model " *
          "shaded by additional values",
    show_legend=false,
    default_scatter=(; color=point_colors),
    default_marker=(; markersize=14)
)
save("cppls_weighted_shaded.svg", cppls_weighted_shaded_plt)

cppls_weighted_yadd_shaded_plt = scoreplot(
    sample_labels,
    fill("samples", length(sample_labels)),
    xscores(m_weighted_yadd, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores from an inverse-frequency-weighted model with additional responses " *
          "shaded by additional values",
    show_legend=false,
    default_scatter=(; color=point_colors),
    default_marker=(; markersize=14)
)
save("cppls_weighted_yadd_shaded.svg", cppls_weighted_yadd_shaded_plt)
nothing # hide
```

![](cppls_weighted_shaded.svg)

![](cppls_weighted_yadd_shaded.svg)

In the first shaded score plot, fitted without additional response information, the
grayscale values are not randomly distributed across the score space. Instead, they are
arranged roughly along the first latent dimension. This indicates that additional signal
correlated with class membership has leaked into the apparent class separation.

With the additional response information included, as shown in the second plot, the
additional structure is much less pronounced in the fitted scores, as indicated by the
grayscale values being more evenly distributed across the plot.

### Choosing Gamma for the Weighted + Additional-Response Model

With observation weights and `Yadd` in place, we can now choose `gamma` for the model we
actually want to interpret.

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
`gammas(model, lv)` and `rhos(model, lv)` contain one point per
candidate supplied to `gamma`. For a grid, that gives a sampled view of the objective
curve across gamma. For `intervalize(...)`, it gives one selected optimum per
interval, which is useful for comparing intervals but is not itself a dense landscape
trace. If the goal is to visualize the shape of the gamma objective, prefer a grid. If
the goal is robust optimization over subranges, prefer intervals.

We first inspect the objective landscape on a dense grid and
then use interval-based optimization to obtain a practical two-component fit.

```@example fit_da
weighted_yadd_grid_m = CPPLSModel(
    ncomponents=1,
    gamma=0:0.01:1,
    analysis_mode=:discriminant,
    center_X=true,
    scale_X=true
)

weighted_yadd_grid_model = fit(
    weighted_yadd_grid_m,
    X,
    classes;
    obs_weights=class_weights,
    Yadd=Yadd
)

weighted_yadd_grid_gammas = gammas(weighted_yadd_grid_model, 1)
weighted_yadd_grid_rhos = rhos(weighted_yadd_grid_model, 1)
selected_weighted_yadd_grid_gamma = gamma(weighted_yadd_grid_model)[1]

println("Best gamma with obs_weights and Yadd: ", selected_weighted_yadd_grid_gamma)
i = findfirst(==(selected_weighted_yadd_grid_gamma), weighted_yadd_grid_gammas)
println("Associated rho^2: ", weighted_yadd_grid_rhos[i])

weighted_yadd_gamma_fig = Figure(size=(900, 450))
weighted_yadd_gamma_ax = Axis(
    weighted_yadd_gamma_fig[1, 1],
    xlabel="Gamma",
    ylabel="Leading Squared Canonical Correlation",
    title="Weighted + additional-response objective landscape over gamma"
)

lines!(weighted_yadd_gamma_ax, weighted_yadd_grid_gammas, weighted_yadd_grid_rhos;
    color=:grey70, linewidth=3)
vlines!(weighted_yadd_gamma_ax, [selected_weighted_yadd_grid_gamma];
    color=:black, linestyle=:dash)

save("gamma_weighted_yadd_grid.svg", weighted_yadd_gamma_fig)
nothing # hide
```

![](gamma_weighted_yadd_grid.svg)

To turn that landscape into an actual optimization step, we next search over adjacent
intervals. The grey curve below is the dense landscape just plotted, the red points are
the best values retained from each interval, and the dashed line marks the overall
winner.

```@example fit_da
weighted_yadd_interval_m = CPPLSModel(
    ncomponents=1,
    gamma=intervalize(0:0.25:1),
    analysis_mode=:discriminant,
    center_X=true,
    scale_X=true
)

weighted_yadd_interval_mf = fit(
    weighted_yadd_interval_m,
    X,
    classes;
    obs_weights=class_weights,
    Yadd=Yadd
)

weighted_yadd_interval_gammas = gammas(weighted_yadd_interval_mf, 1)
weighted_yadd_interval_rhos = rhos(weighted_yadd_interval_mf, 1)
selected_weighted_yadd_gamma = gamma(weighted_yadd_interval_mf)[1]

println("Interval-optimized gamma with obs_weights and Yadd: ", selected_weighted_yadd_gamma)
i = findfirst(==(selected_weighted_yadd_gamma), weighted_yadd_interval_gammas)
println("Associated rho^2: ", weighted_yadd_interval_rhos[i])

weighted_yadd_interval_fig = Figure(size=(900, 450))
weighted_yadd_interval_ax = Axis(
    weighted_yadd_interval_fig[1, 1],
    xlabel="Gamma",
    ylabel="Leading Squared Canonical Correlation",
    title="Interval-wise gamma optimization with weights and additional responses"
)

lines!(weighted_yadd_interval_ax, weighted_yadd_grid_gammas, weighted_yadd_grid_rhos;
    color=:grey70, linewidth=3)
scatter!(weighted_yadd_interval_ax, weighted_yadd_interval_gammas, weighted_yadd_interval_rhos;
    color=:firebrick, markersize=10)
vlines!(weighted_yadd_interval_ax, [selected_weighted_yadd_gamma];
    color=:firebrick, linestyle=:dash)

save("gamma_weighted_yadd_intervals.svg", weighted_yadd_interval_fig)
nothing # hide
```

![](gamma_weighted_yadd_intervals.svg)

In this example, the interval-based optimization returns almost the same `gamma` value as
the dense grid search, so the difference is negligible in practice. That agreement should
not be taken for granted, however. If the objective landscape is rugged or contains
multiple local optima, an interval-based search can miss a narrow global maximum or favor
different local solutions depending on how the interval partition is chosen. It is
therefore advisable to first inspect the landscape on a grid and then decide whether a
coarse or dense interval partition is appropriate for downstream analyses such as cross-
validation and permutation testing.

Finally, we fit the two-component discriminant model with interval-optimized `gamma`
while keeping both inverse-frequency weights and `Yadd`. This is the most favorable DA
setup examined in this example because weighting, additional supervision, and gamma
selection are all aligned with the same objective.

```@example fit_da
weighted_yadd_best_m = CPPLSModel(
    ncomponents=2,
    gamma=intervalize(0:0.25:1),
    analysis_mode=:discriminant,
    center_X=true,
    scale_X=true
)

weighted_yadd_best_mf = fit(
    weighted_yadd_best_m,
    X,
    classes;
    obs_weights=class_weights,
    Yadd=Yadd,
    samplelabels=sample_labels
)

selected_weighted_yadd_rhos = [
    rhos(weighted_yadd_best_mf, lv)[findfirst(
        ==(gamma(weighted_yadd_best_mf)[lv]),
        gammas(weighted_yadd_best_mf, lv),
    )]
    for lv in 1:2
]

println("Selected gammas for the two latent variables: ",
    round.(gamma(weighted_yadd_best_mf), digits=3))
println("Associated rho^2: ", round.(selected_weighted_yadd_rhos, digits=6))

cppls_weighted_yadd_best_plt = scoreplot(
    sample_labels,
    classes,
    xscores(weighted_yadd_best_mf, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores with weights, Yadd, and optimized gamma",
    group_order=["minor", "major"],
    group_marker=Dict("minor" => (; color=orange), "major" => (; color=blue)),
    default_marker=(; markersize=14)
)
save("cppls_weighted_yadd_best.svg", cppls_weighted_yadd_best_plt)
nothing # hide
```

![](cppls_weighted_yadd_best.svg)

Overall, the example shows how observation weights can rebalance class influence, how
additional responses can help separate additional structure from the signal that is
genuinely relevant to group membership, and how `gamma` should be chosen only after
those ingredients are in place.

## Regression Example: Predicting a Continuous Response

The previous sections focused on discriminant analysis, but `CPPLS` is equally suited for
regression tasks. To illustrate this, we now demonstrate a regression workflow using the
same synthetic dataset. Here, we regress $X$ against the continuous response $Y_{\mathrm{add}}$,
while using the original class labels $Y$ as an additional response. This setup mimics
scenarios where the main prediction target is continuous, but additional categorical or
structured information is available to guide the supervised projection.

This example highlights several key points:

- **Regression in CPPLS** uses the same unified interface as DA, with `analysis_mode=:regression`.
- **Additional responses** can help ensure that the extracted latent variables reflect the
  main regression signal, not confounding structure from correlated categorical variables.
- **Observation weighting** is generally less critical in regression with balanced
  synthetic data, but the option remains available for real-world scenarios with
  heteroscedasticity or sample importance.

This approach demonstrates the flexibility of CPPLS for hybrid modeling, where both
continuous and categorical responses can be leveraged.

```@example fit_regression
using CPPLS
using JLD2
using CairoMakie
using Colors
using GLM
using DataFrames

# Load example data
data = load(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"))
sample_labels = data["sample_labels"]
X        = data["X"]
Y_main   = data["Y_add"]  # main regression target

classes  = data["classes"]
# Use only the matrix part of onehot encoding
Yadd_mat, _ = onehot(classes)  # additional response as one-hot matrix


# Set up regression model: predict Y_main from X, use Yadd_mat as additional
m = CPPLSModel(
    ncomponents=2,
    gamma=intervalize(0:0.25:1),
    analysis_mode=:regression,
    center_X=true,
    scale_X=true
)

mf = fit(m, X, Y_main;
    Yadd=Yadd_mat,  # Use one-hot encoded class labels as additional response
    samplelabels=sample_labels
)

# For regression, visualize predicted Y vs. true Y, with regression line
# Ensure both are 1D vectors for the first response column
Y_pred = predict(mf, X, 1)[:, 1, 1]  # predicted values (vector)
Y_true = Y_main[:, 1]  # true values (vector, first response column)

# Fit regression line using GLM
df = DataFrame(y_true=Y_true, y_pred=Y_pred)
lm = fit(LinearModel, @formula(y_pred ~ y_true), df)
Y_fit = predict(lm)

fig = Figure(size=(900, 450))
ax = Axis(fig[1, 1],
    xlabel="True Yadd (first response)",
    ylabel="Predicted Yadd (first response, first component)",
    title="Regression: Predicted vs. True Yadd (first response)"
)
scatter!(ax, Y_true, Y_pred, color=:dodgerblue, markersize=10, label="Samples")
lines!(ax, Y_true, Y_fit, color=:firebrick, linewidth=3, label="Regression line")
axislegend(ax)
save("regression_scoreplot.svg", fig)
nothing # hide
#+# Additional plot: LV1 vs. Y_true
LV1 = xscores(mf, 1)[:, 1]  # first latent variable (component 1)

fig_lv1 = Figure(size=(900, 450))
ax_lv1 = Axis(fig_lv1[1, 1],
    xlabel="LV1 (first component)",
    ylabel="True Yadd (first response)",
    title="LV1 vs. True Yadd (first response)"
)
scatter!(ax_lv1, LV1, Y_true, color=:seagreen, markersize=10, label="Samples")
axislegend(ax_lv1)
save("lv1_vs_ytrue.svg", fig_lv1)
nothing # hide
```

![](regression_scoreplot.svg)

![](lv1_vs_ytrue.svg)

In this plot, each point represents a sample, with its position determined by the first
latent variable (t₁) and the predicted value of $Y_{\mathrm{add}}$. This visualization helps
assess how well the main direction of variance in $X$ (as captured by t₁) aligns with the
regression target. The use of class labels as an additional response ensures that the
extracted components are not unduly influenced by class-related structure, but instead
focus on the continuous outcome of interest.

This regression example demonstrates the versatility of CPPLS for both regression and
classification, and shows how additional responses can be used to disentangle complex
sources of variation in supervised modeling.
