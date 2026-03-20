# Fit

Model fitting in `CPPLS` is performed using [`StatsAPI.fit`](@ref) together with a 
[`CPPLSSpec`](@ref). This unified interface supports both regression and discriminant 
analysis. In either case, you can optionally provide observation weights and auxiliary 
response information. Observation weights determine how much influence each sample has on 
the model, while auxiliary responses can guide the supervised projection without altering 
the primary prediction target. In discriminant analysis (DA), observation weights are 
particularly helpful for handling imbalanced classes. Along with the gamma parameter, 
which controls the balance between predictor and response variances and their correlation, 
observation weights and auxiliary responses are key options for tailoring a CPPLS model to 
your dataset. Other choices, such as the number of components, may also be important 
depending on your analysis.

If you plan to use observation weights or auxiliary responses, specify them before 
selecting the gamma parameter. Both can affect the supervised objective and, consequently, 
the gamma values that are most appropriate.

## Example

In this example, we use a synthetic dataset that ships with the package. The dataset
contains 100 samples: 10 belong to the minority class `minor` and 90 to the majority
class `major`. Accordingly, the predictor matrix has 100 rows and 14 columns,
representing 14 measured traits. The dataset is useful for illustrating three modeling
choices that interact in practice: inverse-frequency observation weighting, the
auxiliary response, and the selection of gamma once those ingredients are in place.

The dataset was constructed so that a plain PCA score plot is dominated by nuisance
structure, whereas CPPLS-DA recovers the class contrast more clearly. That makes it a
useful benchmark for showing how observation weighting and auxiliary responses change the
fitted score space and how gamma should then be chosen for that full model.

In addition to `CPPLS`, the example uses [JLD2](https://github.com/JuliaIO/JLD2.jl) to load 
the dataset from disk, 
[MultivariateStats](https://github.com/JuliaStats/MultivariateStats.jl) to compute the PCA 
baseline, [Colors](https://github.com/JuliaGraphics/Colors.jl) to convert the auxiliary
variable into grayscale values in the later auxiliary-response plots, 
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
Y_aux         = data["Y_aux"]

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

### Observation Weights and Auxiliary Responses

To illustrate the impact of class imbalance and auxiliary response information on latent 
variable extraction, we show how adding each of these factors changes the X scores of the 
fitted model. In this demonstration, we use a fixed gamma value of 0.5 so that only the 
parameter of interest varies. In a real-world scenario, you would typically decide whether 
to include observation weights and auxiliary responses before the main analysis, and then 
optimize gamma in a model that already includes these parameters.

We start with a plain model in which neither observational weights nore  auxiliary response 
information is considered.

```@example fit_da
class_weights = invfreqweights(classes)

spec = CPPLSSpec(
    ncomponents=2,
    gamma=0.5,
    mode=:discriminant
)

m_plain = fit(
    spec,
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
    title="CPPLS-DA scores from an inverse-frequency-weighted model",
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

spec = CPPLSSpec(
    ncomponents=2,
    gamma=0.5,
    mode=:discriminant
)

m_weighted = fit(
    spec,
    X,
    classes;
    obs_weights=class_weights,
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
auxiliary responses. For example, suppose the samples in the two groups were analyzed using 
two different instruments, and the choice of instrument is correlated with group 
membership. If most majority-class samples were measured with instrument A and most 
minority-class samples with instrument B, then instrument effects could confound the class 
separation.

If such auxiliary information is available, we can provide it to the model so that it can 
account for this confounding structure. This is done using the optional keyword argument 
`Y_aux` in the [`fit`](@ref) function. Supplying auxiliary response information changes 
how the latent variables are estimated: instead of forcing all supervised structure into 
the class contrast, the model can also explicitly model variation associated with the 
auxiliary responses. This helps ensure that the separation between the classes is not 
driven by confounding factors, but rather by the traits of true interest.

```@example fit_da
m_weighted_yaux = fit(
    spec,
    X,
    classes;
    obs_weights=class_weights,
    Y_aux=Y_aux,
    samplelabels=sample_labels
)

cppls_weighted_yaux_plt = scoreplot(
    sample_labels,
    classes,
    xscores(m_weighted_yaux, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores from an inverse-frequency-weighted model with auxiliary responses",
    group_order=["minor", "major"],
    group_marker=Dict("minor" => (; color=orange), "major" => (; color=blue)),
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
    xscores(m_weighted, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
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
    xscores(m_weighted_yaux, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
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
grayscale values are not randomly distributed across the score space. Instead, they are 
arranged roughly along the first latent dimension. This indicates that auxiliary signal 
correlated with class membership has leaked into the apparent class separation.

With the auxiliary response information included, as shown in the second plot, the
auxiliary structure is much less pronounced in the fitted scores, as indicated by the 
grayscale values being more evenly distributed across the plot.

### Choosing Gamma for the Weighted + Auxiliary Model

With observation weights and `Y_aux` in place, we can now choose `gamma` for the model we
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
weighted_yaux_grid_spec = CPPLSSpec(
    ncomponents=1,
    gamma=0:0.001:1,
    mode=:discriminant
)

weighted_yaux_grid_model = fit(
    weighted_yaux_grid_spec,
    X,
    classes;
    obs_weights=class_weights,
    Y_aux=Y_aux
)

weighted_yaux_grid_gammas = gammas(weighted_yaux_grid_model, 1)
weighted_yaux_grid_rhos = rhos(weighted_yaux_grid_model, 1)
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
    color=:grey70, linewidth=3)
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
    ncomponents=1,
    gamma=intervalize(0:0.05:1),
    mode=:discriminant
)

weighted_yaux_interval_model = fit(
    weighted_yaux_interval_spec,
    X,
    classes;
    obs_weights=class_weights,
    Y_aux=Y_aux
)

weighted_yaux_interval_gammas = gammas(weighted_yaux_interval_model, 1)
weighted_yaux_interval_rhos = rhos(weighted_yaux_interval_model, 1)
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

In this example, the interval-based optimization returns almost the same `gamma` value as
the dense grid search, so the difference is negligible in practice. That agreement should
not be taken for granted, however. If the objective landscape is rugged or contains
multiple local optima, an interval-based search can miss a narrow global maximum or favor
different local solutions depending on how the interval partition is chosen. It is
therefore advisable to first inspect the landscape on a grid and then decide whether a
coarse or dense interval partition is appropriate for downstream analyses such as cross-
validation and permutation testing.

Finally, we fit the two-component discriminant model with interval-optimized `gamma`
while keeping both inverse-frequency weights and `Y_aux`. This is the most favorable DA
setup examined in this example because weighting, auxiliary supervision, and gamma
selection are all aligned with the same objective.

```@example fit_da
weighted_yaux_best_spec = CPPLSSpec(
    ncomponents=2,
    gamma=intervalize(0:0.05:1),
    mode=:discriminant
)

m_weighted_yaux_best = fit(
    weighted_yaux_best_spec,
    X,
    classes;
    obs_weights=class_weights,
    Y_aux=Y_aux,
    samplelabels=sample_labels
)

selected_weighted_yaux_rhos = [
    rhos(m_weighted_yaux_best, lv)[findfirst(
        ==(gamma(m_weighted_yaux_best)[lv]),
        gammas(m_weighted_yaux_best, lv),
    )]
    for lv in 1:2
]

println("Selected gammas for the two latent variables: ",
    round.(gamma(m_weighted_yaux_best), digits=3))
println("Associated rho^2: ", round.(selected_weighted_yaux_rhos, digits=6))

cppls_weighted_yaux_best_plt = scoreplot(
    sample_labels,
    classes,
    xscores(m_weighted_yaux_best, 1:2);
    backend=:makie,
    figure_kwargs=(; size=(900, 600)),
    title="CPPLS-DA scores with weights, Y_aux, and optimized gamma",
    group_order=["minor", "major"],
    group_marker=Dict("minor" => (; color=orange), "major" => (; color=blue)),
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
