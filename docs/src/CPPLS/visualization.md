# Visualization

[`scoreplot`](@ref) provides a common interface for visualizing latent scores from
CPPLS models. The Plotly backend is most useful for exploratory work because the
resulting figures are interactive, whereas the Makie backend is better suited to
static figures for reports and documentation. The examples below show both a basic
score plot and a more customized workflow in which projected samples are added with
user-defined class order and marker styles.

## Example

We reuse the synthetic discriminant-analysis dataset introduced on the [Fit](@ref)
page. To keep the focus on visualization, the examples below use models that have
already been specified with observation weights and, where relevant, auxiliary
responses.

The packages loaded below play different roles: `CPPLS` provides the modeling,
projection, and plotting functions, `JLD2` reads the example dataset from disk,
`CairoMakie` renders static figures for the documentation, `PlotlyJS` exports
interactive HTML figures, and `Statistics` is used to orient the score axes
consistently.

Most of the following setup is only needed to obtain data and fitted models that can
then be plotted. It is therefore not specific to `scoreplot` itself, except for the
backend imports and the color definitions used to keep the examples visually consistent.

```@example visualization
using CPPLS
using JLD2
using CairoMakie
using PlotlyJS
using Statistics

sample_labels, X, classes, Y_aux = load(
	CPPLS.dataset("synthetic_cppls_da_dataset.jld2"),
	"sample_labels",
	"X",
	"classes",
	"Y_aux"
)

# Use Makie's Wong palette so the class colors stay consistent across examples.
wong = Makie.wong_colors()
wong_plotly = Dict(
	"major" => "rgb(0,114,178)",
	"minor" => "rgb(230,159,0)"
)

# Orient score axes only for examples where separately computed scores must be compared.
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

# Apply the same sign convention to projected scores so they match the training plot.
function orient_like(reference_scores, reference_classes, scores; reference_class="major")
	oriented = copy(scores)
	reference_idx = reference_classes .== reference_class
	for lv in axes(oriented, 2)
		if mean(reference_scores[reference_idx, lv]) < 0
			oriented[:, lv] .*= -1
		end
	end
	oriented
end

basic_spec = CPPLSSpec(
	n_components=2,
	gamma=0.5,
	analysis_mode=:discriminant
)

basic_model = fit(
	basic_spec,
	X,
	classes;
	obs_weights=invfreqweights(classes),
	Y_aux=Y_aux,
	sample_labels=sample_labels
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

advanced_spec = CPPLSSpec(
	n_components=2,
	gamma=intervalize(0:0.25:1),
	analysis_mode=:discriminant
)

advanced_model = fit(
	advanced_spec,
	X_train,
	classes_train;
	obs_weights=invfreqweights(classes_train),
	Y_aux=Y_aux_train,
	sample_labels=labels_train
)

train_scores = orient_scores(X_scores(advanced_model)[:, 1:2], classes_train)
heldout_scores = orient_like(X_scores(advanced_model)[:, 1:2], classes_train,
	project(advanced_model, X_holdout))
nothing # hide
```

## Basic Score Plot

The most common use case is simply `scoreplot(model)`, where the function extracts the
stored sample labels, sample classes, and first two latent variables directly from the
fitted model. In that situation, no further arguments are required unless you want to
customize the appearance. Titles, marker sizes, class order, colors, and other styling
options can all be set explicitly, but it is useful to first see the minimal form. The
more advanced example below then illustrates how those options can be used in practice.

With the Makie backend, the result is a static figure that can be embedded directly into
the documentation.

```@example visualization
basic_makie = scoreplot(
	basic_model;
	backend=:makie
)
save("scoreplot_basic_makie.svg", basic_makie)
nothing # hide
```

![](scoreplot_basic_makie.svg)

The same score plot can also be generated with the Plotly backend. In the static
documentation, it is most practical to export the interactive figure as HTML and link
to it.

```@example visualization
basic_plotly = scoreplot(
	basic_model;
	backend=:plotly
)
PlotlyJS.savefig(basic_plotly, "scoreplot_basic_plotly.html")
nothing # hide
```

[Open the interactive Plotly version](scoreplot_basic_plotly.html)

## Projected Samples With Custom Styling

`scoreplot` also works well when training samples and projected samples need to be shown
in the same score space. In the next example, we fit a model after holding out one
sample from each class, project those held-out samples into the fitted latent space, and
then distinguish them from the training samples by marker shape while keeping the class
colors fixed.

```@example visualization
advanced_makie = scoreplot(
	vcat(labels_train, labels_holdout),
	vcat(classes_train, plot_classes_holdout),
	vcat(train_scores, heldout_scores);
	backend=:makie,
	figure_kwargs=(; size=(900, 600)),
	title="CPPLS-DA scores with projected samples",
	group_order=["minor", "projected minor", "major", "projected major"],
	group_marker=Dict(
		"minor" => (; color=wong[2]),
		"projected minor" => (; color=wong[2], marker=:rect, markersize=18,
			strokecolor=:black, strokewidth=2),
		"major" => (; color=wong[1]),
		"projected major" => (; color=wong[1], marker=:rect, markersize=18,
			strokecolor=:black, strokewidth=2)
	),
	default_marker=(; markersize=14)
)
save("scoreplot_projected_makie.svg", advanced_makie)
nothing # hide
```

![](scoreplot_projected_makie.svg)

The same idea can be implemented with the Plotly backend. Here the projected samples are
drawn with square markers and black outlines so that they remain visually distinct in the
interactive view.

```@example visualization
advanced_plotly = scoreplot(
	vcat(labels_train, labels_holdout),
	vcat(classes_train, plot_classes_holdout),
	vcat(train_scores, heldout_scores);
	backend=:plotly,
	title="CPPLS-DA scores with projected samples",
	group_order=["minor", "projected minor", "major", "projected major"],
	group_marker=Dict(
		"minor" => (; color=wong_plotly["minor"]),
		"projected minor" => (; color=wong_plotly["minor"], symbol="square",
			line=PlotlyJS.attr(color="black", width=1.5)),
		"major" => (; color=wong_plotly["major"]),
		"projected major" => (; color=wong_plotly["major"], symbol="square",
			line=PlotlyJS.attr(color="black", width=1.5))
	),
	default_marker=(; size=11)
)
PlotlyJS.savefig(advanced_plotly, "scoreplot_projected_plotly.html")
nothing # hide
```

[Open the interactive projected Plotly version](scoreplot_projected_plotly.html)

These two examples illustrate the main intended use of the backends. Makie is a good
default choice for publication-style static figures, whereas Plotly is especially useful
for exploratory inspection of scores because sample identities and exact coordinates are
available interactively.

## API

```@docs
CPPLS.scoreplot
```
