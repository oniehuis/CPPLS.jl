module CPPLS

using LinearAlgebra
using Optim
using Random
using Statistics
using StatsBase
using CategoricalArrays

using Reexport: @reexport
@reexport using CategoricalArrays

include("CPPLS/types.jl")
include("CPPLS/preprocessing.jl")
include("CPPLS/cca.jl")
include("CPPLS/fit.jl")
include("CPPLS/predict.jl")
include("CPPLS/metrics.jl")
include("CPPLS/crossvalidation.jl")

include("Utils/encoding.jl")
include("Utils/matrix.jl")
include("Utils/statistics.jl")

export CPPLS
export CPPLSLight
export fit_cppls
export fit_cppls_light
export predict
export predictonehot
export project
export scoreplot
export scoreplot!
export scoreplot_color_mapping
export scoreplot_colors
export response_label_colors
export plot_projection!
export safe_axislegend
export nested_cv_permutation
export nested_cv
export calculate_p_value
export separationaxis
export fisherztrack
export intervalize
export labels_to_one_hot
export one_hot_to_labels
export find_invariant_and_variant_columns
export decision_line

matches_sample_length(value::AbstractVector, n) = length(value) == n
matches_sample_length(value::Tuple, n) = length(value) == n
matches_sample_length(::Any, ::Any) = Base.inferencebarrier(false)

# Makie extension hooks (actual methods live in the Makie optional dependency)
const SCOREPLOT_DOC = """
    scoreplot(cppls; kwargs...) -> Makie.FigureAxisPlot / Plot

Keyword-friendly wrapper around the Makie recipe for CPPLS score plots. Accepts
any `CPPLS` model (or arguments compatible with `scoreplotplot`) and forwards
keywords down to the recipe while supplying sensible axis defaults (`"Compound 1"`
and `"Compound 2"` unless you override them with the `axis` keyword).

Color handling is specialised. When the fitted model stores discriminant-analysis
labels (`cppls.da_categories`), samples are colored by group automatically and a
legend is added by default. You can override this via `color`:

- Scalar color ⇒ every sample uses that color.
- Vector/Tuple ⇒ treated as a palette. Because the plot colors discriminant groups
  automatically, the length must match the number of unique labels (order follows
  the stored response labels). If `color` matches the number of rows it is
  interpreted per sample.

Scatter-style attributes such as `marker`, `markersize`, `strokecolor`,
`strokewidth`, `alpha`, and the scoreplot-specific `dims` keyword (selecting
which CPPLS score components to display) pass straight through to Makie. If
`dims` is not specified and the model stores more than two components, a 3D
score plot (`dims=(1,2,3)`) is selected automatically (an `Axis3` is used when
`axis` is not specified). Set `show_labels=true` to display sample labels above
each point; `labels` defaults to the stored `sample_labels` but can be overridden
with a vector (length must match the number of samples). Text styling is
controlled with `label_color`, `label_fontsize`, and `label_align`.

For interactive backends (e.g. GLMakie), set `hover_labels=true` to show a fixed
label at the hovered sample point (no additional calls required).

Use `show_legend=false` to suppress the automatically generated legend, and
`legend_position`, `legend_marker`, and `legend_markersize` to adjust its style.

Returns the `Plot` object created by `scoreplotplot`, matching Makie’s usual
figure/axis semantics. The implementation itself is provided by the Makie
extension module once Makie is loaded.
"""
function scoreplot end
Base.@doc SCOREPLOT_DOC scoreplot

const SCOREPLOT_BANG_DOC = """
    scoreplot!(axis::Makie.AbstractAxis, cppls; kwargs...) -> Plot
    scoreplot!(args...; kwargs...) -> Plot

In-place variants of [`scoreplot`](@ref) that draw into an existing Makie axis
(first form) or accept the same positional arguments Makie’s `scoreplotplot!`
expects (second form). Both forward the scatter keywords (`color`, `marker`,
`markersize`, `strokecolor`, `strokewidth`, `alpha`, …) plus the
scoreplot-specific `dims` selector and keep automatic axis labelling unless you
override `xlabel`/`ylabel`. Use `show_labels=true` to draw sample labels, with
the same `labels`/`label_*` keywords described in [`scoreplot`](@ref). Returns
the created `Plot`.

For interactive backends (e.g. GLMakie), set `hover_labels=true` to show a fixed
label at the hovered sample point (no additional calls required).

Use `show_legend=false` to suppress the legend and `legend_position`,
`legend_marker`, and `legend_markersize` to adjust its style.
"""
function scoreplot! end
Base.@doc SCOREPLOT_BANG_DOC scoreplot!

const SCOREPLOT_COLOR_MAPPING_DOC = """
    scoreplot_color_mapping(cppls; kwargs...) -> Dict

Return a mapping from stored discriminant-analysis labels to the colors used by
`scoreplot`. This is useful for coloring projected or predicted samples so they
match the training score plot. The implementation lives in the Makie extension
and is available once Makie is loaded.
"""
function scoreplot_color_mapping end
Base.@doc SCOREPLOT_COLOR_MAPPING_DOC scoreplot_color_mapping

const SCOREPLOT_COLORS_DOC = """
    scoreplot_colors(cppls, labels; kwargs...) -> Vector

Return a vector of colors aligned with `labels`, using the same label-to-color
mapping as `scoreplot`. The implementation lives in the Makie extension and is
available once Makie is loaded.
"""
function scoreplot_colors end
Base.@doc SCOREPLOT_COLORS_DOC scoreplot_colors

const RESPONSE_LABEL_COLORS_DOC = """
    response_label_colors(cppls, labels; kwargs...) -> Vector

Return a vector of colors aligned with `labels`, using the response-label palette
stored on the fitted CPPLS model. The implementation lives in the Makie extension
and is available once Makie is loaded.
"""
function response_label_colors end
Base.@doc RESPONSE_LABEL_COLORS_DOC response_label_colors

const PLOT_PROJECTION_DOC = """
    plot_projection!(ax, cppls, scores, bins, Y_project, Y_predicted; kwargs...) -> Nothing

Plot projected scores with different styling for correctly and incorrectly
classified samples. Set `show_legend=true` (default) to add legend entries for
the correct/wrong symbols; use `legend_position` (default `:tr`, accepted aliases
`:tr/:tl/:br/:bl` mapped to Makie’s `:rt/:lt/:rb/:lb`) and `legend_labels` to
customize it. When enabled, any existing axis legend is hidden and replaced with
a combined legend. The implementation lives in the Makie extension and is
available once Makie is loaded.
"""
function plot_projection! end
Base.@doc PLOT_PROJECTION_DOC plot_projection!

const SAFE_AXISLEGEND_DOC = """
    safe_axislegend(ax; kwargs...) -> Union{Nothing, Makie.AxisLegend}

Call `axislegend` but return `nothing` when no plots provide labels. The
implementation lives in the Makie extension and is available once Makie is loaded.
"""
function safe_axislegend end
Base.@doc SAFE_AXISLEGEND_DOC safe_axislegend

end # module CPPLS
