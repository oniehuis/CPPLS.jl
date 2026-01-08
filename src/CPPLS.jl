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
labels (`cppls.da_categories`), samples are colored by group automatically. You
can override this via `color`:

- Scalar color ⇒ every sample uses that color.
- Vector/Tuple ⇒ treated as a palette. Because the plot colors discriminant groups
  automatically, the length must match the number of unique labels (order follows
  the stored response labels). If `color` matches the number of rows it is
  interpreted per sample.

Scatter-style attributes such as `marker`, `markersize`, `strokecolor`,
`strokewidth`, `alpha`, and the scoreplot-specific `dims` keyword (selecting
which CPPLS score components to display) pass straight through to Makie.

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
override `xlabel`/`ylabel`. Returns the created `Plot`.
"""
function scoreplot! end
Base.@doc SCOREPLOT_BANG_DOC scoreplot!

end # module CPPLS
