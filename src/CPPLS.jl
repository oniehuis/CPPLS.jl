module CPPLS

using LinearAlgebra
using Optim
using Random
import StatsAPI: fit, predict, fitted, coef, residuals
using Statistics
using StatsBase
using CategoricalArrays

using Reexport: @reexport
@reexport using CategoricalArrays

include("CPPLS/types.jl")
include("CPPLS/preprocessing.jl")
include("CPPLS/cca.jl")
include("CPPLS/fit.jl")
include("CPPLS/fit_light.jl")
include("CPPLS/predict.jl")
include("CPPLS/metrics.jl")
include("CPPLS/crossvalidation.jl")

include("Utils/encoding.jl")
include("Utils/paths.jl")
include("Utils/statistics.jl")

export AbstractCPPLSFit
export CPPLSFit
export CPPLSFitLight
export CPPLSModel
export coef
export cvda
export cvreg
export fit
export fitted
export gamma
export gammas
export intervalize
export invfreqweights
export mode
export ncomponents
export nestedcv
export nestedcvperm
export onehot
export outlierscan
export permda
export permreg
export predict
export predictorlabels
export project
export projectionmatrix
export pvalue
export coefall
export residuals
export responselabels
export rhos
export sampleclasses
export samplelabels
export scoreplot
export xmean
export xstd
export xscores
export ymean
export ystd

# Score plot backend dispatch (actual methods live in the optional dependencies)
const SCOREPLOT_DOC = """
    scoreplot(samples, groups, scores; backend=:plotly, kwargs...)
    scoreplot(cppls; backend=:plotly, kwargs...)

Backend dispatcher for CPPLS score plots. Use `backend=:plotly` (default) for the
PlotlyJS extension or `backend=:makie` for the Makie extension.

The dispatcher accepts *backend-agnostic* keywords and passes any remaining
keywords to the selected backend. To avoid confusion, think of the keywords as
belonging to three groups:

General (backend-agnostic)
- `backend::Symbol = :plotly`
  Select the backend. Supported values: `:plotly`, `:makie`.

PlotlyJS backend keywords (PlotlyJSExtension)
- `group_order::Union{Nothing,AbstractVector} = nothing`
  Order of groups (also draw order; later is on top). If `nothing`, uses
  `levels(groups)` for `CategoricalArray`, else `unique(groups)`.
- `default_trace = (;)`
  PlotlyJS scatter kwargs applied to every group (except marker).
- `group_trace::AbstractDict = Dict()`
  Per-group PlotlyJS scatter kwargs.
- `default_marker = (;)`
  PlotlyJS marker kwargs for every group (keys must be `Symbol`s).
- `group_marker::AbstractDict = Dict()`
  Per-group marker kwargs (keys must be `Symbol`s).
- `hovertemplate::AbstractString = "Sample: %{text}<br>Group: %{fullData.name}<br>LV1: %{x}<br>LV2: %{y}<extra></extra>"`
  Hover text template. The default shows sample, group, LV1, LV2.
- `layout::Union{Nothing,PlotlyJS.Layout} = nothing`
  Layout object; if `nothing`, a default layout is created using `title`, `xlabel`,
  and `ylabel`.
- `plot_kwargs = (;)`
  Extra kwargs passed to `PlotlyJS.plot` (e.g., `config`).
- `show_legend::Union{Nothing,Bool} = nothing`
  If `false`, sets `showlegend=false` for all traces.
- `title::AbstractString = "Scores"`
- `xlabel::AbstractString = "Latent Variable 1"`
- `ylabel::AbstractString = "Latent Variable 2"`

Makie backend keywords (MakieExtension)
- `group_order::Union{Nothing,AbstractVector} = nothing`
  Order of groups (also draw order).
- `default_scatter = (;)`
  Makie scatter kwargs applied to every group.
- `group_scatter::AbstractDict = Dict()`
  Per-group scatter kwargs.
- `default_trace = (;)`
  Additional scatter kwargs applied to every group (legacy convenience).
- `group_trace::AbstractDict = Dict()`
  Per-group scatter kwargs (legacy convenience).
- `default_marker = (;)`
  Marker-related kwargs applied to every group.
- `group_marker::AbstractDict = Dict()`
  Per-group marker kwargs.
- `title::AbstractString = "Scores"`
- `xlabel::AbstractString = "Latent Variable 1"`
- `ylabel::AbstractString = "Latent Variable 2"`
- `figure = nothing`
  Provide an existing `Figure` to draw into.
- `axis = nothing`
  Provide an existing `Axis` to draw into.
- `figure_kwargs = (;)`
  Extra kwargs passed to `Figure` when it is created.
- `axis_kwargs = (;)`
  Extra kwargs passed to `Axis` when it is created.
- `show_legend::Bool = true`
  If `true`, calls `axislegend`.
- `legend_kwargs = (;)`
  Extra kwargs passed to `axislegend`.
- `show_inspector::Bool = true`
  If `true`, enables `DataInspector` on GLMakie/WGLMakie.
- `inspector_kwargs = (;)`
  Extra kwargs passed to `DataInspector`.

Notes
- The dispatcher checks that the requested backend is loaded and errors with
  "Backend <pkg> not loaded" if not.
- Unknown backend values throw `error("Unknown backend")`.
- `scores` must have at least two columns (LV1 and LV2).
"""
function scoreplot end
Base.@doc SCOREPLOT_DOC scoreplot

function scoreplot_plotly end
function scoreplot_makie end

function _require_extension(extsym::Symbol, pkg::AbstractString)
    Base.get_extension(@__MODULE__, extsym) === nothing &&
        error("Backend $(pkg) not loaded. Run `using $(pkg)` first.")
    return nothing
end

const _require_extension_ref = Ref{Function}(_require_extension)
const _scoreplot_plotly_ref = Ref{Function}(
  (samples, groups, scores; kwargs...) -> scoreplot_plotly(samples, groups, scores; kwargs...)
)
const _scoreplot_makie_ref = Ref{Function}(
  (samples, groups, scores; kwargs...) -> scoreplot_makie(samples, groups, scores; kwargs...)
)

function scoreplot(
    samples::AbstractVector{<:AbstractString},
    groups,
    scores::AbstractMatrix{<:Real};
    backend::Symbol = :plotly,
    kwargs...,
)
    if backend === :plotly
    _require_extension_ref[](:PlotlyJSExtension, "PlotlyJS")
    return _scoreplot_plotly_ref[](samples, groups, scores; kwargs...)
    elseif backend === :makie
    _require_extension_ref[](:MakieExtension, "Makie")
    return _scoreplot_makie_ref[](samples, groups, scores; kwargs...)
    else
        error("Unknown backend")
    end
end

function scoreplot(
    cppls::CPPLSFit;
    backend::Symbol = :plotly,
    kwargs...,
)
    groups = cppls.sampleclasses
    isnothing(groups) && throw(ArgumentError(
        "scoreplot(cppls) requires sampleclasses in the fitted model. " *
        "Use discriminant fits from class labels or call scoreplot(samples, groups, scores) directly."
    ))

    samples = cppls.samplelabels
    scores = cppls.T[:, 1:2]
    if backend === :plotly
      _require_extension_ref[](:PlotlyJSExtension, "PlotlyJS")
      return _scoreplot_plotly_ref[](samples, groups, scores; kwargs...)
    elseif backend === :makie
      _require_extension_ref[](:MakieExtension, "Makie")
      return _scoreplot_makie_ref[](samples, groups, scores; kwargs...)
    else
        error("Unknown backend")
    end
end

end # module CPPLS
