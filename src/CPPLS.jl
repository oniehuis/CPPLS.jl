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
export nested_cv_permutation
export nested_cv
export calculate_p_value
export separationaxis
export fisherztrack
export invfreqweights
export intervalize
export labels_to_one_hot
export one_hot_to_labels
export find_invariant_and_variant_columns
export decision_line

matches_sample_length(value::AbstractVector, n) = length(value) == n
matches_sample_length(value::Tuple, n) = length(value) == n
matches_sample_length(::Any, ::Any) = Base.inferencebarrier(false)

# Score plot backend dispatch (actual methods live in the optional dependencies)
const SCOREPLOT_DOC = """
    scoreplot(samples, groups, scores; backend=:plotly, kwargs...)
    scoreplot(cppls; backend=:plotly, kwargs...)

Backend dispatcher for CPPLS score plots. Use `backend=:plotly` (default) for the
PlotlyJS extension or `backend=:makie` for the Makie extension.
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

function scoreplot(
    samples::AbstractVector{<:AbstractString},
    groups,
    scores::AbstractMatrix{<:Real};
    backend::Symbol = :plotly,
    kwargs...,
)
    if backend === :plotly
        _require_extension(:PlotlyJSExtension, "PlotlyJS")
        return scoreplot_plotly(samples, groups, scores; kwargs...)
    elseif backend === :makie
        _require_extension(:MakieExtension, "Makie")
        return scoreplot_makie(samples, groups, scores; kwargs...)
    else
        error("Unknown backend")
    end
end

function scoreplot(
    cppls::CPPLS;
    backend::Symbol = :plotly,
    kwargs...,
)
    samples = cppls.sample_labels
    groups = cppls.da_categories
    scores = cppls.X_scores[:, 1:2]
    if backend === :plotly
        _require_extension(:PlotlyJSExtension, "PlotlyJS")
        return scoreplot_plotly(samples, groups, scores; kwargs...)
    elseif backend === :makie
        _require_extension(:MakieExtension, "Makie")
        return scoreplot_makie(samples, groups, scores; kwargs...)
    else
        error("Unknown backend")
    end
end

end # module CPPLS
