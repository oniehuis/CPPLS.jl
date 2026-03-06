module PlotlyJSExtension

using PlotlyJS
import CPPLS

const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "ext", "plotly_extensions", "scoreplot.jl"))

end
