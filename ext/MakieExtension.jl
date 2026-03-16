module MakieExtension

using Makie
import CPPLS

const _current_backend_ref = Ref{Function}(Makie.current_backend)

const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "ext", "makie_extensions", "scoreplot.jl"))

end
