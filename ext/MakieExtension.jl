module MakieExtension

using Makie
import CPPLS

const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "ext", "makie_extensions", "cppls.jl"))

end
