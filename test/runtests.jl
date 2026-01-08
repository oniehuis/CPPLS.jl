using CPPLS
using Test

@testset "CPPLS/types.jl" begin
    include(joinpath("CPPLS", "types.jl"))
end

@testset "CPPLS/predict" begin
    include(joinpath("CPPLS", "predict.jl"))
end

@testset "CPPLS/cca" begin
    include(joinpath("CPPLS", "cca.jl"))
end

@testset "CPPLS/fit" begin
    include(joinpath("CPPLS", "fit.jl"))
end

@testset "CPPLS/preprocessing" begin
    include(joinpath("CPPLS", "preprocessing.jl"))
end

@testset "CPPLS/visualization" begin
    include(joinpath("CPPLS", "visualization.jl"))
end


@testset "CPPLS/crossvalidation" begin
    include(joinpath("CPPLS", "crossvalidation.jl"))
end

@testset "CPPLS/metrics" begin
    include(joinpath("CPPLS", "metrices.jl"))
end

@testset "Utils/encoding" begin
    include(joinpath("Utils", "encoding.jl"))
end

@testset "Utils/matrix" begin
    include(joinpath("Utils", "matrix.jl"))
end

@testset "Utils/statistics" begin
    include(joinpath("Utils", "statistics.jl"))
end
