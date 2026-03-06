using CPPLS
using Test

@testset "CPPLS/types.jl" begin
    include(joinpath("CPPLS", "types.jl"))
end

@testset "CPPLS/scoreplot_dispatch" begin
    include(joinpath("CPPLS", "scoreplot_dispatch.jl"))
end

@testset "CPPLS/scoreplot_makie" begin
    include(joinpath("CPPLS", "scoreplot_makie.jl"))
end

if Base.find_package("PlotlyJS") !== nothing
    @testset "CPPLS/scoreplot_plotly" begin
        include(joinpath("CPPLS", "scoreplot_plotly.jl"))
    end
else
    @testset "CPPLS/scoreplot_plotly" begin
        @test true
    end
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
