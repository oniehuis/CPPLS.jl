using CategoricalArrays
using Makie
using Test

const MakieExt = Base.get_extension(CPPLS, :MakieExtension)
@test MakieExt !== nothing
@test CPPLS._require_extension(:MakieExtension, "Makie") === nothing

function set_backend(mod)
    MakieExt._current_backend_ref[] = () -> mod
    return nothing
end

set_backend(missing)

@testset "scoreplot_makie errors" begin
    samples = ["s1", "s2"]
    groups = [:a, :b]
    scores = reshape([1.0, 2.0], 2, 1)  # only one column
    @test_throws ErrorException CPPLS.scoreplot_makie(samples, groups, scores)
end

@testset "scoreplot_makie basic" begin
    set_backend(missing)

    samples = ["s1", "s2", "s3"]
    groups_cat = categorical(["a", "b", "a"]; levels = ["a", "b", "c"])
    scores = [1.0 2.0; 3.0 4.0; 5.0 6.0]

    # categorical order branch (includes missing level "c")
    fig1 = Figure(size = (200, 200))
    ax1 = Axis(fig1[1, 1])
    res1 = CPPLS.scoreplot_makie(
        samples,
        groups_cat,
        scores;
        figure = fig1,
        axis = ax1,
        axis_kwargs = nothing,
        legend_kwargs = nothing,
        show_legend = false,
        show_inspector = true,
        default_marker = Dict("markersize" => 5),
        group_trace = Dict("a" => (; alpha = 0.8)),
    )
    @test res1 === fig1

    # explicit order branch + label override + string-key fallback
    groups = [:a, :b, :a]
    fig2 = Figure(size = (200, 200))
    ax2 = Axis(fig2[1, 1])
    res2 = CPPLS.scoreplot_makie(
        samples,
        groups,
        scores;
        figure = fig2,
        axis = ax2,
        group_order = [:b, :a, :missing],
        show_legend = false,
        show_inspector = false,
        default_scatter = (; color = :red),
        default_trace = Dict(:alpha => 0.5),
        group_marker = Dict(
            "a" => (; label = "custom", color = :blue),
            "b" => (; color = :green),
        ),
    )
    @test res2 === fig2
end

@testset "scoreplot_makie nothing kwargs" begin
    set_backend(missing)

    samples = ["s1", "s2"]
    groups = [:a, :b]
    scores = [1.0 2.0; 3.0 4.0]

    res = CPPLS.scoreplot_makie(
        samples,
        groups,
        scores;
        figure = nothing,
        axis = nothing,
        figure_kwargs = nothing,
        axis_kwargs = nothing,
        legend_kwargs = nothing,
        show_legend = true,
        show_inspector = false,
        default_scatter = nothing,
        default_trace = nothing,
        default_marker = nothing,
    )
    @test res isa Figure
end

@testset "scoreplot_makie inspector" begin
    @eval Main module GLMakie end
    set_backend(Main.GLMakie)

    samples = ["s1", "s2"]
    groups = [:a, :b]
    scores = [1.0 2.0; 3.0 4.0]

    res = CPPLS.scoreplot_makie(
        samples,
        groups,
        scores;
        show_legend = true,
        show_inspector = true,
        legend_kwargs = Dict(:position => :rt),
        inspector_kwargs = Dict(:enabled => true),
    )
    @test res isa Figure
end

set_backend(missing)
