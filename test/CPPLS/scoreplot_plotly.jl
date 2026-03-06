using CategoricalArrays
using PlotlyJS
using Test

const PlotlyExt = Base.get_extension(CPPLS, :PlotlyJSExtension)
@test PlotlyExt !== nothing

@testset "scoreplot_plotly errors" begin
    samples = ["s1", "s2"]
    groups = [:a, :b]
    scores = [1.0; 2.0]
    @test_throws ErrorException CPPLS.scoreplot_plotly(samples, groups, scores)
end

@testset "scoreplot_plotly branches" begin
    samples = ["s1", "s2", "s3"]
    groups = [:a, :b, :a]
    scores = [1.0 2.0; 3.0 4.0; 5.0 6.0]

    default_trace = Dict(
        "marker" => PlotlyJS.attr(color = "red"),
    )
    group_trace = Dict(
        "b" => Dict(
            "name" => "Bee",
            "hovertemplate" => "B",
            "text" => ["tb"],
        ),
    )
    default_marker = Dict("size" => 9)
    group_marker = Dict("a" => Dict("color" => "blue"))

    res = CPPLS.scoreplot_plotly(
        samples,
        groups,
        scores;
        group_order = [:b, :missing, :a],
        default_trace = default_trace,
        group_trace = group_trace,
        default_marker = default_marker,
        group_marker = group_marker,
        show_legend = false,
        plot_kwargs = Dict("config" => PlotlyJS.PlotConfig(displayModeBar = false)),
    )
    @test res isa PlotlyJS.Plot
    @test length(res.data) == 2

    groups_cat = categorical(["x", "y", "x"]; levels = ["x", "y", "z"])
    layout = PlotlyJS.Layout(title = "Custom")
    res2 = CPPLS.scoreplot_plotly(
        samples,
        groups_cat,
        scores;
        default_trace = nothing,
        default_marker = nothing,
        layout = layout,
        group_marker = Dict("x" => Dict("color" => "green")),
    )
    @test res2 isa PlotlyJS.Plot
    @test res2.layout.title == "Custom"
end
