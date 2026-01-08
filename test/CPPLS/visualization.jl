using CategoricalArrays
using Makie
import Makie.ComputePipeline
using Test

const MakieExt = Base.get_extension(CPPLS, :MakieExtension)
const ResolveException = Makie.ComputePipeline.ResolveException

function dummy_cppls(; analysis = :discriminant)
    X = Float64[
        1 0
        0 1
        2 1
        3 2
    ]

    if analysis === :discriminant
        labels = categorical(["class1", "class2", "class1", "class2"])
        return CPPLS.fit_cppls(X, labels, 2; gamma = 0.5)
    else
        y = reshape(Float64[0.1, -0.2, 0.3, -0.4], :, 1)
        return CPPLS.fit_cppls(X, y, 2; gamma = 0.5)
    end
end

@testset "order_preserving_unique" begin
    labels = ["red", "blue", "red", "green", "blue", "yellow"]
    ordered = MakieExt.order_preserving_unique(labels)
    @test ordered == ["red", "blue", "green", "yellow"]

    @test MakieExt.order_preserving_unique(Any[]) == Any[]
end

@testset "matches_sample_length" begin
    @test MakieExt.matches_sample_length([1, 2, 3], 3)
    @test !MakieExt.matches_sample_length([1, 2], 3)
    @test MakieExt.matches_sample_length((:a, :b), 2)
    @test !MakieExt.matches_sample_length((:a, :b), 3)
    @test !MakieExt.matches_sample_length(:foo, 2)
    @test !MakieExt.matches_sample_length(:foo, :bar)
    @test invoke(MakieExt.matches_sample_length, Tuple{Any,Any}, :foo, 2) == false
    @test CPPLS.matches_sample_length(:foo, :bar) == false
end

@testset "normalize_palette" begin
    custom = MakieExt.normalize_palette((:red, :blue), 2)
    @test length(custom) == 2
    @test all(c -> c isa Makie.ColorTypes.Colorant, custom)

    fallback = MakieExt.normalize_palette([:red], 2)
    @test fallback == Makie.wong_colors(2)

    single = MakieExt.normalize_palette(:green, 1)
    @test length(single) == 1
    @test single[1] ≈ Makie.to_color(:green)

    bad_single = MakieExt.normalize_palette(("notacolor",), 1)
    @test bad_single == Makie.wong_colors(1)

    bad_default = MakieExt.normalize_palette("not-a-color", 1)
    @test bad_default == Makie.wong_colors(1)

    bad_entry = MakieExt.normalize_palette((:red, :invalid), 2)
    @test bad_entry == Makie.wong_colors(2)

    @test MakieExt.normalize_palette(nothing, 3) == Makie.wong_colors(3)
    auto_entries = MakieExt.normalize_palette(Makie.automatic, 2)
    @test auto_entries == Makie.wong_colors(2)
end

@testset "manual_color_sequence" begin
    labels = ["g1", "g2", "g1", "g3"]
    palette = MakieExt.manual_color_sequence((:red, :green, :blue), labels)
    @test palette[1] == Makie.to_color(:red)
    @test palette[2] == Makie.to_color(:green)
    @test palette[4] == Makie.to_color(:blue)

    @test length(MakieExt.manual_color_sequence(:purple, labels)) == length(labels)
    @test_throws ArgumentError MakieExt.manual_color_sequence([:red], labels)
    @test_throws ArgumentError MakieExt.manual_color_sequence(
        ("notacolor", :green, :blue),
        labels,
    )
    @test_throws ArgumentError MakieExt.manual_color_sequence("not-a-color", labels)
end

@testset "resolve_label_colors" begin
    labels = ["x", "y", "x", "z"]
    manual_colors =
        MakieExt.resolve_label_colors(labels, (:red, :green, :blue); manual = true)
    @test manual_colors[2] == Makie.to_color(:green)

    default = [:black, :white, :black, :white]
    @test MakieExt.resolve_label_colors(labels, default) === default

    @test MakieExt.resolve_label_colors(String[], :red) === :red

    auto = MakieExt.resolve_label_colors(labels, :cyan)
    expected = MakieExt.normalize_palette(:cyan, 3)
    @test Set(auto) == Set(expected[1:3])
end

@testset "apply_alpha_to_colors" begin
    base = [Makie.to_color(:red), Makie.to_color(:blue)]
    adjusted = MakieExt.apply_alpha_to_colors(base, 0.4)
    @test adjusted[1].alpha ≈ 0.4
    @test base[1].alpha == 1.0  # original unchanged

    @test MakieExt.apply_alpha_to_colors(base, 1.0) === base
    tuple_colors = (Makie.to_color(:red), Makie.to_color(:green))
    adjusted_tuple = MakieExt.apply_alpha_to_colors(tuple_colors, 0.2)
    @test adjusted_tuple[2].alpha ≈ 0.2

    dict_colors = Dict(:a => Makie.to_color(:black))
    dict_adjusted = MakieExt.apply_alpha_to_colors(dict_colors, 0.3)
    @test dict_adjusted[:a].alpha ≈ 0.3
    @test MakieExt.with_alpha(nothing, 0.5) === nothing
end

@testset "cppls_category_labels" begin
    fake_cppls = (; analysis_mode = :discriminant, da_categories = categorical(["a", "b"]))
    labels = MakieExt.cppls_category_labels(fake_cppls)
    @test labels == ["a", "b"]

    reg_cppls = (; analysis_mode = :regression, da_categories = nothing)
    @test MakieExt.cppls_category_labels(reg_cppls) == Any[]
end

@testset "scoreplot_kwdict" begin
    kw = MakieExt.scoreplot_kwdict(NamedTuple{(:markersize,),Tuple{Int}}((12,)))
    @test haskey(kw, :markersize)
    @test !haskey(kw, :color_manual)

    kw = MakieExt.scoreplot_kwdict((color = :red, marker = :circle))
    @test kw[:color_manual]
end

@testset "maybe_apply_axis_label!" begin
    fig = Makie.Figure(size = (100, 100))
    ax = Makie.Axis(fig[1, 1])

    MakieExt.maybe_apply_axis_label!(
        ax,
        :xlabel,
        MakieExt.SCOREPLOT_AUTO_LABEL,
        "Compound 1",
    )
    @test Makie.to_value(ax.xlabel) == "Compound 1"

    MakieExt.maybe_apply_axis_label!(ax, :xlabel, "Custom Label", "ignored")
    @test Makie.to_value(ax.xlabel) == "Custom Label"

    MakieExt.maybe_apply_axis_label!(ax, :xlabel, nothing, "fallback")
    @test Makie.to_value(ax.xlabel) == "Custom Label"
end

@testset "merge_axis_defaults" begin
    merged = MakieExt.merge_axis_defaults((xlabel = "X",))
    @test merged[:xlabel] == "X"
    @test merged[:ylabel] == "Compound 2"

    fallback = MakieExt.merge_axis_defaults(:not_a_tuple)
    @test fallback == MakieExt.SCOREPLOT_AXIS_DEFAULTS
end

@testset "scoreplot wrappers" begin
    da_model = dummy_cppls()
    plot_obj = CPPLS.scoreplot(da_model; axis = (xlabel = "Scores",))
    @test plot_obj isa Makie.FigureAxisPlot
    @test Makie.to_value(plot_obj.axis.xlabel) == "Scores"

    fig = Makie.Figure(size = (120, 120))
    ax = Makie.Axis(fig[1, 1])
    plot_res = CPPLS.scoreplot!(ax, da_model; xlabel = "X", ylabel = "Y")
    @test plot_res isa Makie.Plot
    @test Makie.to_value(ax.xlabel) == "X"
    @test Makie.to_value(ax.ylabel) == "Y"

    plot_auto = CPPLS.scoreplot!(da_model; color = :green)
    @test plot_auto isa Makie.Plot

    reg_model = dummy_cppls(analysis = :regression)
    @test CPPLS.scoreplot(reg_model) isa Makie.FigureAxisPlot

    @test_throws ResolveException CPPLS.scoreplot(da_model; dims = (1, 3))
    @test_throws ResolveException CPPLS.scoreplot(
        da_model;
        color = (:red,),
    )
end
