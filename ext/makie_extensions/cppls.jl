import CPPLS: scoreplot, scoreplot!, matches_sample_length

const SCOREPLOT_AXIS_DEFAULTS = (xlabel = "Compound 1", ylabel = "Compound 2")
const SCOREPLOT_AUTO_LABEL = gensym(:scoreplot_auto_label)

Base.@doc CPPLS.SCOREPLOT_DOC scoreplot

is_automatic_color(value) =
    value isa Makie.Automatic || value === Makie.Automatic() || value === Makie.automatic

function normalize_palette(default_color, n_unique)
    fallback = Makie.wong_colors(max(n_unique, 1))
    provided_palette =
        default_color isa AbstractVector && !(default_color isa AbstractString) ||
        default_color isa Tuple

    entries = if provided_palette
        collect(default_color)
    elseif is_automatic_color(default_color) || default_color === nothing
        []
    else
        [default_color]
    end

    if isempty(entries)
        return fallback
    end

    if provided_palette
        if n_unique > length(entries)
            return fallback
        end

        palette = Vector{Makie.ColorTypes.Colorant}(undef, length(entries))
        for (idx, entry) in pairs(entries)
            try
                palette[idx] = Makie.to_color(entry)
            catch
                return fallback
            end
        end
        return palette
    else
        if n_unique == 1
            try
                color = Makie.to_color(entries[1])
                return [color]
            catch
                return fallback
            end
        else
            return fallback
        end
    end
end

function order_preserving_unique(labels)
    seen = Dict{Any,Bool}()
    ordered = Any[]
    for label in labels
        haskey(seen, label) && continue
        seen[label] = true
        push!(ordered, label)
    end
    return ordered
end

function manual_color_sequence(default_color, labels)
    unique_labels = order_preserving_unique(labels)
    n_groups = length(unique_labels)

    if (default_color isa AbstractVector && !(default_color isa AbstractString)) ||
       default_color isa Tuple
        entries = collect(default_color)
        if length(entries) != n_groups
            throw(
                ArgumentError(
                    "scoreplot color palette provides $(length(entries)) color(s) but the data contains $n_groups label group(s); supply one color per group",
                ),
            )
        end
        palette = [
            try
                Makie.to_color(entry)
            catch err
                throw(
                    ArgumentError(
                        "scoreplot color palette entry `$entry` is not a valid color specification: $(err)",
                    ),
                )
            end for entry in entries
        ]
    else
        color = try
            Makie.to_color(default_color)
        catch err
            throw(
                ArgumentError(
                    "`color`=`$(default_color)` is not a valid color specification: $(err)",
                ),
            )
        end
        palette = fill(color, n_groups)
    end

    lookup = Dict(unique_labels[i] => palette[i] for i in eachindex(unique_labels))
    return [lookup[label] for label in labels]
end

function resolve_label_colors(labels, default_color; manual = false)
    isempty(labels) && return default_color
    n_samples = length(labels)

    if manual
        return manual_color_sequence(default_color, labels)
    end

    if matches_sample_length(default_color, n_samples)
        return default_color
    end

    unique_labels = order_preserving_unique(labels)
    palette = normalize_palette(default_color, length(unique_labels))
    palette_cycle = [palette[(i-1)%length(palette)+1] for i = 1:length(unique_labels)]
    lookup = Dict(unique_labels[i] => palette_cycle[i] for i in eachindex(unique_labels))
    return [lookup[label] for label in labels]
end

function apply_alpha_to_colors(colors, alpha)
    alpha isa Number || return colors
    isone(alpha) && return colors
    return with_alpha(colors, clamp(alpha, 0, 1))
end

with_alpha(colors::AbstractVector, alpha) = [with_alpha(color, alpha) for color in colors]
with_alpha(colors::Tuple, alpha) = tuple((with_alpha(color, alpha) for color in colors)...)
with_alpha(color::Dict, alpha) = Dict(k => with_alpha(v, alpha) for (k, v) in color)
with_alpha(color::Nothing, _) = nothing
with_alpha(color, alpha) =
    is_automatic_color(color) ? color : Makie.RGBAf(Makie.to_color(color), Float32(alpha))

function cppls_category_labels(cppls)
    if hasproperty(cppls, :analysis_mode) &&
       hasproperty(cppls, :da_categories) &&
       cppls.analysis_mode === :discriminant &&
       cppls.da_categories !== nothing
        data = cppls.da_categories
        return Vector{String}(string.(data))
    else
        return Any[]
    end
end

@recipe ScorePlotPlot (cppls,) begin
    dims = (1, 2)
    "Color samples by stored categorical responses when available."
    color_by_response = true
    color_manual = false
    alpha = 1.0

    color = @inherit markercolor
    marker = @inherit marker
    markersize = @inherit markersize
    strokecolor = @inherit markerstrokecolor
    strokewidth = @inherit markerstrokewidth

    Makie.mixin_generic_plot_attributes()...
end

function Makie.plot!(plot::ScorePlotPlot{<:Tuple{<:CPPLS.AbstractCPPLS}})
    input_nodes = [:cppls, :dims, :color, :color_by_response, :color_manual, :alpha]
    output_nodes = [:score_x, :score_y, :point_color]

    map!(
        plot.attributes,
        input_nodes,
        output_nodes,
    ) do cppls, dims, default_color, color_by_response, color_manual, alpha
        hasproperty(cppls, :X_scores) || throw(
            ArgumentError("scoreplot requires a CPPLS model with stored X_scores."),
        )
        dims_tuple = Tuple(dims)
        length(dims_tuple) == 2 || throw(
            ArgumentError(
                "Attribute `dims` must be a tuple with two component indices, got $dims",
            ),
        )

        dims_int = ntuple(i -> Int(dims_tuple[i]), 2)
        scores = cppls.X_scores
        n_components = size(scores, 2)
        any(d -> d < 1 || d > n_components, dims_int) && throw(
            ArgumentError(
                "Model stores $n_components components, but dims=$dims_int requested",
            ),
        )

        label_values = cppls_category_labels(cppls)
        colors = if color_manual && !isempty(label_values)
            resolve_label_colors(label_values, default_color; manual = true)
        elseif color_by_response && !isempty(label_values)
            resolve_label_colors(label_values, default_color)
        else
            default_color
        end

        colors = apply_alpha_to_colors(colors, alpha)

        return (view(scores, :, dims_int[1]), view(scores, :, dims_int[2]), colors)
    end

    scatter!(plot, plot.attributes, plot.score_x, plot.score_y; color = plot.point_color)

    return plot
end

Makie.convert_arguments(::Type{<:ScorePlotPlot}, cppls::CPPLS.AbstractCPPLS) = (cppls,)

merge_axis_defaults(axis::NamedTuple) = merge(SCOREPLOT_AXIS_DEFAULTS, axis)
merge_axis_defaults(axis) = SCOREPLOT_AXIS_DEFAULTS

function scoreplot_kwdict(kwargs)
    kwdict = Dict{Symbol,Any}(pairs(kwargs))
    haskey(kwdict, :color) && (kwdict[:color_manual] = true)
    return kwdict
end

function scoreplot(args...; axis = NamedTuple(), kwargs...)
    axis_kw = axis isa NamedTuple ? merge_axis_defaults(axis) : SCOREPLOT_AXIS_DEFAULTS
    kwdict = scoreplot_kwdict(kwargs)
    return scoreplotplot(args...; axis = axis_kw, (; kwdict...)...)
end

label_is_empty(val) = val === nothing || (val isa AbstractString && isempty(val))

function maybe_apply_axis_label!(axis, prop::Symbol, value, default_text)
    label_attr = getproperty(axis, prop)
    if value === SCOREPLOT_AUTO_LABEL
        current = Makie.to_value(label_attr)
        label_is_empty(current) && setproperty!(axis, prop, default_text)
    elseif value !== nothing
        setproperty!(axis, prop, value)
    end
end

Base.@doc CPPLS.SCOREPLOT_BANG_DOC scoreplot!

function scoreplot!(
    axis::Makie.AbstractAxis,
    args...;
    xlabel = SCOREPLOT_AUTO_LABEL,
    ylabel = SCOREPLOT_AUTO_LABEL,
    kwargs...,
)
    kwdict = scoreplot_kwdict(kwargs)
    plot = scoreplotplot!(axis, args...; (; kwdict...)...)
    maybe_apply_axis_label!(axis, :xlabel, xlabel, SCOREPLOT_AXIS_DEFAULTS.xlabel)
    maybe_apply_axis_label!(axis, :ylabel, ylabel, SCOREPLOT_AXIS_DEFAULTS.ylabel)
    return plot
end

function scoreplot!(args...; kwargs...)
    kwdict = scoreplot_kwdict(kwargs)
    return scoreplotplot!(args...; (; kwdict...)...)
end
