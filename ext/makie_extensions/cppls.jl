import CPPLS:
    scoreplot,
    scoreplot!,
    scoreplot_color_mapping,
    scoreplot_colors,
    response_label_colors,
    plot_projection!,
    safe_axislegend,
    matches_sample_length

const SCOREPLOT_AXIS_DEFAULTS = (xlabel = "Component 1", ylabel = "Component 2")
const SCOREPLOT_AXIS_DEFAULTS_3D = (
    type = Makie.Axis3,
    xlabel = "Component 1",
    ylabel = "Component 2",
    zlabel = "Component 3",
)
const SCOREPLOT_AUTO_LABEL = gensym(:scoreplot_auto_label)

Base.@doc CPPLS.SCOREPLOT_DOC scoreplot

is_automatic(value) =
    value isa Makie.Automatic || value === Makie.Automatic() || value === Makie.automatic

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

function response_label_colors(cppls, labels; color = Makie.automatic, color_manual = false, alpha = 1.0)
    label_strings = string.(labels)
    label_keys = if hasproperty(cppls, :response_labels) && cppls.response_labels !== nothing
        unique(string.(cppls.response_labels))
    else
        String[]
    end
    if isempty(label_keys) || length(label_keys) < length(unique(label_strings))
        label_keys = unique(label_strings)
    end

    n_groups = length(label_keys)
    provided_palette = (color isa AbstractVector && !(color isa AbstractString)) || color isa Tuple
    palette = if color_manual || provided_palette
        entries = collect(color)
        length(entries) == n_groups || throw(ArgumentError("response label palette must have $n_groups colors"))
        [Makie.to_color(entry) for entry in entries]
    elseif color === Makie.automatic || color === Makie.Automatic() || color === nothing
        Makie.wong_colors(max(n_groups, 1))
    else
        fill(Makie.to_color(color), n_groups)
    end

    if alpha isa Number && alpha != 1
        palette = [Makie.RGBAf(Makie.to_color(c), Float32(clamp(alpha, 0, 1))) for c in palette]
    end

    mapping = Dict(label_keys[i] => palette[i] for i in eachindex(label_keys))
    missing = [lbl for lbl in label_strings if !haskey(mapping, lbl)]
    isempty(missing) || throw(ArgumentError("response label colors missing label(s): $(join(unique(missing), ", "))"))
    return [mapping[lbl] for lbl in label_strings]
end

function scoreplot_color_mapping(
    cppls;
    color = Makie.automatic,
    color_manual = false,
    color_by_response = true,
    alpha = 1.0,
)
    label_values = cppls_category_labels(cppls)
    isempty(label_values) && return Dict{String,Makie.ColorTypes.Colorant}()

    colors = if color_manual && !isempty(label_values)
        resolve_label_colors(label_values, color; manual = true)
    elseif color_by_response && !isempty(label_values)
        resolve_label_colors(label_values, color)
    else
        color
    end

    colors = apply_alpha_to_colors(colors, alpha)

    if colors isa AbstractVector || colors isa Tuple
        length(colors) == length(label_values) || throw(
            ArgumentError(
                "scoreplot color mapping expected $(length(label_values)) color(s), got $(length(colors))",
            ),
        )
        palette = [Makie.to_color(entry) for entry in colors]
        return Dict(label_values[i] => palette[i] for i in eachindex(label_values))
    end

    if is_automatic_color(colors) || colors === nothing
        return Dict{String,Makie.ColorTypes.Colorant}()
    end

    colorant = Makie.to_color(colors)
    return Dict(label_values[i] => colorant for i in eachindex(label_values))
end

function scoreplot_colors(
    cppls,
    labels;
    color = Makie.automatic,
    color_manual = false,
    color_by_response = true,
    alpha = 1.0,
)
    mapping = scoreplot_color_mapping(
        cppls;
        color = color,
        color_manual = color_manual,
        color_by_response = color_by_response,
        alpha = alpha,
    )
    isempty(mapping) && return Makie.ColorTypes.Colorant[]

    label_strings = string.(labels)
    missing_labels = String[]
    seen_missing = Set{String}()
    for label in label_strings
        if !haskey(mapping, label) && !(label in seen_missing)
            push!(missing_labels, label)
            push!(seen_missing, label)
        end
    end

    if !isempty(missing_labels)
        throw(
            ArgumentError(
                "scoreplot colors missing label(s): $(join(missing_labels, ", ")); " *
                "ensure labels are part of the fitted CPPLS categories",
            ),
        )
    end

    return [mapping[label] for label in label_strings]
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

function resolve_sample_labels(cppls, labels, show_labels)
    show_labels || return Any[]
    if labels === nothing
        return Any[]
    elseif is_automatic(labels)
        hasproperty(cppls, :sample_labels) || return Any[]
        return cppls.sample_labels
    else
        return labels
    end
end

@recipe ScorePlotPlot (cppls,) begin
    dims = (1, 2)
    "Color samples by stored categorical responses when available."
    color_by_response = true
    color_manual = false
    alpha = 1.0
    "Show text labels above each sample point."
    show_labels = false
    "Show sample labels when hovering a point (interactive backends only)."
    hover_labels = true
    "Labels to display when `show_labels` is true. Defaults to stored sample labels."
    labels = Makie.automatic
    label_color = :black
    label_fontsize = 10
    label_align = (:center, :bottom)

    color = @inherit markercolor
    marker = @inherit marker
    markersize = @inherit markersize
    strokecolor = @inherit markerstrokecolor
    strokewidth = @inherit markerstrokewidth

    Makie.mixin_generic_plot_attributes()...
end

function Makie.plot!(plot::ScorePlotPlot{<:Tuple{<:CPPLS.AbstractCPPLS}})
    input_nodes =
        [
            :cppls,
            :dims,
            :color,
            :color_by_response,
            :color_manual,
            :alpha,
            :labels,
            :show_labels,
            :hover_labels,
        ]
    output_nodes = [
        :score_x,
        :score_y,
        :score_z,
        :point_color,
        :point_labels,
        :labels_available,
        :is_3d,
    ]

    map!(
        plot.attributes,
        input_nodes,
        output_nodes,
    ) do cppls,
        dims,
        default_color,
        color_by_response,
        color_manual,
        alpha,
        labels,
        show_labels,
        hover_labels
        hasproperty(cppls, :X_scores) || throw(
            ArgumentError("scoreplot requires a CPPLS model with stored X_scores."),
        )
        dims_tuple = Tuple(dims)
        length(dims_tuple) in (2, 3) || throw(
            ArgumentError(
                "Attribute `dims` must have two or three component indices, got $dims",
            ),
        )

        dims_len = length(dims_tuple)
        dims_int = ntuple(i -> Int(dims_tuple[i]), dims_len)
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

        n_samples = size(scores, 1)
        label_values = resolve_sample_labels(cppls, labels, show_labels || hover_labels)
        if !isempty(label_values) && !matches_sample_length(label_values, n_samples)
            throw(
                ArgumentError(
                    "scoreplot labels must have length $n_samples, got $(length(label_values))",
                ),
            )
        end

        labels_available = !isempty(label_values)
        label_text = labels_available ? string.(label_values) : fill("", n_samples)

        return (
            view(scores, :, dims_int[1]),
            view(scores, :, dims_int[2]),
            dims_len == 3 ? view(scores, :, dims_int[3]) : nothing,
            colors,
            label_text,
            labels_available,
            dims_len == 3,
        )
    end

    is_3d = Makie.to_value(plot.is_3d)
    scatter_plot = if is_3d
        scatter!(
            plot,
            plot.attributes,
            plot.score_x,
            plot.score_y,
            plot.score_z;
            color = plot.point_color,
        )
    else
        scatter!(
            plot,
            plot.attributes,
            plot.score_x,
            plot.score_y;
            color = plot.point_color,
        )
    end

    if is_3d
        text!(
            plot,
            plot.score_x,
            plot.score_y,
            plot.score_z;
            text = plot.point_labels,
            visible = plot.show_labels,
            color = plot.label_color,
            fontsize = plot.label_fontsize,
            align = plot.label_align,
            inspectable = false,
        )
    else
        text!(
            plot,
            plot.score_x,
            plot.score_y;
            text = plot.point_labels,
            visible = plot.show_labels,
            color = plot.label_color,
            fontsize = plot.label_fontsize,
            align = plot.label_align,
            inspectable = false,
        )
    end

    if Makie.to_value(plot.hover_labels)
        hover_text = Makie.Observable("")
        hover_visible = Makie.Observable(false)
        hover_pos = is_3d ? Makie.Observable(Makie.Point3f(0, 0, 0)) :
                    Makie.Observable(Makie.Point2f(0, 0))

        if is_3d
            text!(
                plot,
                hover_pos;
                text = hover_text,
                visible = hover_visible,
                color = plot.label_color,
                fontsize = plot.label_fontsize,
                align = plot.label_align,
                inspectable = false,
            )
        else
            text!(
                plot,
                hover_pos;
                text = hover_text,
                visible = hover_visible,
                color = plot.label_color,
                fontsize = plot.label_fontsize,
                align = plot.label_align,
                inspectable = false,
            )
        end

        scene = Makie.parent_scene(plot)
        range = 10.0
        Makie.on(Makie.events(scene).mouseposition) do mp
            if !plot.hover_labels[] || !plot.labels_available[]
                hover_visible[] = false
                return
            end
            screen = Makie.getscreen(scene)
            if screen === nothing
                hover_visible[] = false
                return
            end
            scene_w, scene_h = Makie.widths(scene)
            if scene_w <= 0 || scene_h <= 0
                hover_visible[] = false
                return
            end
            picked_plot = nothing
            idx = 0
            try
                picked_plot, idx = Makie.pick(scene, mp, range)
            catch
                hover_visible[] = false
                return
            end
            if picked_plot === scatter_plot && idx != 0
                pos = Makie.position_on_plot(scatter_plot, idx, apply_transform = false)
                hover_pos[] = pos
                hover_text[] = plot.point_labels[][idx]
                hover_visible[] = true
            else
                hover_visible[] = false
            end
            return
        end
    end

    return plot
end

Makie.convert_arguments(::Type{<:ScorePlotPlot}, cppls::CPPLS.AbstractCPPLS) = (cppls,)

axis_defaults_for_dims(dims) = begin
    dims_tuple = try
        Tuple(dims)
    catch
        ()
    end
    length(dims_tuple) == 3 ? SCOREPLOT_AXIS_DEFAULTS_3D : SCOREPLOT_AXIS_DEFAULTS
end

merge_axis_defaults(axis::NamedTuple, dims) = merge(axis_defaults_for_dims(dims), axis)
merge_axis_defaults(axis, dims) = axis_defaults_for_dims(dims)
merge_axis_defaults(axis::NamedTuple) = merge_axis_defaults(axis, (1, 2))
merge_axis_defaults(axis) = merge_axis_defaults(axis, (1, 2))

default_scoreplot_dims(cppls) = begin
    hasproperty(cppls, :X_scores) || return (1, 2)
    backend = try
        Makie.current_backend()
    catch
        nothing
    end
    if backend !== nothing && !ismissing(backend) && nameof(backend) == :CairoMakie
        return (1, 2)
    end
    n_components = size(cppls.X_scores, 2)
    return n_components > 2 ? (1, 2, 3) : (1, 2)
end

function scoreplot_kwdict(kwargs)
    kwdict = Dict{Symbol,Any}(pairs(kwargs))
    haskey(kwdict, :color) && (kwdict[:color_manual] = true)
    return kwdict
end

function add_scoreplot_legend!(
    axis,
    cppls;
    color = Makie.automatic,
    color_manual = false,
    color_by_response = true,
    alpha = 1.0,
    position = :rb,
    marker = :circle,
    markersize = 8,
)
    mapping = scoreplot_color_mapping(
        cppls;
        color = color,
        color_manual = color_manual,
        color_by_response = color_by_response,
        alpha = alpha,
    )
    isempty(mapping) && return nothing

    labels = sort!(unique(string.(cppls_category_labels(cppls))))
    for key in labels
        haskey(mapping, key) || continue
        scatter!(axis, [NaN], [NaN]; color = mapping[key], label = key, marker = marker, markersize = markersize)
    end

    return axislegend(axis; position = position)
end

function scoreplot(
    args...;
    axis = NamedTuple(),
    show_legend = true,
    legend_position = :rb,
    legend_marker = :circle,
    legend_markersize = 8,
    kwargs...,
)
    kwdict = scoreplot_kwdict(kwargs)
    dims = haskey(kwdict, :dims) ? kwdict[:dims] : default_scoreplot_dims(first(args))
    axis_kw = axis isa NamedTuple ? merge_axis_defaults(axis, dims) : axis_defaults_for_dims(dims)
    result = scoreplotplot(args...; axis = axis_kw, (; kwdict...)...)
    if result isa Makie.FigureAxisPlot
        cppls = first(args)
        if show_legend && !isempty(cppls_category_labels(cppls))
            add_scoreplot_legend!(
                result.axis,
                cppls;
                color = get(kwdict, :color, Makie.automatic),
                color_manual = get(kwdict, :color_manual, false),
                color_by_response = get(kwdict, :color_by_response, true),
                alpha = get(kwdict, :alpha, 1.0),
                position = legend_position,
                marker = legend_marker,
                markersize = legend_markersize,
            )
        end
    end
    return result
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
    zlabel = SCOREPLOT_AUTO_LABEL,
    show_legend = true,
    legend_position = :rb,
    legend_marker = :circle,
    legend_markersize = 8,
    kwargs...,
)
    kwdict = scoreplot_kwdict(kwargs)
    plot = scoreplotplot!(axis, args...; (; kwdict...)...)
    maybe_apply_axis_label!(axis, :xlabel, xlabel, SCOREPLOT_AXIS_DEFAULTS.xlabel)
    maybe_apply_axis_label!(axis, :ylabel, ylabel, SCOREPLOT_AXIS_DEFAULTS.ylabel)
    if hasproperty(axis, :zlabel)
        maybe_apply_axis_label!(axis, :zlabel, zlabel, SCOREPLOT_AXIS_DEFAULTS_3D.zlabel)
    end
    cppls = first(args)
    if show_legend && !isempty(cppls_category_labels(cppls))
        add_scoreplot_legend!(
            axis,
            cppls;
            color = get(kwdict, :color, Makie.automatic),
            color_manual = get(kwdict, :color_manual, false),
            color_by_response = get(kwdict, :color_by_response, true),
            alpha = get(kwdict, :alpha, 1.0),
            position = legend_position,
            marker = legend_marker,
            markersize = legend_markersize,
        )
    end
    return plot
end

function scoreplot!(
    args...;
    show_legend = true,
    legend_position = :rb,
    legend_marker = :circle,
    legend_markersize = 8,
    kwargs...,
)
    kwdict = scoreplot_kwdict(kwargs)
    return scoreplotplot!(args...; (; kwdict...)...)
end

function plot_projection!(
    ax,
    cppls,
    scores,
    bins,
    Y_project,
    Y_predicted;
    correct = (marker = :cross, markersize = 12, alpha = 0.8, strokecolor = :black, strokewidth = 0.50),
    wrong = (marker = :cross, markersize = 8, alpha = 0.1, strokecolor = :black, strokewidth = 0.0),
    style_by_match = true,
    color_by = :pred_bins,
    correct_color = nothing,
    wrong_color = nothing,
    show_labels = false,
    labels = nothing,
    label_color = :black,
    label_fontsize = 10,
    label_align = (:center, :bottom),
    show_legend = true,
    legend_position = :tr,
    legend_labels = ("correct", "wrong"),
)
    correct isa NamedTuple || throw(ArgumentError("`correct` must be a NamedTuple"))
    wrong isa NamedTuple || throw(ArgumentError("`wrong` must be a NamedTuple"))

    mask = vec(all(Y_project .== Y_predicted; dims = 2))
    if !style_by_match
        mask = trues(length(mask))
    end
    scores_correct = scores[mask, :]
    scores_wrong = scores[.!mask, :]

    label_source = if color_by == :true_bins
        bins
    elseif color_by == :pred_bins
        pred_indices = CPPLS.one_hot_to_labels(Y_predicted)
        if hasproperty(cppls, :response_labels) && cppls.response_labels !== nothing
            string.(cppls.response_labels)[pred_indices]
        else
            string.(cppls.da_categories)[pred_indices]
        end
    elseif color_by == :fixed
        nothing
    else
        throw(ArgumentError("`color_by` must be :true_bins, :pred_bins, or :fixed"))
    end

    colors_correct = if color_by == :fixed
        correct_color === nothing && throw(ArgumentError("`correct_color` must be set when color_by=:fixed"))
        correct_color
    else
        response_label_colors(cppls, label_source[mask])
    end

    colors_wrong = if color_by == :fixed
        wrong_color === nothing && throw(ArgumentError("`wrong_color` must be set when color_by=:fixed"))
        wrong_color
    else
        response_label_colors(cppls, label_source[.!mask])
    end

    if !isempty(scores_correct)
        kwargs = merge((color = colors_correct,), correct)
        scatter!(ax, scores_correct[:, 1], scores_correct[:, 2]; kwargs...)
    end
    if !isempty(scores_wrong)
        kwargs = merge((color = colors_wrong,), wrong)
        scatter!(ax, scores_wrong[:, 1], scores_wrong[:, 2]; kwargs...)
    end

    if show_labels
        labels === nothing && throw(ArgumentError("`labels` must be provided when show_labels=true"))
        length(labels) == size(scores, 1) || throw(ArgumentError("`labels` must match number of rows in scores"))
        text!(ax, scores[:, 1], scores[:, 2];
            text = string.(labels),
            color = label_color,
            fontsize = label_fontsize,
            align = label_align,
        )
    end

    if show_legend
        correct_label, wrong_label = legend_labels
        scatter!(ax, [NaN], [NaN];
            label = correct_label,
            marker = get(correct, :marker, :cross),
            markersize = get(correct, :markersize, 12),
            color = :black,
            strokecolor = get(correct, :strokecolor, :black),
            strokewidth = get(correct, :strokewidth, 0.50),
        )
        scatter!(ax, [NaN], [NaN];
            label = wrong_label,
            marker = get(wrong, :marker, :cross),
            markersize = get(wrong, :markersize, 8),
            color = :black,
            strokecolor = get(wrong, :strokecolor, :black),
            strokewidth = get(wrong, :strokewidth, 0.0),
        )
        hide_axis_legends!(ax)
        safe_axislegend(ax; position = normalize_legend_position(legend_position), merge = true, unique = true)
    end

    return nothing
end

function safe_axislegend(ax; kwargs...)
    try
        axislegend(ax; kwargs...)
    catch err
        if err isa ErrorException && occursin("There are no plots with labels", err.msg)
            return nothing
        end
        rethrow()
    end
end

function normalize_legend_position(pos)
    pos isa Symbol || return pos
    pos === :tr && return :rt
    pos === :tl && return :lt
    pos === :br && return :rb
    pos === :bl && return :lb
    return pos
end

function hide_axis_legends!(ax)
    parent = getproperty(ax, :parent)
    parent isa Makie.Figure || return nothing
    ax_bbox = try
        Makie.to_value(ax.scene.viewport)
    catch
        nothing
    end
    for block in parent.content
        block isa Makie.Legend || continue
        if hasproperty(block, :bbox)
            bbox = try
                Makie.to_value(getproperty(block, :bbox))
            catch
                nothing
            end
            if ax_bbox === nothing || bbox === nothing || bbox == ax_bbox
                Makie.hide!(block)
            end
        else
            Makie.hide!(block)
        end
    end
    return nothing
end
