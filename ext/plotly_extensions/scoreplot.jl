using CategoricalArrays

function CPPLS.scoreplot_plotly(
    samples::AbstractVector{<:AbstractString},
    groups,
    scores::AbstractMatrix{<:Real};
    group_order::Union{Nothing,AbstractVector}=nothing,
    default_trace=(;),
    group_trace::AbstractDict=Dict(),
    default_marker=(;),
    group_marker::AbstractDict=Dict(),
    hovertemplate::AbstractString="Sample: %{text}<br>Group: %{fullData.name}<br>LV1: %{x}<br>LV2: %{y}<extra></extra>",
    layout::Union{Nothing,PlotlyJS.Layout}=nothing,
    plot_kwargs=(;),
    show_legend::Union{Nothing,Bool}=nothing,
    title::AbstractString="Scores",
    xlabel::AbstractString="Latent Variable 1",
    ylabel::AbstractString="Latent Variable 2",
)
    size(scores, 2) >= 2 || error("scores must have at least two columns.")

    to_nt(x::NamedTuple) = x
    to_nt(x::AbstractDict) = (; (Symbol(k) => v for (k, v) in x)...)
    to_nt(::Nothing) = (;)

    groups_vec = collect(groups)
    order = if !isnothing(group_order)
        group_order
    elseif groups isa CategoricalArray
        levels(groups)
    else
        unique(groups_vec)
    end

    # Helper: get per-group kwargs by exact key, then by string key
    getkw(d, g) = haskey(d, g) ? d[g] :
                  haskey(d, string(g)) ? d[string(g)] : (;)

    traces = PlotlyJS.AbstractTrace[]
    for g in order
        idx = findall(==(g), groups_vec)
        isempty(idx) && continue

        xs = scores[idx, 1]
        ys = scores[idx, 2]
        sub_samples = samples[idx]

        tkw = merge(to_nt(default_trace), to_nt(getkw(group_trace, g)))
        show_legend === false && (tkw = merge(tkw, (showlegend=false,)))

        # If user provided marker in trace kwargs, use it as-is.
        hasmarker = haskey(tkw, :marker)

        mkw = merge(to_nt(default_marker), to_nt(getkw(group_marker, g)))

        basekw = (x=xs, y=ys)
        haskey(tkw, :text) || (basekw = merge(basekw, (text=sub_samples,)))
        haskey(tkw, :name) || (basekw = merge(basekw, (name=string(g),)))
        haskey(tkw, :mode) || (basekw = merge(basekw, (mode="markers",)))
        haskey(tkw, :hovertemplate) || (basekw = merge(basekw, (hovertemplate=hovertemplate,)))

        if hasmarker
            trace = PlotlyJS.scatter(; basekw..., tkw...)
        else
            marker = PlotlyJS.attr(; mkw...)
            trace = PlotlyJS.scatter(; basekw..., marker=marker, tkw...)
        end
        push!(traces, trace)
    end

    layout = isnothing(layout) ? PlotlyJS.Layout(
        title=title,
        xaxis=PlotlyJS.attr(title=xlabel),
        yaxis=PlotlyJS.attr(title=ylabel)
    ) : layout

    PlotlyJS.plot(traces, layout; to_nt(plot_kwargs)...)
end
