using CategoricalArrays

function CPPLS.scoreplot_makie(
    samples::AbstractVector{<:AbstractString},
    groups,
    scores::AbstractMatrix{<:Real};
    group_order::Union{Nothing,AbstractVector}=nothing,
    default_scatter=(;),
    group_scatter::AbstractDict=Dict(),
    default_trace=(;),
    group_trace::AbstractDict=Dict(),
    default_marker=(;),
    group_marker::AbstractDict=Dict(),
    title::AbstractString="Scores",
    xlabel::AbstractString="Latent Variable 1",
    ylabel::AbstractString="Latent Variable 2",
    figure=nothing,
    axis=nothing,
    figure_kwargs=(;),
    axis_kwargs=(;),
    show_legend::Bool=true,
    legend_kwargs=(;),
    show_inspector::Bool=true,
    inspector_kwargs=(;),
)

    size(scores, 2) >= 2 || error("scores must have at least two columns.")

    to_nt(x::NamedTuple) = x
    to_nt(x::AbstractDict) = (; (Symbol(k) => v for (k, v) in x)...)
    to_nt(::Nothing) = (;)

    groups_vec = collect(groups)
    n_samples = length(groups_vec)
    order = if !isnothing(group_order)
        group_order
    elseif groups isa CategoricalArray
        levels(groups)
    else
        unique(groups_vec)
    end

    getkw(d, g) = haskey(d, g) ? d[g] :
                  haskey(d, string(g)) ? d[string(g)] : (;)
    subset_kw(kw, idx) = (; (
        key => (value isa AbstractVector && length(value) == n_samples ? value[idx] : value)
        for (key, value) in pairs(kw)
    )...)

    fig = isnothing(figure) ? Makie.Figure(; to_nt(figure_kwargs)...) : figure
    ax = isnothing(axis) ? Makie.Axis(
        fig[1, 1];
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        to_nt(axis_kwargs)...
    ) : axis

    base_scatter = merge(
        to_nt(default_scatter),
        to_nt(default_trace),
        to_nt(default_marker)
    )

    backend_mod = _current_backend_ref[]()
    backend_name = backend_mod === missing ? :missing : nameof(backend_mod)
    inspector_enabled = show_inspector && (backend_name == :GLMakie || backend_name == :WGLMakie)

    for g in order
        idx = findall(==(g), groups_vec)
        isempty(idx) && continue

        xs = scores[idx, 1]
        ys = scores[idx, 2]
        sub_samples = samples[idx]

        kw = merge(
            base_scatter,
            to_nt(getkw(group_scatter, g)),
            to_nt(getkw(group_trace, g)),
            to_nt(getkw(group_marker, g))
        )
        kw = subset_kw(kw, idx)
        haskey(kw, :label) || (kw = merge(kw, (label=string(g),)))

        if inspector_enabled
            haskey(kw, :inspectable) || (kw = merge(kw, (inspectable=true,)))
            haskey(kw, :inspector_label) || (kw = merge(kw, (inspector_label = (plot, i, pos) -> begin
                s = sub_samples[i]
                "Sample: $(s)\nGroup: $(g)\nLV1: $(round(pos[1]; digits=4))\nLV2: $(round(pos[2]; digits=4))"
            end,)))
        end

        Makie.scatter!(ax, xs, ys; kw...)
    end

    show_legend && Makie.axislegend(ax; to_nt(legend_kwargs)...)
    inspector_enabled && Makie.DataInspector(fig; to_nt(inspector_kwargs)...)
    fig
end
