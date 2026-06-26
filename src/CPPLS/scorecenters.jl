"""
    ScoreCenters

Centers of stored training samples or projected samples in CPPLS score space, grouped by class.
"""
struct ScoreCenters
    classes
    sampleindices
    samplelabels
    centers
    comps
    center_method
end

"""
    ScoreRepresentatives

Samples nearest to class centers in CPPLS score space.
"""
struct ScoreRepresentatives
    classes
    sampleindices
    samplelabels
    scores
    distances
    centers
    comps
    center_method
end

function Base.show(io::IO, sc::ScoreCenters)
    print(io, "ScoreCenters(",
        "classes=", length(sc.classes),
        ", components=", repr(sc.comps),
        ", center=", repr(sc.center_method),
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", sc::ScoreCenters)
    println(io, "ScoreCenters")
    println(io, "  classes: ", repr(sc.classes))
    println(io, "  components: ", repr(sc.comps))
    print(io, "  center: ", sc.center_method)
end

function Base.show(io::IO, sr::ScoreRepresentatives)
    n_rep = isempty(sr.sampleindices) ? 0 : maximum(length, sr.sampleindices)
    print(io, "ScoreRepresentatives(",
        "classes=", length(sr.classes),
        ", representatives=", n_rep,
        ", components=", repr(sr.comps),
        ", center=", repr(sr.center_method),
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", sr::ScoreRepresentatives)
    println(io, "ScoreRepresentatives")
    println(io, "  classes: ", repr(sr.classes))
    println(io, "  representatives per class: ", repr(length.(sr.sampleindices)))
    println(io, "  components: ", repr(sr.comps))
    print(io, "  center: ", sr.center_method)
end

"""
    scorecenters(fit::CPPLSFit; comps=1:ncomponents(fit), center=:median)

Calculate one score-space center per stored sample class in a fitted CPPLS model.
`center` can be `:median` or `:mean`.
"""
function scorecenters(
    fit::CPPLSFit;
    comps=1:ncomponents(fit),
    center::Symbol=:median,
)
    classes = sampleclasses(fit)
    isnothing(classes) && throw(ArgumentError(
        "scorecenters(fit) requires sampleclasses in the fitted model. " *
        "Pass `sampleclasses=` to `fit`."))

    scorecenters(xscores(fit), classes;
        samplelabels=samplelabels(fit),
        comps=comps,
        center=center,
    )
end

"""
    scorecenters(scores, classes; samplelabels=string.(axes(scores, 1)), comps=axes(scores, 2), center=:median)

Calculate one score-space center per class from a score matrix and class labels.
`scores` can be stored training scores or projected scores for new samples.
`comps` can be one component or any valid subset of score columns.
"""
function scorecenters(
    scores::AbstractMatrix{<:Real},
    classes::AbstractVector;
    samplelabels=string.(axes(scores, 1)),
    comps=axes(scores, 2),
    center::Symbol=:median,
)
    n_samples = size(scores, 1)
    length(classes) == n_samples || throw(ArgumentError(
        "`classes` must have length $n_samples, got $(length(classes))"))
    length(samplelabels) == n_samples || throw(ArgumentError(
        "`samplelabels` must have length $n_samples, got $(length(samplelabels))"))

    comps = normalize_scorecenter_comps(comps, size(scores, 2))
    validate_scorecenter_method(center)

    grouped_classes = collect(unique(classes))
    grouped_indices = [findall(==(class), classes) for class in grouped_classes]
    grouped_labels = [collect(samplelabels[inds]) for inds in grouped_indices]
    centers = Matrix{Float64}(undef, length(grouped_classes), length(comps))

    for (i, inds) in pairs(grouped_indices)
        centers[i, :] = scorecenter_values(view(scores, inds, comps), center)
    end

    ScoreCenters(
        grouped_classes,
        grouped_indices,
        grouped_labels,
        centers,
        comps,
        center,
    )
end

"""
    scorerepresentatives(fit::CPPLSFit; comps=1:ncomponents(fit), center=:median, n=1)

Return the `n` training samples per class nearest to their score-space class center.
Distances are Euclidean in the selected score components.
"""
function scorerepresentatives(
    fit::CPPLSFit;
    comps=1:ncomponents(fit),
    center::Symbol=:median,
    n::Integer=1,
)
    classes = sampleclasses(fit)
    isnothing(classes) && throw(ArgumentError(
        "scorerepresentatives(fit) requires sampleclasses in the fitted model. " *
        "Pass `sampleclasses=` to `fit`."))

    scorerepresentatives(xscores(fit), classes;
        samplelabels=samplelabels(fit),
        comps=comps,
        center=center,
        n=n,
    )
end

"""
    scorerepresentatives(scores, classes; samplelabels=string.(axes(scores, 1)), comps=axes(scores, 2), center=:median, n=1)

Return the `n` samples per class nearest to their score-space class center.
`scores` can be stored training scores or projected scores for new samples.
Distances are Euclidean in the selected score components. `comps` can be
one component or any valid subset of score columns.
"""
function scorerepresentatives(
    scores::AbstractMatrix{<:Real},
    classes::AbstractVector;
    samplelabels=string.(axes(scores, 1)),
    comps=axes(scores, 2),
    center::Symbol=:median,
    n::Integer=1,
)
    n > 0 || throw(ArgumentError("`n` must be greater than zero"))

    centers = scorecenters(scores, classes;
        samplelabels=samplelabels,
        comps=comps,
        center=center,
    )
    representative_indices = Vector{Vector{Int}}(undef, length(centers.classes))
    representative_labels = similar(centers.samplelabels)
    representative_scores = Vector{Matrix{Float64}}(undef, length(centers.classes))
    representative_distances = Vector{Vector{Float64}}(undef, length(centers.classes))

    for i in eachindex(centers.classes)
        inds = centers.sampleindices[i]
        class_scores = view(scores, inds, centers.comps)
        class_center = view(centers.centers, i, :)
        dists = scorecenter_distances(class_scores, class_center)
        order = sort(collect(eachindex(dists)); by=j -> (dists[j], inds[j]))
        selected = order[1:min(Int(n), length(order))]

        representative_indices[i] = inds[selected]
        representative_labels[i] = centers.samplelabels[i][selected]
        representative_scores[i] = Matrix{Float64}(class_scores[selected, :])
        representative_distances[i] = dists[selected]
    end

    ScoreRepresentatives(
        centers.classes,
        representative_indices,
        representative_labels,
        representative_scores,
        representative_distances,
        centers,
        centers.comps,
        centers.center_method,
    )
end

function normalize_scorecenter_comps(comp::Integer, ncomps::Integer)
    1 <= comp <= ncomps || throw(ArgumentError(
        "Component index $comp out of bounds (1:$ncomps)"))
    [Int(comp)]
end

function normalize_scorecenter_comps(comps::AbstractVector{<:Integer}, ncomps::Integer)
    isempty(comps) && throw(ArgumentError("`comps` must not be empty"))
    all(1 .<= comps .<= ncomps) || throw(ArgumentError(
        "Component indices $(comps) out of bounds (1:$ncomps)"))
    collect(Int, comps)
end

function normalize_scorecenter_comps(comps::AbstractUnitRange{<:Integer}, ncomps::Integer)
    isempty(comps) && throw(ArgumentError("`comps` must not be empty"))
    (1 <= first(comps) <= ncomps && 1 <= last(comps) <= ncomps) || throw(
        ArgumentError("Component range $(comps) out of bounds (1:$ncomps)"))
    collect(Int, comps)
end

function validate_scorecenter_method(center::Symbol)
    center in (:median, :mean) || throw(ArgumentError(
        "`center` must be :median or :mean, got $(repr(center))"))
    center
end

function scorecenter_values(scores::AbstractMatrix{<:Real}, center::Symbol)
    if center === :median
        return vec(median(scores; dims=1))
    elseif center === :mean
        return vec(mean(scores; dims=1))
    end

    throw(ArgumentError("`center` must be :median or :mean, got $(repr(center))"))
end

function scorecenter_distances(scores::AbstractMatrix{<:Real}, center)
    distances = Vector{Float64}(undef, size(scores, 1))

    for i in axes(scores, 1)
        total = 0.0
        for j in axes(scores, 2)
            total += abs2(scores[i, j] - center[j])
        end
        distances[i] = sqrt(total)
    end

    distances
end

function scorecenter_classindex(sc::ScoreCenters, class)
    idx = findfirst(c -> isequal(c, class), sc.classes)
    idx !== nothing && return idx

    string_matches = findall(c -> string(c) == string(class), sc.classes)
    length(string_matches) == 1 && return only(string_matches)

    throw(KeyError(class))
end

function scorerepresentative_classindex(sr::ScoreRepresentatives, class)
    idx = findfirst(c -> isequal(c, class), sr.classes)
    idx !== nothing && return idx

    string_matches = findall(c -> string(c) == string(class), sr.classes)
    length(string_matches) == 1 && return only(string_matches)

    throw(KeyError(class))
end

"""
    sampleindices(sc::ScoreCenters, class)

Return the original sample indices used for one class center.
"""
sampleindices(sc::ScoreCenters, class) = sc.sampleindices[scorecenter_classindex(sc, class)]

"""
    sampleindices(sr::ScoreRepresentatives, class)

Return the original sample indices selected as representatives for one class.
"""
sampleindices(sr::ScoreRepresentatives, class) =
    sr.sampleindices[scorerepresentative_classindex(sr, class)]

"""
    samplelabels(sc::ScoreCenters, class)

Return the sample labels used for one class center.
"""
samplelabels(sc::ScoreCenters, class) = sc.samplelabels[scorecenter_classindex(sc, class)]

"""
    samplelabels(sr::ScoreRepresentatives, class)

Return the sample labels selected as representatives for one class.
"""
samplelabels(sr::ScoreRepresentatives, class) =
    sr.samplelabels[scorerepresentative_classindex(sr, class)]

"""
    scorecenter(sc::ScoreCenters, class)

Return the score-space center for one class.
"""
scorecenter(sc::ScoreCenters, class) = view(sc.centers, scorecenter_classindex(sc, class), :)

"""
    scorecenters(sr::ScoreRepresentatives)

Return the score centers used to select representatives.
"""
scorecenters(sr::ScoreRepresentatives) = sr.centers

"""
    scorecenter(sr::ScoreRepresentatives, class)

Return the score-space center used to select representatives for one class.
"""
scorecenter(sr::ScoreRepresentatives, class) = scorecenter(sr.centers, class)

"""
    representativescores(sr::ScoreRepresentatives, class)

Return the selected representative scores for one class.
"""
representativescores(sr::ScoreRepresentatives, class) =
    sr.scores[scorerepresentative_classindex(sr, class)]

"""
    representativedistances(sr::ScoreRepresentatives, class)

Return the Euclidean distances from representatives to the score-space class center.
"""
representativedistances(sr::ScoreRepresentatives, class) =
    sr.distances[scorerepresentative_classindex(sr, class)]
