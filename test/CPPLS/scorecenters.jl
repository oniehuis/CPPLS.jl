function mock_scorecenter_fit(; with_classes::Bool=true)
    n_samples = 5
    n_predictors = 2
    n_responses = 1
    ncomponents = 2

    CPPLS.CPPLSFit(
        reshape(collect(1.0:(n_predictors*n_responses*ncomponents)),
            n_predictors, n_responses, ncomponents),
        [1.0 0.0; 2.0 0.0; 4.0 0.0; 9.0 1.0; 11.0 1.0],
        reshape(collect(1.0:(n_predictors*ncomponents)), n_predictors, ncomponents),
        reshape(collect(11.0:(10+n_predictors*ncomponents)), n_predictors, ncomponents),
        reshape(collect(1.0:(n_samples*ncomponents)), n_samples, ncomponents),
        reshape(collect(1.0:(n_responses*ncomponents)), n_responses, ncomponents),
        reshape(collect(21.0:(20+n_predictors*ncomponents)), n_predictors, ncomponents),
        reshape(collect(1.0:(n_samples*n_responses*ncomponents)),
            n_samples, n_responses, ncomponents),
        reshape(collect(31.0:(30+n_samples*n_responses*ncomponents)),
            n_samples, n_responses, ncomponents),
        [0.5, 0.25],
        0.75,
        [0.4, 0.3],
        [0.9, 0.8],
        reshape([0.4, 0.3], 1, ncomponents),
        reshape([0.9, 0.8], 1, ncomponents),
        fill(false, ncomponents, n_predictors),
        reshape(collect(1.0:(n_responses*ncomponents)), n_responses, ncomponents),
        reshape(collect(4.0:(3+n_responses*ncomponents)), n_responses, ncomponents),
        reshape(collect(1.0:(n_predictors*n_responses*ncomponents)),
            n_predictors, n_responses, ncomponents),
        [1.0, 2.0],
        [2.0, 4.0],
        [1.0];
        samplelabels = ["s1", "s2", "s3", "s4", "s5"],
        predictorlabels = ["x1", "x2"],
        responselabels = ["y"],
        analysis_mode = :discriminant,
        sampleclasses = with_classes ? ["A", "A", "A", "B", "B"] : nothing,
    )
end

@testset "scorecenters from score matrix" begin
    scores = [
        1.0 10.0 100.0
        3.0 14.0 140.0
        9.0 18.0 180.0
        11.0 22.0 220.0
        13.0 26.0 260.0
    ]
    classes = ["A", "A", "B", "B", "B"]
    labels = ["s1", "s2", "s3", "s4", "s5"]

    sc = CPPLS.scorecenters(scores, classes; samplelabels = labels, comps = 1:2)

    @test sc isa CPPLS.ScoreCenters
    @test sc.classes == ["A", "B"]
    @test sc.comps == [1, 2]
    @test sc.center_method == :median
    @test CPPLS.sampleindices(sc, "A") == [1, 2]
    @test CPPLS.sampleindices(sc, "B") == [3, 4, 5]
    @test CPPLS.samplelabels(sc, "A") == ["s1", "s2"]
    @test CPPLS.scorecenter(sc, "A") == [2.0, 12.0]
    @test CPPLS.scorecenter(sc, "B") == [11.0, 22.0]
    @test_throws KeyError CPPLS.scorecenter(sc, "C")
end

@testset "scorecenters supports mean and selected components" begin
    scores = [
        1.0 10.0 100.0
        3.0 14.0 140.0
        9.0 18.0 180.0
        11.0 22.0 220.0
    ]
    classes = [:A, :A, :B, :B]

    sc = CPPLS.scorecenters(scores, classes; comps = [3, 1], center = :mean)

    @test sc.classes == [:A, :B]
    @test sc.comps == [3, 1]
    @test CPPLS.scorecenter(sc, :A) == [120.0, 2.0]
    @test CPPLS.scorecenter(sc, "B") == [200.0, 10.0]
end

@testset "scorecenters accepts projected scores and subset components" begin
    projected_scores = [
        1.0 100.0 10.0
        3.0 300.0 14.0
        9.0 900.0 18.0
        11.0 1100.0 22.0
    ]
    classes = ["A", "A", "B", "B"]
    labels = ["new1", "new2", "new3", "new4"]

    sc = CPPLS.scorecenters(projected_scores, classes;
        samplelabels = labels,
        comps = [1, 3],
    )

    @test sc.comps == [1, 3]
    @test CPPLS.samplelabels(sc, "A") == ["new1", "new2"]
    @test CPPLS.scorecenter(sc, "A") == [2.0, 12.0]
    @test CPPLS.scorecenter(sc, "B") == [10.0, 20.0]
end

@testset "scorecenters from CPPLSFit" begin
    fit = mock_scorecenter_fit()

    sc = CPPLS.scorecenters(fit; comps = 2, center = :mean)

    @test CPPLS.ncomponents(fit) == 2
    @test sc.classes == ["A", "B"]
    @test sc.comps == [2]
    @test CPPLS.sampleindices(sc, "A") == [1, 2, 3]
    @test CPPLS.samplelabels(sc, "B") == ["s4", "s5"]
    @test CPPLS.scorecenter(sc, "A") == [0.0]
    @test CPPLS.scorecenter(sc, "B") == [1.0]
end

@testset "scorecenters validates inputs" begin
    scores = [1.0 2.0; 3.0 4.0]

    @test_throws ArgumentError CPPLS.scorecenters(scores, ["A"])
    @test_throws ArgumentError CPPLS.scorecenters(scores, ["A", "B"]; samplelabels = ["s1"])
    @test_throws ArgumentError CPPLS.scorecenters(scores, ["A", "B"]; comps = Int[])
    @test_throws ArgumentError CPPLS.scorecenters(scores, ["A", "B"]; comps = 3)
    @test_throws ArgumentError CPPLS.scorecenters(scores, ["A", "B"]; center = :mode)
    @test_throws ArgumentError CPPLS.scorecenters(mock_scorecenter_fit(with_classes = false))
end

@testset "scorerepresentatives from score matrix" begin
    scores = [
        1.0 10.0
        2.0 11.0
        4.0 14.0
        8.0 20.0
        10.0 22.0
        13.0 24.0
    ]
    classes = ["A", "A", "A", "B", "B", "B"]
    labels = ["s1", "s2", "s3", "s4", "s5", "s6"]

    sr = CPPLS.scorerepresentatives(scores, classes;
        samplelabels = labels,
        comps = 1:2,
        n = 2,
    )

    @test sr isa CPPLS.ScoreRepresentatives
    @test sr.classes == ["A", "B"]
    @test sr.comps == [1, 2]
    @test sr.center_method == :median
    @test CPPLS.sampleindices(sr, "A") == [2, 1]
    @test CPPLS.samplelabels(sr, "A") == ["s2", "s1"]
    @test CPPLS.representativescores(sr, "A") == [2.0 11.0; 1.0 10.0]
    @test CPPLS.representativedistances(sr, "A") ≈ [0.0, sqrt(2.0)]
    @test CPPLS.scorecenters(sr) === sr.centers
    @test CPPLS.scorecenter(sr, "B") == [10.0, 22.0]
    @test CPPLS.sampleindices(sr, "B") == [5, 4]
end

@testset "scorerepresentatives accepts projected scores and subset components" begin
    projected_scores = [
        1.0 100.0 10.0
        2.0 200.0 11.0
        4.0 400.0 14.0
        8.0 800.0 20.0
        10.0 1000.0 22.0
        13.0 1300.0 24.0
    ]
    classes = ["A", "A", "A", "B", "B", "B"]
    labels = ["new1", "new2", "new3", "new4", "new5", "new6"]

    sr = CPPLS.scorerepresentatives(projected_scores, classes;
        samplelabels = labels,
        comps = [1, 3],
        n = 1,
    )

    @test sr.comps == [1, 3]
    @test CPPLS.sampleindices(sr, "A") == [2]
    @test CPPLS.samplelabels(sr, "A") == ["new2"]
    @test CPPLS.representativescores(sr, "B") == reshape([10.0, 22.0], 1, 2)
    @test CPPLS.representativedistances(sr, "B") == [0.0]
end

@testset "scorerepresentatives from CPPLSFit" begin
    fit = mock_scorecenter_fit()

    sr = CPPLS.scorerepresentatives(fit; comps = 1, center = :mean)

    @test sr.classes == ["A", "B"]
    @test sr.comps == [1]
    @test CPPLS.sampleindices(sr, "A") == [2]
    @test CPPLS.samplelabels(sr, "B") == ["s4"]
    @test CPPLS.representativescores(sr, "A") == reshape([2.0], 1, 1)
    @test CPPLS.representativedistances(sr, "B") == [1.0]
end

@testset "scorerepresentatives validates inputs" begin
    scores = [1.0 2.0; 3.0 4.0]

    @test_throws ArgumentError CPPLS.scorerepresentatives(scores, ["A", "B"]; n = 0)
    @test_throws ArgumentError CPPLS.scorerepresentatives(scores, ["A"]; n = 1)
    @test_throws ArgumentError CPPLS.scorerepresentatives(scores, ["A", "B"]; center = :mode)
    @test_throws ArgumentError CPPLS.scorerepresentatives(
        mock_scorecenter_fit(with_classes = false))
end
