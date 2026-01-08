@testset "fisherztrack aggregates correlations per axis₁" begin
    X = reshape(
        Float64[
            1 2 3 4
            2 3 4 5
            3 4 5 6
            4 5 6 7
        ],
        4,
        2,
        2,
    )
    scores = [1.0, 2.0, 3.0, 4.0]

    manual = zeros(2)
    for axis1 = 1:2
        rs = Float64[]
        ws = Float64[]
        for axis2 = 1:2
            push!(rs, CPPLS.robustcor(view(X, :, axis1, axis2), scores))
            push!(ws, CPPLS.mean(view(X, :, axis1, axis2)))
        end
        zs = atanh.(clamp.(rs, nextfloat(-1.0), prevfloat(1.0)))
        manual[axis1] = tanh(sum(ws .* zs) / (sum(ws) + eps(Float64)))
    end

    result = CPPLS.fisherztrack(X, scores; weights = :mean)
    @test result ≈ manual
end

@testset "robustcor handles constants and finite values" begin
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]
    @test CPPLS.robustcor(x, y) ≈ 1.0

    constant = fill(3.0, 4)
    noisy = [0.5, 0.2, 0.7, 0.6]
    @test CPPLS.robustcor(constant, noisy) == 0.0
end

@testset "separationaxis orients single and multi component projections" begin
    Y = [
        1 0
        0 1
        1 0
        0 1
    ]

    X_single = [4.0; -1.0; 3.0; -2.0]
    direction, scores = CPPLS.separationaxis(reshape(X_single, :, 1), Y)
    @test direction[1] == 1.0 || direction[1] == -1.0
    @test minimum(scores[Y[:, 1] .== 1]) > maximum(scores[Y[:, 2] .== 1])
    flipped_dir, flipped_scores = CPPLS.separationaxis(
        reshape(X_single, :, 1),
        Y;
        positive_class = 2,
    )
    @test flipped_dir == -direction
    @test flipped_scores == -scores

    X_multi = [
        3.0 1.0
        -2.0 0.0
        1.0 3.0
        -1.0 -2.0
    ]
    dir_centroid, scores_centroid =
        CPPLS.separationaxis(X_multi, Y; method = :centroid)
    dir_lda, scores_lda = CPPLS.separationaxis(X_multi, Y; method = :lda)
    @test size(dir_centroid) == (2,)
    @test size(dir_lda) == (2,)
    @test minimum(scores_centroid[Y[:, 1] .== 1]) > maximum(scores_centroid[Y[:, 2] .== 1])
    @test minimum(scores_lda[Y[:, 1] .== 1]) > maximum(scores_lda[Y[:, 2] .== 1])
    @test dir_centroid' * dir_lda ≈ abs(dir_centroid' * dir_lda)
    dir_centroid_neg, scores_centroid_neg = CPPLS.separationaxis(
        X_multi,
        Y;
        method = :centroid,
        positive_class = 2,
    )
    @test dir_centroid_neg ≈ -dir_centroid
    @test scores_centroid_neg ≈ -scores_centroid

    X_equal_means = [
        1.0 2.0
        1.0 2.0
        1.0 2.0
        1.0 2.0
    ]
    @test_throws ArgumentError CPPLS.separationaxis(
        X_equal_means,
        Y;
        method = :centroid,
    )
end

@testset "intervalize builds adjacent intervals" begin
    @test CPPLS.intervalize(0:0.5:1) == [(0.0, 0.5), (0.5, 1.0)]
    @test CPPLS.intervalize([0.2]) == [(0.2, 0.2)]
    @test_throws ArgumentError CPPLS.intervalize(Float64[])
end
