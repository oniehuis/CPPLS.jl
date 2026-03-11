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

@testset "invfreqweights returns normalized inverse frequencies" begin
    samples = ["A", "A", "B", "C", "C", "C"]
    w = CPPLS.invfreqweights(samples)

    @test length(w) == length(samples)
    @test sum(w) ≈ 1.0
    @test w[1] == w[2]

    # Rare groups receive higher per-sample weights.
    @test w[3] > w[1] > w[4]

    # Total weight per class is balanced.
    @test sum(w[[1, 2]]) ≈ sum(w[[3]])
    @test sum(w[[3]]) ≈ sum(w[[4, 5, 6]])
end

@testset "intervalize builds adjacent intervals" begin
    @test CPPLS.intervalize(0:0.5:1) == [(0.0, 0.5), (0.5, 1.0)]
    @test CPPLS.intervalize([0.2]) == [(0.2, 0.2)]
    @test_throws ArgumentError CPPLS.intervalize(Float64[])
end
