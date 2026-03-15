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
