@testset "centerweight applies weighted centering and scaling" begin
    M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    weights = [1.0, 2.0, 1.0]

    weighted_cs = CPPLS.centerweight(M, weights)
    unweighted_cs = CPPLS.centerweight(M, nothing)

    expected_weighted = (M .- (weights' * M) / sum(weights)) .* weights
    expected_unweighted = M .- CPPLS.mean(M, dims = 1)

    @test weighted_cs == expected_weighted
    @test unweighted_cs == expected_unweighted
end

@testset "float64 preserves Float64 and converts other types" begin
    float_input = rand(3, 3)
    @test CPPLS.float64(float_input) === float_input

    int_input = [1 2; 3 4]
    converted = CPPLS.float64(int_input)
    @test converted isa Matrix{Float64}
    @test converted == float.(int_input)
end

@testset "float64 handles vectors" begin
    float_vec = rand(5)
    @test CPPLS.float64(float_vec) === float_vec

    int_vec = [1, 2, 3]
    converted_vec = CPPLS.float64(int_vec)
    @test converted_vec isa Vector{Float64}
    @test converted_vec == float.(int_vec)
end

@testset "preprocess validates shapes and returns deflated matrices" begin
    X = Float32[1 2; 3 4; 5 6; 7 8]
    Y_prim = Float32[1 0; 0 1; 1 0; 0 1]
    Y_aux = Float32[0.1 0.2; 0.2 0.3; 0.3 0.4; 0.4 0.5]
    weights = Float32[1, 2, 1, 2]

    model = CPPLS.CPPLSModel(ncomponents=2, gamma=0.5, center_Y=false)  

    d = CPPLS.preprocess(model, X, Y_prim, Y_aux, weights)

    X_prep      = d.X
    Y_prim_prep = d.Yprim
    X_mean      = d.X_mean
    Y_mean      = d.Yprim_mean
    Y_all       = d.Y
    X_def       = d.X_def
    W_comp      = d.W_comp
    P           = d.P
    C           = d.C
    zero_mask   = d.zero_mask
    B           = d.B
    n_samples   = d.n_samples_X
    n_targets   = d.n_targets_Y

    @test X_prep isa Matrix{Float64}
    @test Y_prim_prep isa Matrix{Float64}
    # @test Y_all == hcat(Y_prim_prep, Float64.(Y_aux))
    @test length(X_mean) == size(X, 2)
    @test length(Y_mean) == size(Y_prim, 2)
    @test size(X_def) == size(X_prep)
    @test size(W_comp) == (size(X, 2), 2)
    @test size(P) == (size(X, 2), 2)
    @test size(C) == (size(Y_prim, 2), 2)
    @test size(zero_mask) == (2, size(X, 2))
    @test size(B) == (size(X, 2), size(Y_prim, 2), 2)
    @test n_samples == size(X, 1)
    @test n_targets == size(Y_prim, 2)

    @test_throws DimensionMismatch CPPLS.preprocess(
        model,
        X,
        Y_prim[1:3, :],
        nothing,
        nothing
    )
    @test_throws DimensionMismatch CPPLS.preprocess(
        model,
        X,
        Y_prim,
        nothing,
        [1, 2, 3]
    )
end
