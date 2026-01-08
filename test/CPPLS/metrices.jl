@testset "nmc computes weighted and unweighted error" begin
    Y_true = [
        1 0 0
        0 1 0
        0 1 0
        0 1 0
        0 0 1
    ]
    Y_pred = [
        0 1 0  # misclass class 1 -> 2
        0 1 0  # correct
        0 0 1  # misclass class 2 -> 3
        0 1 0  # correct
        0 0 1  # correct
    ]

    @test CPPLS.nmc(Y_true, Y_pred, false) == 4 / 15

    # weighted version over-penalises the minority class error (class 1)
    weighted = CPPLS.nmc(Y_true, Y_pred, true)
    @test weighted == 4 / 9
end

@testset "nmc validates shapes and zero-length inputs" begin
    Y_true = [1 0; 0 1]
    Y_pred = [1 0; 1 0]

    @test_throws DimensionMismatch CPPLS.nmc(Y_true, Y_pred[1:1, :], false)
    @test_throws ErrorException CPPLS.nmc(
        Y_true[1:0, :],
        Y_pred[1:0, :],
        true,
    )
end

@testset "calculate_p_value counts permutations below threshold" begin
    perms = [0.4, 0.6, 0.5, 0.55]
    model_acc = 0.5

    p_value = CPPLS.calculate_p_value(perms, model_acc)
    @test p_value == 2 / (length(perms) + 1)
end
