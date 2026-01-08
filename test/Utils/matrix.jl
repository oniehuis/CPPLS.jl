@testset "find_invariant_and_variant_columns detects constant columns" begin
    M = [
        1 4 7
        1 5 8
        1 6 9
    ]

    invariant, variant = CPPLS.find_invariant_and_variant_columns(M)

    @test invariant == [1]
    @test variant == [2, 3]
end

@testset "find_invariant_and_variant_columns handles mixed precision" begin
    M = [
        0.5 1.0 3.0
        0.5 1.0 3.0
        0.5 2.0 3.1
    ]

    invariant, variant = CPPLS.find_invariant_and_variant_columns(M)

    @test invariant == [1]
    @test variant == [2, 3]
end
