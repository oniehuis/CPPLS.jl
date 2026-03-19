@testset "onehot with explicit indices" begin
    label_indices = [1, 3, 2, 3, 1]
    n_labels = 3
    expected = [
        1 0 0
        0 0 1
        0 1 0
        0 0 1
        1 0 0
    ]

    encoded = CPPLS.onehot(label_indices, n_labels)

    @test encoded == expected
    @test_throws ArgumentError CPPLS.onehot([1, 4], 3)
    @test_throws ArgumentError CPPLS.onehot([0, 1], 3)
    @test_throws ArgumentError CPPLS.onehot([1, 2], -1)
end

@testset "onehot discovers label order" begin
    raw_labels = ["cat", "dog", "cat", "owl", "dog"]
    encoded, uniques = CPPLS.onehot(raw_labels)

    @test uniques == ["cat", "dog", "owl"]
    @test size(encoded) == (length(raw_labels), length(uniques))

    expected = [
        1 0 0
        0 1 0
        1 0 0
        0 0 1
        0 1 0
    ]
    @test encoded == expected
end

@testset "sampleclasses decodes argmax positions" begin
    one_hot = [
        0 1 0
        1 0 0
        0 0 1
        0 1 0
    ]

    decoded = CPPLS.sampleclasses(one_hot)
    @test decoded == [2, 1, 3, 2]
    @test_throws ArgumentError CPPLS.sampleclasses([1 1 0; 0 1 0])
    @test_throws ArgumentError CPPLS.sampleclasses([0 0 0; 0 1 0])
    @test_throws ArgumentError CPPLS.sampleclasses([2 0 0; 0 1 0])
end
