@testset "labels_to_one_hot with explicit indices" begin
    label_indices = [1, 3, 2, 3, 1]
    n_labels = 3
    expected = [
        1 0 0
        0 0 1
        0 1 0
        0 0 1
        1 0 0
    ]

    encoded = CPPLS.labels_to_one_hot(label_indices, n_labels)

    @test encoded == expected
end

@testset "labels_to_one_hot discovers label order" begin
    raw_labels = ["cat", "dog", "cat", "owl", "dog"]
    encoded, uniques = CPPLS.labels_to_one_hot(raw_labels)

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

@testset "one_hot_to_labels decodes argmax positions" begin
    one_hot = [
        0 1 0
        1 0 0
        0 0 1
        0 1 0
    ]

    decoded = CPPLS.one_hot_to_labels(one_hot)
    @test decoded == [2, 1, 3, 2]
end
