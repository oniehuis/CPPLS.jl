using CategoricalArrays
using Test

@testset "scoreplot dispatch guards" begin
    samples = ["s1", "s2"]
    groups = ["g1", "g2"]
    scores = [1.0 2.0; 3.0 4.0]

    @test_throws ErrorException CPPLS._require_extension(:MissingExtension, "Missing")
    @test_throws ErrorException CPPLS.scoreplot(samples, groups, scores; backend = :unknown)
end

@testset "scoreplot dispatch success" begin
    @eval CPPLS begin
        _require_extension(extsym::Symbol, pkg::String) = nothing
        function scoreplot_plotly(
            samples::AbstractVector{SubString{String}},
            groups,
            scores::AbstractMatrix{<:Real};
            kwargs...,
        )
            return (:plotly, samples, groups, scores, kwargs)
        end
        function scoreplot_makie(
            samples::AbstractVector{SubString{String}},
            groups,
            scores::AbstractMatrix{<:Real};
            kwargs...,
        )
            return (:makie, samples, groups, scores, kwargs)
        end
    end

    override_method = which(CPPLS._require_extension, (Symbol, String))

    samples = [SubString("s1", 1:2), SubString("s2", 1:2)]
    groups = ["g1", "g2"]
    scores = [1.0 2.0; 3.0 4.0]

    res = CPPLS.scoreplot(samples, groups, scores; backend = :plotly)
    @test res[1] == :plotly
    @test res[2] == samples
    @test res[3] == groups
    @test res[4] == scores

    res = CPPLS.scoreplot(samples, groups, scores; backend = :makie)
    @test res[1] == :makie
    @test res[2] == samples
    @test res[3] == groups
    @test res[4] == scores

    n_samples = 2
    n_predictors = 2
    n_responses = 2
    n_components = 2

    B = reshape(collect(1.0:(n_predictors*n_responses*n_components)),
        n_predictors, n_responses, n_components)
    T = reshape(collect(1.0:(n_samples*n_components)), n_samples, n_components)
    P = reshape(collect(1.0:(n_predictors*n_components)), n_predictors, n_components)
    W_comp = reshape(collect(10.0:(9+n_predictors*n_components)),
        n_predictors, n_components)
    U = reshape(collect(1.0:(n_samples*n_components)), n_samples, n_components)
    C = reshape(collect(5.0:(4+n_responses*n_components)),
        n_responses, n_components)
    R = reshape(collect(2.0:(1+n_predictors*n_components)),
        n_predictors, n_components)
    X_bar = reshape(collect(1.0:n_predictors), 1, n_predictors)
    Y_bar = reshape(collect(1.0:n_responses), 1, n_responses)
    Y_hat = reshape(collect(1.0:(n_samples*n_responses*n_components)),
        n_samples, n_responses, n_components)
    F = reshape(collect(11.0:(10+n_samples*n_responses*n_components)),
        n_samples, n_responses, n_components)
    X_var = [0.5, 0.25]
    X_var_total = 1.0
    gamma = [0.7, 0.6]
    rho = [0.8, 0.75]
    zero_mask = reshape(Bool[false, true, true, false], n_components, n_predictors)
    a = reshape(collect(3.0:(2+n_responses*n_components)),
        n_responses, n_components)
    b = reshape(collect(6.0:(5+n_responses*n_components)),
        n_responses, n_components)
    W0 = reshape(collect(7.0:(6+n_predictors*n_responses*n_components)),
        n_predictors, n_responses, n_components)
    Z = reshape(collect(9.0:(8+n_samples*n_responses*n_components)),
        n_samples, n_responses, n_components)

    sample_labels = [SubString("a", 1:1), SubString("b", 1:1)]
    predictor_labels = ["x1", "x2"]
    response_labels = ["class1", "class2"]
    sample_classes = categorical(["class1", "class2"])

    cppls = CPPLS.CPPLSFit(
        B,
        T,
        P,
        W_comp,
        U,
        C,
        R,
        X_bar,
        Y_bar,
        Y_hat,
        F,
        X_var,
        X_var_total,
        gamma,
        rho,
        zero_mask,
        a,
        b,
        W0,
        Z;
        sample_labels = sample_labels,
        predictor_labels = predictor_labels,
        response_labels = response_labels,
        analysis_mode = :discriminant,
        sample_classes = sample_classes,
    )

    res = CPPLS.scoreplot(cppls; backend = :makie)
    @test res[1] == :makie
    @test res[2] == cppls.sample_labels
    @test res[3] == cppls.sample_classes
    @test res[4] == cppls.T[:, 1:2]

    res = CPPLS.scoreplot(cppls; backend = :plotly)
    @test res[1] == :plotly
    @test res[2] == cppls.sample_labels
    @test res[3] == cppls.sample_classes
    @test res[4] == cppls.T[:, 1:2]

    @test_throws ErrorException CPPLS.scoreplot(cppls; backend = :unknown)

    Base.delete_method(override_method)
end
