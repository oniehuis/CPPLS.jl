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
    old_require = CPPLS._require_extension_ref[]
    old_plotly = CPPLS._scoreplot_plotly_ref[]
    old_makie = CPPLS._scoreplot_makie_ref[]

    CPPLS._require_extension_ref[] = (extsym::Symbol, pkg::AbstractString) -> nothing
    CPPLS._scoreplot_plotly_ref[] = (samples, groups, scores; kwargs...) ->
        (:plotly, samples, groups, scores, kwargs)
    CPPLS._scoreplot_makie_ref[] = (samples, groups, scores; kwargs...) ->
        (:makie, samples, groups, scores, kwargs)

    try
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
        ncomponents = 2

        B = reshape(collect(1.0:(n_predictors*n_responses*ncomponents)),
            n_predictors, n_responses, ncomponents)
        T = [1.0 2.0; 3.0 4.0]
        P = reshape(collect(1.0:(n_predictors*ncomponents)), n_predictors, ncomponents)
        W_comp = reshape(collect(10.0:(9+n_predictors*ncomponents)),
            n_predictors, ncomponents)
        U = reshape(collect(1.0:(n_samples*ncomponents)), n_samples, ncomponents)
        C = reshape(collect(5.0:(4+n_responses*ncomponents)),
            n_responses, ncomponents)
        R = reshape(collect(2.0:(1+n_predictors*ncomponents)),
            n_predictors, ncomponents)
        Y_hat = reshape(collect(1.0:(n_samples*n_responses*ncomponents)),
            n_samples, n_responses, ncomponents)
        F = reshape(collect(11.0:(10+n_samples*n_responses*ncomponents)),
            n_samples, n_responses, ncomponents)
        X_var = [0.5, 0.25]
        X_var_total = 1.0
        gamma = [0.7, 0.6]
        rho = [0.8, 0.75]
        gammas = reshape([0.7, 0.6], 1, ncomponents)
        rhos = reshape([0.8, 0.75], 1, ncomponents)
        zero_mask = reshape(Bool[false, true, true, false], ncomponents, n_predictors)
        a = reshape(collect(3.0:(2+n_responses*ncomponents)),
            n_responses, ncomponents)
        b = reshape(collect(6.0:(5+n_responses*ncomponents)),
            n_responses, ncomponents)
        W0 = reshape(collect(7.0:(6+n_predictors*n_responses*ncomponents)),
            n_predictors, n_responses, ncomponents)

        samplelabels = [SubString("a", 1:1), SubString("b", 1:1)]
        predictorlabels = ["x1", "x2"]
        responselabels = ["class1", "class2"]
        sampleclasses = categorical(["class1", "class2"])

        # Add dummy values for the new nine fields
        X_z = reshape(collect(1.0:(n_samples*n_predictors)), n_samples, n_predictors)
        X_mean = collect(1.0:n_predictors)
        X_std = collect(1.0:n_predictors)
        Yprim_z = reshape(collect(1.0:(n_samples*n_responses)), n_samples, n_responses)
        Yprim_std = collect(1.0:n_responses)
        cppls = CPPLS.CPPLSFit(
            B,
            T,
            P,
            W_comp,
            U,
            C,
            R,
            Y_hat,
            F,
            X_var,
            X_var_total,
            gamma,
            rho,
            gammas,
            rhos,
            zero_mask,
            a,
            b,
            W0,
            X_mean,
            X_std,
            Yprim_std,
            samplelabels,
            predictorlabels,
            responselabels,
            :discriminant,
            sampleclasses,
        )
        @test res[4] == cppls.T[:, 1:2]

        res = CPPLS.scoreplot(cppls; backend = :plotly)
        @test res[1] == :plotly
        @test res[2] == cppls.samplelabels
        @test res[3] == cppls.sampleclasses
        @test res[4] == cppls.T[:, 1:2]

        cppls_no_groups = CPPLS.CPPLSFit(
            B,
            T,
            P,
            W_comp,
            U,
            C,
            R,
            Y_hat,
            F,
            X_var,
            X_var_total,
            gamma,
            rho,
            gammas,
            rhos,
            zero_mask,
            a,
            b,
            W0,
            X_mean,
            X_std,
            Yprim_std,
            samplelabels,
            predictorlabels,
            String[],
            :regression,
            nothing,
        )

        @test_throws ArgumentError CPPLS.scoreplot(cppls_no_groups; backend = :plotly)

        @test_throws ErrorException CPPLS.scoreplot(cppls; backend = :unknown)
    finally
        CPPLS._require_extension_ref[] = old_require
        CPPLS._scoreplot_plotly_ref[] = old_plotly
        CPPLS._scoreplot_makie_ref[] = old_makie
    end
end
