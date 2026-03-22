@testset "CPPLSFit stores selected training artefact" begin
    configs = [
        (
            Float32,
            Int8,
            (; n_samples = 5, n_predictors = 4, n_responses = 3, ncomponents = 2),
        ),
        (
            Float64,
            Int16,
            (; n_samples = 3, n_predictors = 2, n_responses = 2, ncomponents = 1),
        ),
    ]

    for (T_el, Tmask, dims) in configs
        n_samples = dims.n_samples
        n_predictors = dims.n_predictors
        n_responses = dims.n_responses
        ncomponents = dims.ncomponents

        B = reshape(
            T_el.(1:(n_predictors*n_responses*ncomponents)),
            n_predictors,
            n_responses,
            ncomponents,
        )
        T_scores = reshape(T_el.(1:(n_samples*ncomponents)), n_samples, ncomponents)
        P = reshape(T_el.(1:(n_predictors*ncomponents)), n_predictors, ncomponents)
        W_comp =
            reshape(T_el.(101:(100+n_predictors*ncomponents)), n_predictors, ncomponents)
        U = reshape(T_el.(1:(n_samples*ncomponents)), n_samples, ncomponents)
        C =
            reshape(T_el.(51:(50+n_responses*ncomponents)), n_responses, ncomponents)
        R =
            reshape(T_el.(11:(10+n_predictors*ncomponents)), n_predictors, ncomponents)
        X_bar = reshape(T_el.(1:n_predictors), 1, n_predictors)
        Y_bar = reshape(T_el.(1:n_responses), 1, n_responses)
        Y_hat = reshape(
            T_el.(1:(n_samples*n_responses*ncomponents)),
            n_samples,
            n_responses,
            ncomponents,
        )
        F = reshape(
            T_el.(401:(400+n_samples*n_responses*ncomponents)),
            n_samples,
            n_responses,
            ncomponents,
        )
        X_var = T_el.(1:ncomponents) ./ T_el(ncomponents + 1)
        X_var_total = T_el(5.0)
        gamma = T_el.(reverse(1:ncomponents)) ./ T_el(ncomponents + 2)
        rho = T_el.(1:ncomponents) ./ T_el(ncomponents + 3)
        gammas = reshape(
            T_el.(501:(500+ncomponents)),
            1,
            ncomponents,
        )
        rhos = reshape(
            T_el.(601:(600+ncomponents)),
            1,
            ncomponents,
        )
        zero_mask =
            reshape(Tmask.(0:(ncomponents*n_predictors-1)), ncomponents, n_predictors)
        a =
            reshape(T_el.(21:(20+n_responses*ncomponents)), n_responses, ncomponents)
        b = reshape(
            T_el.(301:(300+n_responses*ncomponents)),
            n_responses,
            ncomponents,
        )
        W0 = reshape(
            T_el.(701:(700+n_predictors*n_responses*ncomponents)),
            n_predictors,
            n_responses,
            ncomponents,
        )
        Z = reshape(
            T_el.(901:(900+n_samples*n_responses*ncomponents)),
            n_samples,
            n_responses,
            ncomponents,
        )
        samplelabels = ["sample_$i" for i = 1:n_samples]
        predictorlabels = collect(1:n_predictors)
        responselabels = [Symbol("resp_$i") for i = 1:n_responses]
        sampleclasses = ["class_$(1 + (i % 2))" for i = 1:n_samples]

        cppls = CPPLS.CPPLSFit(
            B,
            T_scores,
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
            gammas,
            rhos,
            zero_mask,
            a,
            b,
            W0,
            Z;
            samplelabels = samplelabels,
            predictorlabels = predictorlabels,
            responselabels = responselabels,
            mode = :regression,
            sampleclasses = nothing,
        )

        @test cppls isa CPPLS.AbstractCPPLSFit
        @test cppls isa CPPLS.CPPLSFit{
            T_el,
            Tmask,
            typeof(samplelabels),
            typeof(predictorlabels),
            typeof(responselabels),
            Nothing,
        }
        @test cppls.B === B
        @test cppls.T === T_scores
        @test cppls.P === P
        @test cppls.W_comp === W_comp
        @test cppls.U === U
        @test cppls.C === C
        @test cppls.R === R
        @test cppls.X_bar === X_bar
        @test cppls.Y_bar === Y_bar
        @test cppls.Y_hat === Y_hat
        @test cppls.F === F
        @test cppls.X_var === X_var
        @test cppls.X_var_total === X_var_total
        @test cppls.gamma === gamma
        @test cppls.rho === rho
        @test cppls.zero_mask === zero_mask
        @test cppls.a === a
        @test cppls.b === b
        @test cppls.W0 === W0
        @test cppls.Z === Z
        @test cppls.samplelabels === samplelabels
        @test cppls.predictorlabels === predictorlabels
        @test cppls.responselabels === responselabels
        @test cppls.mode === :regression
        @test cppls.sampleclasses === nothing
        @test size(cppls.B) ==
              (n_predictors, n_responses, ncomponents)
        @test size(cppls.Y_hat) == (n_samples, n_responses, ncomponents)
        @test size(cppls.F) == (n_samples, n_responses, ncomponents)
        @test size(cppls.T) == (n_samples, ncomponents)
        @test size(cppls.U) == (n_samples, ncomponents)
        @test size(cppls.X_bar) == (1, n_predictors)
        @test size(cppls.Y_bar) == (1, n_responses)
        @test size(cppls.Z) == (n_samples, n_responses, ncomponents)

        cppls_default = CPPLS.CPPLSFit(
            B,
            T_scores,
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
            gammas,
            rhos,
            zero_mask,
            a,
            b,
            W0,
            Z,
        )
        @test isempty(cppls_default.samplelabels)
        @test isempty(cppls_default.predictorlabels)
        @test isempty(cppls_default.responselabels)
        @test cppls_default.mode === :regression
        @test cppls_default.sampleclasses === nothing

        cppls_da = CPPLS.CPPLSFit(
            B,
            T_scores,
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
            gammas,
            rhos,
            zero_mask,
            a,
            b,
            W0,
            Z;
            samplelabels = samplelabels,
            predictorlabels = predictorlabels,
            responselabels = responselabels,
            mode = :discriminant,
            sampleclasses = sampleclasses,
        )
        @test cppls_da.mode === :discriminant
        @test cppls_da.sampleclasses === sampleclasses
    end
end

@testset "CPPLSFitLight keeps prediction essentials" begin
    configs = [
        (Float32, (; n_predictors = 3, n_responses = 2, ncomponents = 1)),
        (Float64, (; n_predictors = 4, n_responses = 3, ncomponents = 2)),
    ]

    for (T, dims) in configs
        n_predictors = dims.n_predictors
        n_responses = dims.n_responses
        ncomponents = dims.ncomponents

        B = reshape(
            T.(1:(n_predictors*n_responses*ncomponents)),
            n_predictors,
            n_responses,
            ncomponents,
        )
        X_bar = reshape(T.(1:n_predictors), 1, n_predictors)
        Y_bar = reshape(T.(1:n_responses) .+ T(100), 1, n_responses)

        light_model =
            CPPLSFitLight(B, X_bar, Y_bar, :regression)

        @test light_model isa CPPLS.AbstractCPPLSFit
        @test light_model isa CPPLSFitLight{T}
        @test light_model.B === B
        @test light_model.X_bar === X_bar
        @test light_model.Y_bar === Y_bar
        @test light_model.mode === :regression
        @test size(light_model.B) ==
              (n_predictors, n_responses, ncomponents)
        @test size(light_model.X_bar) == (1, n_predictors)
        @test size(light_model.Y_bar) == (1, n_responses)
        light_da = CPPLSFitLight(B, X_bar, Y_bar, :discriminant)
        @test light_da.mode === :discriminant
    end
end

@testset "CPPLSModel stores hyperparameters" begin
    spec = CPPLS.CPPLSModel()
    @test spec.ncomponents == 2
    @test spec.gamma == 0.5
    @test spec.center === true
    @test spec.mode === :regression

    tuned = CPPLS.CPPLSModel(
        ncomponents = 3,
        gamma = (0.2, 0.8),
        center = false,
        X_tolerance = 1e-8,
        X_loading_weight_tolerance = 1e-9,
        t_squared_norm_tolerance = 1e-7,
        gamma_rel_tol = 1e-5,
        gamma_abs_tol = 1e-9,
        mode = :discriminant,
    )
    @test tuned.ncomponents == 3
    @test tuned.gamma == (0.2, 0.8)
    @test tuned.center === false
    @test tuned.mode === :discriminant

    @test_throws ArgumentError CPPLS.CPPLSModel(ncomponents = 0)
    @test_throws ArgumentError CPPLS.CPPLSModel(mode = :unsupported)
end

@testset "custom show methods summarize CPPLS types" begin
    spec = CPPLS.CPPLSModel(ncomponents = 3, gamma = (0.2, 0.8), mode = :discriminant)
    spec_inline = sprint(show, spec)
    spec_plain = sprint(io -> show(io, MIME"text/plain"(), spec))
    @test occursin("CPPLSModel(", spec_inline)
    @test occursin("ncomponents=3", spec_inline)
    @test occursin("mode=discriminant", spec_inline)
    @test occursin("CPPLSModel", spec_plain)
    @test occursin("ncomponents: 3", spec_plain)
    @test occursin("mode: discriminant", spec_plain)

    B = reshape(Float64.(1:8), 2, 2, 2)
    T_scores = reshape(Float64.(1:6), 3, 2)
    P = reshape(Float64.(1:4), 2, 2)
    W_comp = reshape(Float64.(11:14), 2, 2)
    U = reshape(Float64.(21:26), 3, 2)
    C = reshape(Float64.(31:34), 2, 2)
    R = reshape(Float64.(41:44), 2, 2)
    X_bar = reshape([0.5, 1.5], 1, :)
    Y_bar = reshape([2.5, 3.5], 1, :)
    Y_hat = reshape(Float64.(51:62), 3, 2, 2)
    F = reshape(Float64.(71:82), 3, 2, 2)
    X_var = [0.1, 0.2]
    X_var_total = 1.0
    gamma = [0.5, 0.5]
    rho = [0.9, 0.8]
    gammas = reshape([0.4, 0.6], 1, 2)
    rhos = reshape([0.81, 0.64], 1, 2)
    zero_mask = zeros(Int, 2, 2)
    a = reshape(Float64.(91:94), 2, 2)
    b = reshape(Float64.(101:104), 2, 2)
    W0 = reshape(Float64.(111:118), 2, 2, 2)
    Z = reshape(Float64.(121:132), 3, 2, 2)
    samplelabels = String[]
    predictorlabels = String[]
    responselabels = String[]
    sampleclasses = nothing

    model = CPPLS.CPPLSFit(
        B, T_scores, P, W_comp, U, C, R, X_bar, Y_bar, Y_hat, F, X_var, X_var_total,
        gamma, rho, gammas, rhos, zero_mask, a, b, W0, Z,
        samplelabels, predictorlabels, responselabels, :regression, sampleclasses,
    )
    model_inline = sprint(show, model)
    model_plain = sprint(io -> show(io, MIME"text/plain"(), model))
    @test occursin("CPPLSFit(", model_inline)
    @test occursin("mode=regression", model_inline)
    @test occursin("gamma=[0.5, 0.5]", model_inline)
    @test occursin("samples=3", model_inline)
    @test occursin("predictors=2", model_inline)
    @test occursin("responses=2", model_inline)
    @test occursin("components=2", model_inline)
    @test occursin("CPPLSFit", model_plain)
    @test occursin("mode: regression", model_plain)
    @test occursin("gamma: [0.5, 0.5]", model_plain)
    @test occursin("samples: 3", model_plain)

    light = CPPLS.CPPLSFitLight(B, X_bar, Y_bar, :discriminant)
    light_inline = sprint(show, light)
    light_plain = sprint(io -> show(io, MIME"text/plain"(), light))
    @test occursin("CPPLSFitLight(", light_inline)
    @test occursin("mode=discriminant", light_inline)
    @test occursin("predictors=2", light_inline)
    @test occursin("responses=2", light_inline)
    @test occursin("components=2", light_inline)
    @test occursin("CPPLSFitLight", light_plain)
    @test occursin("mode: discriminant", light_plain)
end
