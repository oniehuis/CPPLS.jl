@testset "CPPLSFit stores selected training artefact" begin
    configs = [
        (
            Float32,
            Int8,
            (; n_samples = 5, n_predictors = 4, n_responses = 3, n_components = 2),
        ),
        (
            Float64,
            Int16,
            (; n_samples = 3, n_predictors = 2, n_responses = 2, n_components = 1),
        ),
    ]

    for (T_el, Tmask, dims) in configs
        n_samples = dims.n_samples
        n_predictors = dims.n_predictors
        n_responses = dims.n_responses
        n_components = dims.n_components

        B = reshape(
            T_el.(1:(n_predictors*n_responses*n_components)),
            n_predictors,
            n_responses,
            n_components,
        )
        T_scores = reshape(T_el.(1:(n_samples*n_components)), n_samples, n_components)
        P = reshape(T_el.(1:(n_predictors*n_components)), n_predictors, n_components)
        W_comp =
            reshape(T_el.(101:(100+n_predictors*n_components)), n_predictors, n_components)
        U = reshape(T_el.(1:(n_samples*n_components)), n_samples, n_components)
        C =
            reshape(T_el.(51:(50+n_responses*n_components)), n_responses, n_components)
        R =
            reshape(T_el.(11:(10+n_predictors*n_components)), n_predictors, n_components)
        X_bar = reshape(T_el.(1:n_predictors), 1, n_predictors)
        Y_bar = reshape(T_el.(1:n_responses), 1, n_responses)
        Y_hat = reshape(
            T_el.(1:(n_samples*n_responses*n_components)),
            n_samples,
            n_responses,
            n_components,
        )
        F = reshape(
            T_el.(401:(400+n_samples*n_responses*n_components)),
            n_samples,
            n_responses,
            n_components,
        )
        X_var = T_el.(1:n_components) ./ T_el(n_components + 1)
        X_var_total = T_el(5.0)
        gamma = T_el.(reverse(1:n_components)) ./ T_el(n_components + 2)
        rho = T_el.(1:n_components) ./ T_el(n_components + 3)
        zero_mask =
            reshape(Tmask.(0:(n_components*n_predictors-1)), n_components, n_predictors)
        a =
            reshape(T_el.(21:(20+n_responses*n_components)), n_responses, n_components)
        b = reshape(
            T_el.(301:(300+n_responses*n_components)),
            n_responses,
            n_components,
        )
        W0 = reshape(
            T_el.(701:(700+n_predictors*n_responses*n_components)),
            n_predictors,
            n_responses,
            n_components,
        )
        Z = reshape(
            T_el.(901:(900+n_samples*n_responses*n_components)),
            n_samples,
            n_responses,
            n_components,
        )
        sample_labels = ["sample_$i" for i = 1:n_samples]
        predictor_labels = collect(1:n_predictors)
        response_labels = [Symbol("resp_$i") for i = 1:n_responses]
        sample_classes = ["class_$(1 + (i % 2))" for i = 1:n_samples]

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
            zero_mask,
            a,
            b,
            W0,
            Z;
            sample_labels = sample_labels,
            predictor_labels = predictor_labels,
            response_labels = response_labels,
            analysis_mode = :regression,
            sample_classes = nothing,
        )

        @test cppls isa CPPLS.AbstractCPPLSFit
        @test cppls isa CPPLS.CPPLSFit{
            T_el,
            Tmask,
            typeof(sample_labels),
            typeof(predictor_labels),
            typeof(response_labels),
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
        @test cppls.sample_labels === sample_labels
        @test cppls.predictor_labels === predictor_labels
        @test cppls.response_labels === response_labels
        @test cppls.analysis_mode === :regression
        @test cppls.sample_classes === nothing
        @test size(cppls.B) ==
              (n_predictors, n_responses, n_components)
        @test size(cppls.Y_hat) == (n_samples, n_responses, n_components)
        @test size(cppls.F) == (n_samples, n_responses, n_components)
        @test size(cppls.T) == (n_samples, n_components)
        @test size(cppls.U) == (n_samples, n_components)
        @test size(cppls.X_bar) == (1, n_predictors)
        @test size(cppls.Y_bar) == (1, n_responses)
        @test size(cppls.Z) == (n_samples, n_responses, n_components)

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
            zero_mask,
            a,
            b,
            W0,
            Z,
        )
        @test isempty(cppls_default.sample_labels)
        @test isempty(cppls_default.predictor_labels)
        @test isempty(cppls_default.response_labels)
        @test cppls_default.analysis_mode === :regression
        @test cppls_default.sample_classes === nothing

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
        @test cppls_da.analysis_mode === :discriminant
        @test cppls_da.sample_classes === sample_classes
    end
end

@testset "CPPLSFitLight keeps prediction essentials" begin
    configs = [
        (Float32, (; n_predictors = 3, n_responses = 2, n_components = 1)),
        (Float64, (; n_predictors = 4, n_responses = 3, n_components = 2)),
    ]

    for (T, dims) in configs
        n_predictors = dims.n_predictors
        n_responses = dims.n_responses
        n_components = dims.n_components

        B = reshape(
            T.(1:(n_predictors*n_responses*n_components)),
            n_predictors,
            n_responses,
            n_components,
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
        @test light_model.analysis_mode === :regression
        @test size(light_model.B) ==
              (n_predictors, n_responses, n_components)
        @test size(light_model.X_bar) == (1, n_predictors)
        @test size(light_model.Y_bar) == (1, n_responses)
        light_da = CPPLSFitLight(B, X_bar, Y_bar, :discriminant)
        @test light_da.analysis_mode === :discriminant
    end
end

@testset "CPPLSSpec stores hyperparameters" begin
    spec = CPPLS.CPPLSSpec()
    @test spec.n_components == 2
    @test spec.gamma == 0.5
    @test spec.center === true
    @test spec.analysis_mode === :regression

    tuned = CPPLS.CPPLSSpec(
        n_components = 3,
        gamma = (0.2, 0.8),
        center = false,
        X_tolerance = 1e-8,
        X_loading_weight_tolerance = 1e-9,
        t_squared_norm_tolerance = 1e-7,
        gamma_rel_tol = 1e-5,
        gamma_abs_tol = 1e-9,
        analysis_mode = :discriminant,
    )
    @test tuned.n_components == 3
    @test tuned.gamma == (0.2, 0.8)
    @test tuned.center === false
    @test tuned.analysis_mode === :discriminant

    @test_throws ArgumentError CPPLS.CPPLSSpec(n_components = 0)
    @test_throws ArgumentError CPPLS.CPPLSSpec(analysis_mode = :unsupported)
end

@testset "custom show methods summarize CPPLS types" begin
    spec = CPPLS.CPPLSSpec(n_components = 3, gamma = (0.2, 0.8), analysis_mode = :discriminant)
    spec_inline = sprint(show, spec)
    spec_plain = sprint(io -> show(io, MIME"text/plain"(), spec))
    @test occursin("CPPLSSpec(", spec_inline)
    @test occursin("n_components=3", spec_inline)
    @test occursin("analysis_mode=discriminant", spec_inline)
    @test occursin("CPPLSSpec", spec_plain)
    @test occursin("n_components: 3", spec_plain)
    @test occursin("analysis_mode: discriminant", spec_plain)

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
    zero_mask = zeros(Int, 2, 2)
    a = reshape(Float64.(91:94), 2, 2)
    b = reshape(Float64.(101:104), 2, 2)
    W0 = reshape(Float64.(111:118), 2, 2, 2)
    Z = reshape(Float64.(121:132), 3, 2, 2)
    sample_labels = String[]
    predictor_labels = String[]
    response_labels = String[]
    sample_classes = nothing

    model = CPPLS.CPPLSFit(
        B, T_scores, P, W_comp, U, C, R, X_bar, Y_bar, Y_hat, F, X_var, X_var_total,
        gamma, rho, zero_mask, a, b, W0, Z, sample_labels, predictor_labels,
        response_labels, :regression, sample_classes,
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
