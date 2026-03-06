@testset "CPPLS stores selected training artefact" begin
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

    for (T, Tmask, dims) in configs
        n_samples = dims.n_samples
        n_predictors = dims.n_predictors
        n_responses = dims.n_responses
        n_components = dims.n_components

        regression_coefficients = reshape(
            T.(1:(n_predictors*n_responses*n_components)),
            n_predictors,
            n_responses,
            n_components,
        )
        X_scores = reshape(T.(1:(n_samples*n_components)), n_samples, n_components)
        X_loadings = reshape(T.(1:(n_predictors*n_components)), n_predictors, n_components)
        X_loading_weights =
            reshape(T.(101:(100+n_predictors*n_components)), n_predictors, n_components)
        Y_scores = reshape(T.(1:(n_samples*n_components)), n_samples, n_components)
        Y_loadings =
            reshape(T.(51:(50+n_responses*n_components)), n_responses, n_components)
        projection =
            reshape(T.(11:(10+n_predictors*n_components)), n_predictors, n_components)
        X_means = reshape(T.(1:n_predictors), 1, n_predictors)
        Y_means = reshape(T.(1:n_responses), 1, n_responses)
        fitted_values = reshape(
            T.(1:(n_samples*n_responses*n_components)),
            n_samples,
            n_responses,
            n_components,
        )
        residuals = reshape(
            T.(401:(400+n_samples*n_responses*n_components)),
            n_samples,
            n_responses,
            n_components,
        )
        X_variance = T.(1:n_components) ./ T(n_components + 1)
        X_total_variance = T(5.0)
        gammas = T.(reverse(1:n_components)) ./ T(n_components + 2)
        canonical_correlations = T.(1:n_components) ./ T(n_components + 3)
        small_norm_indices =
            reshape(Tmask.(0:(n_components*n_predictors-1)), n_components, n_predictors)
        canonical_coefficients =
            reshape(T.(21:(20+n_responses*n_components)), n_responses, n_components)
        canonical_coefficients_y = reshape(
            T.(301:(300+n_responses*n_components)),
            n_responses,
            n_components,
        )
        W0_weights = reshape(
            T.(701:(700+n_predictors*n_responses*n_components)),
            n_predictors,
            n_responses,
            n_components,
        )
        Z = reshape(
            T.(901:(900+n_samples*n_responses*n_components)),
            n_samples,
            n_responses,
            n_components,
        )
        sample_labels = ["sample_$i" for i = 1:n_samples]
        predictor_labels = collect(1:n_predictors)
        response_labels = [Symbol("resp_$i") for i = 1:n_responses]
        da_categories = ["class_$(1 + (i % 2))" for i = 1:n_samples]

        cppls = CPPLS.CPPLS(
            regression_coefficients,
            X_scores,
            X_loadings,
            X_loading_weights,
            Y_scores,
            Y_loadings,
            projection,
            X_means,
            Y_means,
            fitted_values,
            residuals,
            X_variance,
            X_total_variance,
            gammas,
            canonical_correlations,
            small_norm_indices,
            canonical_coefficients,
            canonical_coefficients_y,
            W0_weights,
            Z;
            sample_labels = sample_labels,
            predictor_labels = predictor_labels,
            response_labels = response_labels,
            analysis_mode = :regression,
            da_categories = nothing,
        )

        @test cppls isa CPPLS.AbstractCPPLS
        @test cppls isa CPPLS.CPPLS{
            T,
            Tmask,
            typeof(sample_labels),
            typeof(predictor_labels),
            typeof(response_labels),
            Nothing,
        }
        @test cppls.regression_coefficients === regression_coefficients
        @test cppls.X_scores === X_scores
        @test cppls.X_loadings === X_loadings
        @test cppls.X_loading_weights === X_loading_weights
        @test cppls.Y_scores === Y_scores
        @test cppls.Y_loadings === Y_loadings
        @test cppls.projection === projection
        @test cppls.X_means === X_means
        @test cppls.Y_means === Y_means
        @test cppls.fitted_values === fitted_values
        @test cppls.residuals === residuals
        @test cppls.X_variance === X_variance
        @test cppls.X_total_variance === X_total_variance
        @test cppls.gammas === gammas
        @test cppls.canonical_correlations === canonical_correlations
        @test cppls.small_norm_indices === small_norm_indices
        @test cppls.canonical_coefficients === canonical_coefficients
        @test cppls.canonical_coefficients_y === canonical_coefficients_y
        @test cppls.W0_weights === W0_weights
        @test cppls.Z === Z
        @test cppls.sample_labels === sample_labels
        @test cppls.predictor_labels === predictor_labels
        @test cppls.response_labels === response_labels
        @test cppls.analysis_mode === :regression
        @test cppls.da_categories === nothing
        @test size(cppls.regression_coefficients) ==
              (n_predictors, n_responses, n_components)
        @test size(cppls.fitted_values) == (n_samples, n_responses, n_components)
        @test size(cppls.residuals) == (n_samples, n_responses, n_components)
        @test size(cppls.X_scores) == (n_samples, n_components)
        @test size(cppls.Y_scores) == (n_samples, n_components)
        @test size(cppls.X_means) == (1, n_predictors)
        @test size(cppls.Y_means) == (1, n_responses)
        @test size(cppls.Z) == (n_samples, n_responses, n_components)

        cppls_default = CPPLS.CPPLS(
            regression_coefficients,
            X_scores,
            X_loadings,
            X_loading_weights,
            Y_scores,
            Y_loadings,
            projection,
            X_means,
            Y_means,
            fitted_values,
            residuals,
            X_variance,
            X_total_variance,
            gammas,
            canonical_correlations,
            small_norm_indices,
            canonical_coefficients,
            canonical_coefficients_y,
            W0_weights,
            Z,
        )
        @test isempty(cppls_default.sample_labels)
        @test isempty(cppls_default.predictor_labels)
        @test isempty(cppls_default.response_labels)
        @test cppls_default.analysis_mode === :regression
        @test cppls_default.da_categories === nothing

        cppls_da = CPPLS.CPPLS(
            regression_coefficients,
            X_scores,
            X_loadings,
            X_loading_weights,
            Y_scores,
            Y_loadings,
            projection,
            X_means,
            Y_means,
            fitted_values,
            residuals,
            X_variance,
            X_total_variance,
            gammas,
            canonical_correlations,
            small_norm_indices,
            canonical_coefficients,
            canonical_coefficients_y,
            W0_weights,
            Z;
            sample_labels = sample_labels,
            predictor_labels = predictor_labels,
            response_labels = response_labels,
            analysis_mode = :discriminant,
            da_categories = da_categories,
        )
        @test cppls_da.analysis_mode === :discriminant
        @test cppls_da.da_categories === da_categories
    end
end

@testset "CPPLSLight keeps prediction essentials" begin
    configs = [
        (Float32, (; n_predictors = 3, n_responses = 2, n_components = 1)),
        (Float64, (; n_predictors = 4, n_responses = 3, n_components = 2)),
    ]

    for (T, dims) in configs
        n_predictors = dims.n_predictors
        n_responses = dims.n_responses
        n_components = dims.n_components

        regression_coefficients = reshape(
            T.(1:(n_predictors*n_responses*n_components)),
            n_predictors,
            n_responses,
            n_components,
        )
        X_means = reshape(T.(1:n_predictors), 1, n_predictors)
        Y_means = reshape(T.(1:n_responses) .+ T(100), 1, n_responses)

        light_model = CPPLSLight(regression_coefficients, X_means, Y_means, :regression)

        @test light_model isa CPPLS.AbstractCPPLS
        @test light_model isa CPPLSLight{T}
        @test light_model.regression_coefficients === regression_coefficients
        @test light_model.X_means === X_means
        @test light_model.Y_means === Y_means
        @test light_model.analysis_mode === :regression
        @test size(light_model.regression_coefficients) ==
              (n_predictors, n_responses, n_components)
        @test size(light_model.X_means) == (1, n_predictors)
        @test size(light_model.Y_means) == (1, n_responses)
        light_da = CPPLSLight(regression_coefficients, X_means, Y_means, :discriminant)
        @test light_da.analysis_mode === :discriminant
    end
end
