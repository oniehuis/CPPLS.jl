"""
    fit_cppls_light(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real},
        n_components::Int;
        gamma::Union{
            <:Real, 
            <:NTuple{2, <:Real}, 
            <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
        }=0.5,
        observation_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        Y_aux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}=nothing,
        center::Bool=true,
        X_tolerance::Real=1e-12,
        X_loading_weight_tolerance::Real=eps(Float64),
        gamma_rel_tol::Real=1e-6,
        gamma_abs_tol::Real=1e-12,
        t_squared_norm_tolerance::Real=1e-10,
        analysis_mode::Symbol=:regression
    )

Low-level CPPLS fitting routine used by internal cross-validation helpers that returns
a `CPPLSFitLight`. Prefer `fit` with a CPPLSSpec for the public entry point and full
parameter documentation.
"""
function fit_cppls_light(
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    n_components::Int=2;
    gamma::T1=0.5,
    obs_weights::T2=nothing,
    Y_aux::T3=nothing,
    center::Bool=true,
    X_tolerance::T4=1e-12,
    X_loading_weight_tolerance::T5=eps(Float64),
    gamma_rel_tol::T6=1e-6,
    gamma_abs_tol::T7=1e-12,
    t_squared_norm_tolerance::T8=1e-10,
    analysis_mode::Symbol=:regression,
) where {
    T1<:Union{
        <:Real, 
        <:NTuple{2, <:Real},
        <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
    },
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing} ,
    T4<:Real,
    T5<:Real,
    T6<:Real,
    T7<:Real,
    T8<:Real
}

    X, Y_prim, Y, obs_weights, X_bar, Y_bar, X_def, W_comp, P, C, zero_mask, B, _, _ = 
        cppls_prepare_data(X, Y_prim, n_components, Y_aux, obs_weights, center)

    for i = 1:n_components
        wᵢ, _, _, _, _, _, _, _ = compute_cppls_weights(X_def, Y, Y_prim, obs_weights,
            gamma, gamma_rel_tol, gamma_abs_tol)

        process_component!(i, X_def, wᵢ, Y_prim, W_comp, P, C, B, zero_mask, X_tolerance,
            X_loading_weight_tolerance, t_squared_norm_tolerance)
    end

    analysis_mode in (:discriminant, :regression) || throw( ArgumentError(
        "analysis_mode must be :discriminant or :regression, got $analysis_mode"))

    CPPLSFitLight(B, X_bar, Y_bar, analysis_mode)
end

"""
    fit_cppls_light(
        X::AbstractMatrix{<:Real}, 
        sample_classes::AbstractCategoricalArray, 
        n_components::Int=2; 
        kwargs...
    )
    
    fit_cppls_light(
        X::AbstractMatrix{<:Real}, 
        sample_classes::AbstractVector,
        n_components:Int=2; 
        kwargs...
    )

Label-based convenience wrappers that convert class labels to a one-hot response matrix
and forward into `fit_cppls_light`. Intended for internal cross-validation; prefer `fit`
for the public entry point and full docs.
"""
function fit_cppls_light(
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractCategoricalArray{T,1,R,V,C,U},
    n_components::Int=2;
    kwargs...
) where {T,R,V,C,U}
    
    fit_cppls_light_from_sample_classes(X, sample_classes, n_components; kwargs...)
end

function fit_cppls_light(
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractVector,
    n_components::Int=2;
    kwargs...
)
    fit_cppls_light_from_sample_classes(X, sample_classes, n_components; kwargs...)
end

function fit_cppls_light(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...
) where {T,R,V,C,U}
    
    analysis_mode(model) ≡ :discriminant || throw(ArgumentError(
        "CPPLSSpec must use analysis_mode=:discriminant when fitting from sample_classes."))
    
    fit_cppls_light_from_sample_classes(X, sample_classes, model.n_components;
        cppls_model_fit_kwargs(model)..., kwargs...)
end

function fit_cppls_light(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    sample_classes::AbstractVector;
    kwargs...,
)
    analysis_mode(model) ≡ :discriminant || throw(ArgumentError(
        "CPPLSSpec must use analysis_mode=:discriminant when fitting from sample_classes."))
    
    fit_cppls_light_from_sample_classes(X, sample_classes, n_components(model);
        cppls_model_fit_kwargs(model)..., kwargs...)
end

function fit_cppls_light_from_sample_classes(
    X::AbstractMatrix{<:Real},
    sample_classes,
    n_components::Int;
    gamma::T1=0.5,
    obs_weights::T2=nothing,
    Y_aux::T3=nothing,
    center::Bool=true,
    X_tolerance::T4=1e-12,
    X_loading_weight_tolerance::T5=eps(Float64),
    gamma_rel_tol::T6=1e-6,
    gamma_abs_tol::T7=1e-12,
    t_squared_norm_tolerance::T8=1e-10,
) where {
    T1<:Union{
        <:Real, 
        <:NTuple{2, <:Real},
        <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
    },
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T4<:Real,
    T5<:Real,
    T6<:Real,
    T7<:Real,
    T8<:Real
}
    Y_prim, _ = labels_to_one_hot(sample_classes)

    fit_cppls_light(X, Y_prim, n_components; gamma=gamma, obs_weights=obs_weights,
        Y_aux=Y_aux, center=center, X_tolerance=X_tolerance, 
        X_loading_weight_tolerance=X_loading_weight_tolerance, gamma_rel_tol=gamma_rel_tol,
        gamma_abs_tol=gamma_abs_tol, t_squared_norm_tolerance=t_squared_norm_tolerance,
        analysis_mode=:discriminant)
end

"""
    fit_cppls_light(
        X::AbstractMatrix{<:Real}, 
        y::AbstractVector{<:Real}, 
        n_components::Int=2;
        kwargs...
    )

Convenience wrapper that reshapes a single response vector to a one-column matrix and
forwards into `fit_cppls_light`. Intended for internal cross-validation; prefer `fit`
for the public entry point and full docs.
"""
function fit_cppls_light(
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real},
    n_components::Int=2;
    gamma::T1=0.5,
    obs_weights::T2=nothing,
    Y_aux::T3=nothing,
    center::Bool=true,
    X_tolerance::T4=1e-12,
    X_loading_weight_tolerance::T5=eps(Float64),
    gamma_rel_tol::T6=1e-6,
    gamma_abs_tol::T7=1e-12,
    t_squared_norm_tolerance::T8=1e-10
) where {
    T1<:Union{
        <:Real, 
        <:NTuple{2, <:Real},
        <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}
    },
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T4<:Real,
    T5<:Real,
    T6<:Real,
    T7<:Real,
    T8<:Real
}
    Y_matrix = reshape(Y_prim, :, 1)

    fit_cppls_light(X, Y_matrix, n_components; gamma=gamma, obs_weights=obs_weights,
        Y_aux=Y_aux, center=center, X_tolerance=X_tolerance, 
        X_loading_weight_tolerance=X_loading_weight_tolerance, gamma_rel_tol=gamma_rel_tol,
        gamma_abs_tol=gamma_abs_tol, t_squared_norm_tolerance=t_squared_norm_tolerance,
        analysis_mode=:regression)
end

function fit_cppls_light(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Y_aux::T2=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    fit_cppls_light(X, Y_prim, n_components(model); 
        cppls_model_fit_kwargs_with_mode(model)..., obs_weights=obs_weights, Y_aux=Y_aux)
end

function fit_cppls_light(
    model::CPPLSSpec,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real};
    obs_weights::T1=nothing,
    Y_aux::T2=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    fit_cppls_light(X, Y_prim, n_components(model);
        cppls_model_fit_kwargs_with_mode(model)..., obs_weights=obs_weights, Y_aux=Y_aux)
end
