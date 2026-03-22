"""
    fit_cppls_light(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real},
        ncomponents::Int;
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
        mode::Symbol=:regression
    )

Low-level CPPLS fitting routine used by internal cross-validation helpers that returns
a `CPPLSFitLight`. Prefer `fit` with a CPPLSModel for the public entry point and full
parameter documentation.
"""
function fit_cppls_light_core(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real},
    ncomponents::Int=2;
    gamma::T1=0.5,
    obs_weights::T2=nothing,
    Y_aux::T3=nothing,
    center::Bool=true,
    X_tolerance::T4=1e-12,
    X_loading_weight_tolerance::T5=eps(Float64),
    gamma_rel_tol::T6=1e-6,
    gamma_abs_tol::T7=1e-12,
    t_squared_norm_tolerance::T8=1e-10,
    mode::Symbol=:regression,
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
        cppls_prepare_data(m, X, Y_prim, Y_aux, obs_weights)

    for i = 1:m.ncomponents
        wᵢ, _, _, _, _, _, _, _ = compute_cppls_weights(m, X_def, Y, Y_prim, obs_weights,
            m.gamma)

        process_component!(m, i, X_def, wᵢ, Y_prim, W_comp, P, C, B, zero_mask)
    end

    CPPLSFitLight(B, X_bar, Y_bar, m.mode)
end


function fit_cppls_light(
    model::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Y_aux::T2=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    fit_cppls_light_core(
        model,
        X, 
        Y_prim, 
        ncomponents(model); 
        obs_weights=obs_weights, 
        Y_aux=Y_aux,
        cppls_model_fit_kwargs_with_mode(model)...
    )
end

function fit_cppls_light(
    model::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real};
    obs_weights::T1=nothing,
    Y_aux::T2=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}

    Y_matrix = reshape(Y_prim, :, 1)

    fit_cppls_light_core(
        model,
        X, 
        Y_matrix, 
        ncomponents(model); 
        cppls_model_fit_kwargs(model)..., 
        mode=:regression,
        obs_weights=obs_weights, 
        Y_aux=Y_aux
    )
end

function fit_cppls_light(
    model::CPPLSModel,
    X::AbstractMatrix{<:Real},
    sampleclasses::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...
) where {T,R,V,C,U}
    
    mode(model) ≡ :discriminant || throw(ArgumentError(
        "CPPLSModel must use mode=:discriminant when fitting from sampleclasses."))
    
    fit_cppls_light_from_sample_classes(
        model,
        X, 
        sampleclasses, 
        model.ncomponents;
        cppls_model_fit_kwargs(model)..., 
        kwargs...
    )
end

function fit_cppls_light(
    model::CPPLSModel,
    X::AbstractMatrix{<:Real},
    sampleclasses::AbstractVector;
    kwargs...,
)
    mode(model) ≡ :discriminant || throw(ArgumentError(
        "CPPLSModel must use mode=:discriminant when fitting from sampleclasses."))
    
    fit_cppls_light_from_sample_classes(
        model,
        X, 
        sampleclasses, 
        ncomponents(model);
        cppls_model_fit_kwargs(model)..., 
        kwargs...)
end

function fit_cppls_light_from_sample_classes(
    model::CPPLSModel,
    X::AbstractMatrix{<:Real},
    sampleclasses,
    ncomponents::Int;
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
    Y_prim, _ = onehot(sampleclasses)

    fit_cppls_light_core(model, X, Y_prim, ncomponents; gamma=gamma, obs_weights=obs_weights,
        Y_aux=Y_aux, center=center, X_tolerance=X_tolerance, 
        X_loading_weight_tolerance=X_loading_weight_tolerance, gamma_rel_tol=gamma_rel_tol,
        gamma_abs_tol=gamma_abs_tol, t_squared_norm_tolerance=t_squared_norm_tolerance,
        mode=:discriminant)
end
