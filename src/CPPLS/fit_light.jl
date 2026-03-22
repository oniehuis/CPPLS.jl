"""
    fit_cppls_light(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real};
        obs_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        Y_aux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}=nothing
    )

Low-level CPPLS fitting routine used by internal cross-validation helpers that returns
a `CPPLSFitLight`. Prefer `fit` with a CPPLSModel for the public entry point and full
parameter documentation.
"""
function fit_cppls_light_core(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Y_aux::T2=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    X, Y_prim, Y, obs_weights, X_bar, Y_bar, X_def, W_comp, P, C, zero_mask, B, _, _ = 
        cppls_prepare_data(m, X, Y_prim, Y_aux, obs_weights)

    for i = 1:m.ncomponents
        wᵢ, _, _, _, _, _, _, _ = 
            compute_cppls_weights(m, X_def, Y, Y_prim, obs_weights, m.gamma)
        process_component!(m, i, X_def, wᵢ, Y_prim, W_comp, P, C, B, zero_mask)
    end

    CPPLSFitLight(B, X_bar, Y_bar, m.mode)
end


function fit_cppls_light(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Y_aux::T2=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    fit_cppls_light_core(m, X, Y_prim; obs_weights=obs_weights, Y_aux=Y_aux)
end

function fit_cppls_light(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real};
    obs_weights::T1=nothing,
    Y_aux::T2=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    Y_matrix = reshape(Y_prim, :, 1)
    fit_cppls_light_core(m, X, Y_matrix; obs_weights=obs_weights, Y_aux=Y_aux)
end

function fit_cppls_light(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    sampleclasses::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...
) where {T,R,V,C,U}
    
    mode(m) ≡ :discriminant || throw(ArgumentError(
        "CPPLSModel must use mode=:discriminant when fitting from sampleclasses."))
    
    fit_cppls_light_from_sample_classes(m, X, sampleclasses; kwargs...)
end

function fit_cppls_light(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    sampleclasses::AbstractVector;
    kwargs...,
)
    mode(m) ≡ :discriminant || throw(ArgumentError(
        "CPPLSModel must use mode=:discriminant when fitting from sampleclasses."))
    
    fit_cppls_light_from_sample_classes(m, X, sampleclasses; kwargs...)
end

function fit_cppls_light_from_sample_classes(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    sampleclasses;
    obs_weights::T1=nothing,
    Y_aux::T2=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    Y_prim, _ = onehot(sampleclasses)
    fit_cppls_light_core(m, X, Y_prim; obs_weights=obs_weights, Y_aux=Y_aux)
end
