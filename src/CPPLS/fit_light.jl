"""
    fit_cppls_light(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real};
        obs_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        Yaux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}=nothing,
        response_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        target_weights::Union{AbstractVector{<:Real}, Nothing}=nothing
    )

Low-level CPPLS fitting routine used by internal cross-validation helpers that returns
a `CPPLSFitLight`. Prefer `fit` with a CPPLSModel for the public entry point and full
parameter documentation. `response_weights` and `target_weights` have the same semantics
as in `fit`: the former weights all response columns when constructing the supervised
projection, and the latter weights only the primary-response block in the later CCA
alignment step.
"""
function fit_cppls_light_core(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Yaux::T2=nothing,
    response_weights::T3=nothing,
    target_weights::T4=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T3<:Union{AbstractVector{<:Real}, Nothing},
    T4<:Union{AbstractVector{<:Real}, Nothing}
}

    d = preprocess(m, X, Y_prim, Yaux, obs_weights, response_weights, target_weights)

    for i = 1:m.ncomponents
        wᵢ = compute_cppls_weights(m, d.X_def, d.Y, d.Yprim, obs_weights,
            d.response_weights, d.target_weights, m.gamma)[1]
        process_component!(m, i, d.X_def, wᵢ, d.Yprim, d.W_comp, d.P, d.C, d.B, 
            d.zero_mask)
    end

    CPPLSFitLight(d.B, vec(d.X_mean), vec(d.X_std), vec(d.Yprim_std), 
        m.mode)
end


function fit_cppls_light(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Yaux::T2=nothing,
    response_weights::T3=nothing,
    target_weights::T4=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T3<:Union{AbstractVector{<:Real}, Nothing},
    T4<:Union{AbstractVector{<:Real}, Nothing}
}
    fit_cppls_light_core(m, X, Y_prim; obs_weights=obs_weights, Yaux=Yaux,
        response_weights=response_weights, target_weights=target_weights)
end

function fit_cppls_light(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real};
    obs_weights::T1=nothing,
    Yaux::T2=nothing,
    response_weights::T3=nothing,
    target_weights::T4=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T3<:Union{AbstractVector{<:Real}, Nothing},
    T4<:Union{AbstractVector{<:Real}, Nothing}
}
    Y_matrix = reshape(Y_prim, :, 1)
    fit_cppls_light_core(m, X, Y_matrix; obs_weights=obs_weights, Yaux=Yaux,
        response_weights=response_weights, target_weights=target_weights)
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
    Yaux::T2=nothing,
    response_weights::T3=nothing,
    target_weights::T4=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T3<:Union{AbstractVector{<:Real}, Nothing},
    T4<:Union{AbstractVector{<:Real}, Nothing}
}
    Y_prim, _ = onehot(sampleclasses)
    fit_cppls_light_core(m, X, Y_prim; obs_weights=obs_weights, Yaux=Yaux,
        response_weights=response_weights, target_weights=target_weights)
end
