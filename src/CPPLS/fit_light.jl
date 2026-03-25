"""
    fit_cppls_light(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real};
        obs_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        Yaux::Union{LinearAlgebra.AbstractVecOrMat, Nothing}=nothing
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
    Yaux::T2=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}

    d = preprocess(m, X, Y_prim, Yaux, obs_weights)

    for i = 1:m.ncomponents
        wᵢ = compute_cppls_weights(m, d.X_def, d.Y, d.Yprim, obs_weights, m.gamma)[1]
        process_component!(m, i, d.X_def, wᵢ, d.Yprim, d.W_comp, d.P, d.C, d.B, 
            d.zero_mask)
    end

    CPPLSFitLight(d.B, vec(d.X_mean), vec(d.X_std), vec(d.Yprim_mean), vec(d.Yprim_std), 
        m.mode)
end


function fit_cppls_light(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Yaux::T2=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    fit_cppls_light_core(m, X, Y_prim; obs_weights=obs_weights, Yaux=Yaux)
end

function fit_cppls_light(
    m::CPPLSModel,
    X::AbstractMatrix{<:Real},
    Y_prim::AbstractVector{<:Real};
    obs_weights::T1=nothing,
    Yaux::T2=nothing,
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing}, 
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    Y_matrix = reshape(Y_prim, :, 1)
    fit_cppls_light_core(m, X, Y_matrix; obs_weights=obs_weights, Yaux=Yaux)
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
    Yaux::T2=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}
    Y_prim, _ = onehot(sampleclasses)
    fit_cppls_light_core(m, X, Y_prim; obs_weights=obs_weights, Yaux=Yaux)
end
