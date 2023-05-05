using ChainRulesCore
using LinearAlgebra
using KrylovKit

@doc raw"
    num_grad(f, K::Real; [δ = 1e-5])
return the numerical gradient of `f` at `K` calculated with
`(f(K+δ/2) - f(K-δ/2))/δ`
# example
```jldoctest; setup = :(using TensorNetworkAD)
julia> TensorNetworkAD.num_grad(x -> x * x, 3) ≈ 6
true
```
"
function num_grad(f, K; δ::Real=1e-5)
    if eltype(K) == ComplexF64
        (f(K + δ / 2) - f(K - δ / 2)) / δ + 
            (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
    else
        (f(K + δ / 2) - f(K - δ / 2)) / δ
    end
end


@doc raw"
    num_grad(f, K::AbstractArray; [δ = 1e-5])
    
return the numerical gradient of `f` for each element of `K`.
# example
```jldoctest; setup = :(using TensorNetworkAD, LinearAlgebra)
julia> TensorNetworkAD.num_grad(tr, rand(2,2)) ≈ I
true
```
"
function num_grad(f, a::AbstractArray; δ::Real=1e-5)
    b = Array(copy(a))
    df = map(CartesianIndices(b)) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
        num_grad(foo, b[i], δ=δ)
    end
    return _arraytype(a)(df)
end

# patch since it's currently broken otherwise
function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T, rows, xs...)
    function back(ȳ)
        return NoTangent(), NoTangent(), NoTangent(), permutedims(ȳ)...
    end
    return y, back
end

# improves performance compared to default implementation, also avoids errors
# with some complex arrays
function ChainRulesCore.rrule(::typeof(LinearAlgebra.norm), A::AbstractArray)
    n = norm(A)
    function back(Δ)
        return NoTangent(), Δ .* A ./ (n + eps(0f0)), NoTangent()
    end
    return n, back
end

"""
    function ChainRulesCore.rrule(::typeof(norm_L), Au::AbstractArray{T}, Ad::AbstractArray{T}, L::AbstractArray{T}; kwargs...) where {T}
```
           ┌──       ──┐   
           │     │     │   
dAu  = -   L ────┼──── ξl  
           │     │     │   
           └──   Ad  ──┘   
           ┌──   Au  ──┐ 
           │     │     │ 
dAd  = -   L  ───┼───  ξl
           │     │     │ 
           └──       ──┘ 
```
"""

function ChainRulesCore.rrule(::typeof(norm_L), Au::AbstractArray{T}, Ad::AbstractArray{T}, L::AbstractArray{T}; kwargs...) where {T}
    λl, L = norm_L(Au, Ad, L)
    function back((dλ, dL))
        dL -= Array(ein"ab,ab ->"(conj(L), dL))[] * L
        ξl, info = linsolve(FR -> ein"(be,acb), dce -> ad"(FR,Au,Ad), conj(dL), -λl, 1; maxiter = 100)
        info.converged == 0 && @warn "norm_L ad not converged"
        # errL = ein"abc,cba ->"(L, ξl)[]
        # abs(errL) > 1e-1 && throw("L and ξl aren't orthometric. err = $(errL)")
        dAu = -ein"(ad,dce), be -> acb"(L, Ad, ξl) 
        dAd = -ein"(ad,acb), be -> dce"(L, Au, ξl)
        return NoTangent(), conj(dAu), conj(dAd), NoTangent()...
    end
    return (λl, L), back
end

"""
    ChainRulesCore.rrule(::typeof(leftenv), AL::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
```
           ┌──  AL ──┐       
           │    │    │       
dM   =  -  E ──   ── ξE      
           │    │    │       
           └──  AL ──┘       
           ┌──     ──┐    
           │    │    │    
dAu  =  -  E ── M ── ξE   
           │    │    │    
           └──  Ad ──┘    

           ┌──  Au ──┐        a ────┬──── c 
           │    │    │        │     b     │  
dAu  = -   E ── M ── ξE       ├─ d ─┼─ e ─┤ 
           │    │    │        │     g     │ 
           └──     ──┘        f ────┴──── h 
```
"""
function ChainRulesCore.rrule(::typeof(env_E), Au::AbstractArray{T}, Ad::AbstractArray{T}, M::AbstractArray{T}, E::AbstractArray{T}; kwargs...) where {T}
    λ, E = env_E(Au, Ad, M, E)
    function back((dλ, dE))
        ξE, info = linsolve(Ǝ -> ein"((abc,ceh),dgeb),fgh -> adf"(Au,Ǝ,M,Ad), permutedims(dE, (3, 2, 1)), -λ, 1)
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dAu = -conj!(ein"((adf,fgh),dgeb),ceh -> abc"(E, Ad, M, ξE))
        dM  = -conj!(ein"(adf,fgh),(abc,ceh) -> dgeb"(E, Ad,Au, ξE))
        dAd = -conj!(ein"((adf,abc),dgeb),ceh -> fgh"(E, Au, M, ξE))
        return  NoTangent(), dAu, dAd, dM, NoTangent()...
    end
    return (λ, E), back
end

function ChainRulesCore.rrule(::typeof(C工linear), T, C, Ɔ, Cb)
    x = C工linear(T, C, Ɔ, Cb)
    function back(dx)
        dCb, info = linsolve(x->x - ein"(abc,ce),dbe->ad"(T,x,conj(T)) + ein"(ab,ab),cd->cd"(C,x,Ɔ), conj(dx))
        @assert info.converged == 1 "C工linear not converged"
        dT = conj!(ein"(ab,bcd),ed->ace"(x, conj(T), dCb)) + ein"(ab,acd),de->bce"(x, T, dCb)
        dC = -conj!(ein"(ab,ab),cd->cd"(x, Ɔ, dCb))
        dƆ = -conj!(ein"ab,(cd,cd)->ab"(x, C, dCb))
        return NoTangent(), dT, dC, dƆ, conj(dCb)
    end
    return x, back
end

function ChainRulesCore.rrule(::typeof(工Ɔlinear), T, C, Ɔ, Ɔb)
    x = 工Ɔlinear(T, C, Ɔ, Ɔb)
    function back(dx)
        dƆb, info = linsolve(x->x - ein"abc,(ad,dbe)->ce"(T,x,conj(T)) + ein"(ab,ab),cd->cd"(x,Ɔ,C), conj(dx))
        @assert info.converged == 1 "linear 工Ɔ not converged"
        dT = conj!(ein"(ab,bcd),ed->ace"(dƆb, conj(T), x)) + ein"(ab,acd),de->bce"(dƆb, T, x)
        dC = -conj!(ein"(ab,ab),cd->cd"(dƆb, Ɔ, x))
        dƆ = -conj!(ein"ab,(cd,cd)->ab"(dƆb, C, x))
        return NoTangent(), dT, dC, dƆ, conj(dƆb)
    end
    return x, back
end