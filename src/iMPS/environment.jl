using LinearAlgebra
using KrylovKit
using OMEinsum
using TeneT: qrpos, selectpos

"""
tensor order graph: rom left to right, top to bottom.
```
a ────┬──── c    a──────┬──────b   a──────┬──────b 
│     b     │    │      │      │          │                         
├─ d ─┼─ e ─┤    │      c      │          c                      
│     g     │    │      │      │          │       
f ────┴──── h    d──────┴──────e   d──────┴──────e

a ────┬──── c  
│     b     │
├─ d ─┼─ e ─┤
│     f     │
├─ g ─┼─ h ─┤           
│     i     │
j ────┴──── k     
```
"""

"""
    cm = cmap(ALu, ALd, FL)

```
┌── Au─       ┌──        a──┬──b
c   │  =   λc c             c      
└── Ad─       └──        d──┴──e                
```
"""
function cmap(ci::AbstractArray{T, 3}, Aui, Adi) where T
    cij = ein"(adi,acbi),dcei->bei"(ci,Aui,Adi)
    circshift(cij, (0,0,1))
end

function cmap(c::AbstractArray{T, 2}, Au, Ad) where T
    ein"(ad,acb),dce->be"(c,Au,conj(Ad))
end

"""
    ɔm = ɔmap(ARu, ARd, M, FR, i)

```
─ Au──┐       ──┐        a──┬──b
   │  ɔ  = λɔ   ɔ           c    
─ Ad──┘       ──┘        d──┴──e 
```
"""
function ɔmap(ɔi::AbstractArray{T, 3}, Aui, Adi) where T
    ɔij = ein"(acbi,bei),dcei->adi"(Aui,ɔi,Adi)
    circshift(ɔij, (0,0,-1))
end

function ɔmap(ɔ::AbstractArray{T, 2}, Au, Ad) where T
    ein"(acb,be),dce->ad"(Au,ɔ,conj(Ad))
end

function cint(A)
    χ, Ni, Nj = size(A)[[1,4,5]]
    atype = _arraytype(A)
    c = atype == Array ? rand(ComplexF64, χ, χ, Ni, Nj) : CUDA.rand(ComplexF64, χ, χ, Ni, Nj)
    return c
end

function env_c(Au, Ad, c = cint(Au); kwargs...) 
    Ni,Nj = size(Au)[[4,5]]
    λc = zeros(eltype(c),Ni)
    for i in 1:Ni
        λcs, cs, info = eigsolve(X->cmap(X, Au[:,:,:,i,:], Ad[:,:,:,i,:]), c[:,:,i,:], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        info.converged == 0 && @warn "env_c not converged"
        λc[i], c[:,:,i,:] = selectpos(λcs, cs, Nj)
    end
    return λc, c
end

function env_ɔ(Au, Ad, ɔ = cint(Au); kwargs...) 
    Ni,Nj = size(Au)[[4,5]]
    λɔ = zeros(eltype(ɔ),Ni)
    for i in 1:Ni
        λɔs, ɔs, info = eigsolve(X->ɔmap(X, Au[:,:,:,i,:], Ad[:,:,:,i,:]), ɔ[:,:,i,:], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        info.converged == 0 && @warn "env_c not converged"
        λɔ[i], ɔ[:,:,i,:] = selectpos(λɔs, ɔs, Nj)
    end
    return λɔ, ɔ
end

"""
    λ, L = norm_L(Au, Ad, L; kwargs...)
Compute the left environment tensor for normalization, by finding the left fixed point
of `Au - Ad` contracted Aung the physical dimension.

"""
function norm_L(Au, Ad, L = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(Ad,1))); kwargs...)
    λs, Ls, info = eigsolve(L -> ein"(ad,acb),dce -> be"(L,Au,Ad), L, 1, :LM; ishermitian = false, kwargs...)
    info.converged == 0 && @warn "norm_L not converged"
    return λs[1], Ls[1]
end

"""
    λ, R = norm_R(Au, Ad, R; kwargs...)
Compute the right environment tensor for normalization, by finding the right fixed point
of `Au - Ad` contracted Aung the physical dimension.

"""
function norm_R(Au, Ad, R = _arraytype(Au)(rand(eltype(Au), size(Au,3), size(Ad,3))); kwargs...)
    Au = permutedims(Au,(3,2,1))
    Ad = permutedims(Ad,(3,2,1))
    return norm_L(Au, Ad, R; kwargs...)
end

"""
    λ, E = env_E(Au, Ad, M, E = rand(eltype(Au), size(Au,1), size(M,1), size(Ad,1)); kwargs...)

Compute the left environment tensor for MPS `Au`,`Ad` and MPO `M`, by finding the left fixed point
of `Au - M - Ad` contracted along the physical dimension.
```
┌───Au─      ┌──        a ────┬──── c      
│   │        │          │     b     │        
E───M ─  = λ E─         ├─ d ─┼─ e ─┤      
│   │        │          │     g     │        
└───Ad─      └──        f ────┴──── h      
```
"""
function env_E(Au, Ad, M, E = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(M,1), size(Au,1))); kwargs...)
    λs, Es, info = eigsolve(E -> ein"((adf,abc),dgeb),fgh -> ceh"(E,Au,M,Ad), E, 1, :LM; ishermitian = false, kwargs...)
    info.converged == 0 && @warn "env_E not converged"
    return λs[1], Es[1]
end

"""
    λ, Ǝ = env_Ǝ(Au, Ad, M, Ǝ = rand(eltype(A), size(A,3), size(M,3), size(A,3)); kwargs...)

Compute the right environment tensor for MPS `Au`,`Ad` and MPO `M`, by finding the right fixed point
of `Au - M - Ad` contracted along the physical dimension.
```
 ──Au──┐       ──┐   
   │   │         │   
 ──M ──Ǝ   = λ ──Ǝ  
   │   │         │   
 ──Ad──┘       ──┘  
```
"""
function env_Ǝ(Au, Ad, M, Ǝ = _arraytype(Au)(rand(eltype(Au), size(Au,3), size(M,3), size(Ad,3))); kwargs...)
    Au = permutedims(Au,(3,2,1)  )
    Ad = permutedims(Ad,(3,2,1)  )
    M  = permutedims(M, (3,2,1,4))
    return env_E(Au, Ad, M, Ǝ; kwargs...)
end

"""
    ACm = ACmap(ACij, FLj, FRj, Mj, II)

```
                                ┌─────── ACᵢⱼ ─────┐              a ────┬──── c  
┌───── ACᵢ₊₁ⱼ ─────┐            │        │         │              │     b     │ 
│        │         │      =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ           ├─ d ─┼─ e ─┤ 
                                │        │         │              │     g     │ 
                                                                  f ────┴──── h 
                                                               
```
"""
function ACmap(ACj, FLj, FRj, Mj)
    ACij = ein"((adfj,abcj),dgebj),cehj -> fghj"(FLj,ACj,Mj,FRj)
    circshift(ACij, (0,0,0,1))
end

"""
    Cmap(Cij, FLjp, FRj, II)

```
                    ┌────Cᵢⱼ ───┐            a ─── b
┌── Cᵢ₊₁ⱼ ──┐       │           │            │     │
│           │  =   FLᵢⱼ₊₁ ──── FRᵢⱼ          ├─ c ─┤
                    │           │            │     │
                                             d ─── e                                    
```
"""
function Cmap(Cj, FLjr, FRj)
    Cij = ein"(acdj,abj),bcej -> dej"(FLjr,Cj,FRj)
    circshift(Cij, (0,0,1))
end

"""
    ACenv(AC, FL, M, FR;kwargs...)

Compute the up environment tensor for MPS `FL`,`FR` and MPO `M`, by finding the up fixed point
        of `FL - M - FR` contracted along the physical dimension.
```
┌─────── ACᵢⱼ ─────┐         
│        │         │         =  λACᵢⱼ ┌─── ACᵢ₊₁ⱼ ──┐
FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ               │      │      │   
│        │         │   
```
"""
ACenv(AC, FL, M, FR; kwargs...) = ACenv!(copy(AC), FL, M, FR; kwargs...)
function ACenv!(AC, FL, M, FR; kwargs...)
    Ni,Nj = size(M)[[5,6]]
    λAC = zeros(eltype(AC),Nj)
    for j in 1:Nj
        λACs, ACs, info = eigsolve(X->ACmap(X, FL[:,:,:,:,j], FR[:,:,:,:,j], M[:,:,:,:,:,j]), AC[:,:,:,:,j], 1, :SR; maxiter=100, ishermitian = false, kwargs...)
        @debug "ACenv! eigsolve" λACs info sort(abs.(λACs))
        info.converged == 0 && @warn "ACenv Not converged"
        λAC[j], AC[:,:,:,:,j] = selectpos(λACs, ACs, Ni)
    end
    return λAC, AC
end

"""
    Cenv(C, FL, FR;kwargs...)

Compute the up environment tensor for MPS `FL` and `FR`, by finding the up fixed point
    of `FL - FR` contracted along the physical dimension.
```
┌────Cᵢⱼ ───┐
│           │       =  λCᵢⱼ ┌──Cᵢⱼ ─┐
FLᵢⱼ₊₁ ──── FRᵢⱼ            │       │
│           │   
```
"""
Cenv(C, FL, FR; kwargs...) = Cenv!(copy(C), FL, FR; kwargs...)
function Cenv!(C, FL, FR; kwargs...)
    Ni,Nj = size(C)[[3,4]]
    λC = zeros(eltype(C),Nj)
    for j in 1:Nj
        jr = j + 1 - Nj * (j==Nj)
        λCs, Cs, info = eigsolve(X->Cmap(X, FL[:,:,:,:,jr], FR[:,:,:,:,j]), C[:,:,:,j], 1, :SR; maxiter=100, ishermitian = false, kwargs...)
        @debug "Cenv! eigsolve" λCs info sort(abs.(λCs))
        info.converged == 0 && @warn "Cenv Not converged"
        λC[j], C[:,:,:,j] = selectpos(λCs, Cs, Ni)
    end
    return λC, C
end

function overlap(Au, Ad)
    _, FLu_n = norm_L(Au, conj(Au))
    _, FRu_n = norm_R(Au, conj(Au))
    _, FLd_n = norm_L(Ad, conj(Ad))
    _, FRd_n = norm_R(Ad, conj(Ad))

    nu = ein"(ad,acb),(dce,be) ->"(FLu_n,Au,conj(Au),FRu_n)[]/ein"ab,ab ->"(FLu_n,FRu_n)[]
    Au /= sqrt(nu)
    nd = ein"(ad,acb),(dce,be) ->"(FLd_n,Ad,conj(Ad),FRd_n)[]/ein"ab,ab ->"(FLd_n,FRd_n)[]
    Ad /= sqrt(nd)

    _, FLud_n = norm_L(Au, conj(Ad))
    _, FRud_n = norm_R(Au, conj(Ad))
    @show ein"(ad,acb),(dce,be) ->"(FLud_n,Au,conj(Ad),FRud_n)[]/ein"ab,ab ->"(FLud_n,FRud_n)[]
    abs2(ein"(ad,acb),(dce,be) ->"(FLud_n,Au,conj(Ad),FRud_n)[]/ein"ab,ab ->"(FLud_n,FRud_n)[])
end