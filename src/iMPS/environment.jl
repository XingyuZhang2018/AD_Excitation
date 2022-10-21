using LinearAlgebra
using KrylovKit
using OMEinsum

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
    λ, L = norm_L(Au, Ad, L; kwargs...)
Compute the left environment tensor for normalization, by finding the left fixed point
of `Au - Ad` contracted Aung the physical dimension.
```
┌── Au─       ┌──        a──────┬──────b
L   │  =   λL L                 │       
└── Ad─       └──               c       
                                │       
                         d──────┴──────e
```
"""
function norm_L(Au, Ad, L = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(Ad,1))); kwargs...)
    λs, Ls, info = eigsolve(L -> ein"(ad,acb),dce -> be"(L,Au,Ad), L, 1, :LM; ishermitian = false, kwargs...)
    info.converged == 0 && @warn "leftenv not converged"
    return λs[1], Ls[1]
end

"""
    λ, R = norm_R(Au, Ad, R; kwargs...)
Compute the right environment tensor for normalization, by finding the right fixed point
of `Au - Ad` contracted Aung the physical dimension.
```
 ─ Au──┐       ──┐       a──────┬──────b
   │   R  = λR   R              │       
 ─ Ad──┘       ──┘              c       
                                │       
                         d──────┴──────e
```
"""
function norm_R(Au, Ad, R = _arraytype(Au)(randn(eltype(Au), size(Au,3), size(Ad,3))); kwargs...)
    Au = permutedims(Au,(3,2,1))
    Ad = permutedims(Ad,(3,2,1))
    return norm_L(Au, Ad, R; kwargs...)
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

safesign(x::Number) = iszero(x) ? one(x) : sign(x)
"""
    qrpos(A)
Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end
