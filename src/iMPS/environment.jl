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