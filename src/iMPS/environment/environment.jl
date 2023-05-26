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
function cmap(ci, Aui, Adi)
    cij = ein"(adi,acbi),dcei->bei"(ci,Aui,Adi)
    circshift(cij, (0,0,1))
end

"""
    ɔm = ɔmap(ARu, ARd, M, FR, i)

```
─ Au──┐       ──┐        a──┬──b
   │  ɔ  = λɔ   ɔ           c    
─ Ad──┘       ──┘        d──┴──e 
```
"""
function ɔmap(ɔi, Aui, Adi)
    ɔij = ein"(acbi,bei),dcei->adi"(Aui,ɔi,Adi)
    circshift(ɔij, (0,0,-1))
end

function cint(A)
    χ, Ni, Nj = size(A)[[1,4,5]]
    atype = _arraytype(A)
    c = atype == Array ? rand(ComplexF64, χ, χ, Ni, Nj) : CUDA.rand(ComplexF64, χ, χ, Ni, Nj)
    return c
end

function env_c(Au, Ad, c = cint(Au); ifcor_len=false, outfolder=nothing, kwargs...) 
    Ni,Nj = size(Au)[[4,5]]
    λc = zeros(eltype(c),Ni)
    ifcor_len ? (n=2) : (n=1)
    for i in 1:Ni
        λcs, cs, info = eigsolve(X->cmap(X, Au[:,:,:,i,:], Ad[:,:,:,i,:]), c[:,:,i,:], n, :LM; maxiter=100, ishermitian = false)
        info.converged == 0 && @warn "env_c not converged"
        if ifcor_len 
            logfile = open("$outfolder/correlation_length_c.log", "w")
            ξ = -1/log(abs(λcs[2]))
            write(logfile, "$(ξ)")
            close(logfile)
            println("save correlation length to $logfile")
        end
        λc[i], c[:,:,i,:] = selectpos(λcs, cs, Nj)
    end
    return λc, c
end

function env_ɔ(Au, Ad, ɔ = cint(Au); ifcor_len=false, outfolder=nothing, kwargs...) 
    Ni,Nj = size(Au)[[4,5]]
    λɔ = zeros(eltype(ɔ),Ni)
    ifcor_len ? (n=2) : (n=1)
    for i in 1:Ni
        λɔs, ɔs, info = eigsolve(X->ɔmap(X, Au[:,:,:,i,:], Ad[:,:,:,i,:]), ɔ[:,:,i,:], n, :LM; maxiter=100, ishermitian = false, kwargs...)
        info.converged == 0 && @warn "env_ɔ not converged"
        if ifcor_len 
            logfile = open("$outfolder/correlation_length_ɔ.log", "w")
            ξ = -1/log(abs(λɔs[2]))
            write(logfile, "$(ξ)")
            close(logfile)
            println("save correlation length to $logfile")
        end
        λɔ[i], ɔ[:,:,i,:] = selectpos(λɔs, ɔs, Nj)
    end
    return λɔ, ɔ
end

function envir(A; infolder = Defaults.infolder, outfolder = Defaults.outfolder)
    χ, D, _ = size(A)
    atype = _arraytype(A)
    Zygote.@ignore begin
        in_chkp_file = joinpath([infolder, "env", "D$(D)_χ$(χ).jld2"]) 
        if isfile(in_chkp_file)
            # println("environment load from $(in_chkp_file)")
            L_n, R_n = map(atype, load(in_chkp_file)["env"])
        else
            L_n = atype(rand(eltype(A), size(A,1), size(A,1)))
            R_n = atype(rand(eltype(A), size(A,3), size(A,3)))
        end 
    end
    _, L_n = norm_L(A, conj(A), L_n)
    _, R_n = norm_R(A, conj(A), R_n)

    Zygote.@ignore begin
        out_chkp_file = joinpath([outfolder,"env","D$(D)_χ$(χ).jld2"]) 
        save(out_chkp_file, "env", map(Array, (L_n, R_n)))
    end
    return L_n, R_n
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

EMmap(E, M, T, ⟂) = ein"((adfij,abcij),dgebij),fghij -> cehij"(E,T,M,conj(⟂))
MƎmap(Ǝ, M, T, ⟂) = ein"((abcij,cehij),dgebij),fghij -> adfij"(T,Ǝ,M,conj(⟂))
C工map(C, T, ⟂) = ein"(adij,abcij),dbeij->ceij"(C,T,conj(⟂))
工Ɔmap(Ɔ, T, ⟂) = ein"(abcij,ceij),dbeij->adij"(T,Ɔ,conj(⟂))

"""
    ```
     ┌───A───┐               a ────┬──── c
     │   │   │               │     b     │
     E───M───Ǝ               ├─ d ─┼─ e ─┤
     │   │   │               │     g     │
     └──   ──┘               f ────┴──── h 
    ```
"""
eindB(A, E, M, Ǝ) = ein"((adfij,abcij),dgebij),cehij->fghij"(E,A,M,Ǝ)

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
    # ACij = zero(ACj)
    # W = Int((size(Mj, 1)-2)/3)
    # for (i,j) in [(2,1),(2+W,1),(2+2*W,1),(2+3*W,1+W),(2+3*W,1+2*W),(2+3*W,1+3*W),(2+3*W,2),(2+3*W,2+W),(2+3*W,2+2*W)]
    #     ACij .+= ein"((adj,abcj),cfj),ebj->defj"(FLj[:,i,:,:],ACj,FRj[:,j,:,:],Mj[i,:,j,:,:])
    # end
    # for i in 2:W
    #     ACij .+= ein"(adj,abcj),cfj->dbfj"(FLj[:,i+1,:,:],ACj,FRj[:,i,:,:])
    #     ACij .+= ein"(adj,abcj),cfj->dbfj"(FLj[:,i+1+W,:,:],ACj,FRj[:,i+W,:,:])
    #     ACij .+= ein"(adj,abcj),cfj->dbfj"(FLj[:,i+1+2*W,:,:],ACj,FRj[:,i+2*W,:,:])
    # end

    # for i in [1,2+3*W]
    #     ACij .+= ein"(adj,abcj),cfj->dbfj"(FLj[:,i,:,:],ACj,FRj[:,i,:,:])
    # end
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

"""
    L_n, R_n = env_norm(A)

    get normalized environment of A

    ```  
                            a──────┬──────b
    ┌───┐                   │      │      │
    L   R  = 1              │      c      │
    └───┘                   │      │      │
                            d──────┴──────e      
                                                        
    ┌─ A ─      ┌─            ─ A──┐       ─┐
    L  │   =    L               │  R  =     R  
    └─ A*─      └─            ─ A*─┘       ─┘
    ```  
"""
function env_norm(A)
    _, L_n = norm_L(A, conj(A))
    _, R_n = norm_R(A, conj(A))
    n = Array(ein"((ad,acb),dce),be->"(L_n,A,conj(A),R_n))[]
    L_n /= n
    return L_n, R_n
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

"""
    ACm = ACmap2(ACij, FLj, FRj, Mj, II)

```
                               ┌──────   ACᵢⱼ   ───────┐          a ────┬─────┬──── d  
┌───── ACᵢ₊₁ⱼ ─────┐           │       │       │       │          │     b     c     │
│      │     │     │      =    FLᵢⱼ ── Mᵢⱼ ─── Mᵢⱼ₊₁── FRᵢⱼ₊₁     ├─ e ─┼─ f ─┼─ g ─┤ 
                               │       │       │       │          │     k     l     │ 
                                                                  h ────┴─────┴──── m 
                                                               
```
"""
function ACmap2(ACj, FLj, FRj, Mj, Mjr)
    ACij = ein"(((aehj,abcdj),ekfbj),flgcj),dgmj -> hklmj"(FLj,ACj,Mj,Mjr,FRj)
    circshift(ACij, (0,0,0,0,1))
end

"""
    ACenv2(AC, FL, M, FR;kwargs...)

Compute the up environment tensor for MPS `FL`,`FR` and MPO `M`, by finding the up fixed point
        of `FL - M - FR` contracted along the physical dimension.
```
┌──────   ACᵢⱼ   ───────┐         
│       │       │       │         =      λACᵢⱼ ┌─── ACᵢ₊₁ⱼ ──┐
FLᵢⱼ ── Mᵢⱼ ─── Mᵢⱼ₊₁── FRᵢⱼ₊₁                 │    │    │   │   
│       │       │       │   
```
"""
ACenv2(AC, FL, M, FR; kwargs...) = ACenv2!(copy(AC), FL, M, FR; kwargs...)
function ACenv2!(AC, FL, M, FR; kwargs...)
    Ni,Nj = size(M)[[5,6]]
    λAC = zeros(eltype(AC),Nj)
    for j in 1:Nj
        jr = mod1(j+1, Nj) 
        λACs, ACs, info = eigsolve(X->ACmap2(X, FL[:,:,:,:,j], FR[:,:,:,:,j], M[:,:,:,:,:,j], M[:,:,:,:,:,jr]), AC[:,:,:,:,:,j], 1, :SR; maxiter=100, ishermitian = false, kwargs...)
        @debug "ACenv! eigsolve" λACs info sort(abs.(λACs))
        info.converged == 0 && @warn "ACenv Not converged"
        λAC[j], AC[:,:,:,:,:,j] = selectpos(λACs, ACs, Ni)
    end
    return λAC, AC
end
