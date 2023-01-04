export excitation_spectrum_MPO

"""
    energy_gs_MPO(A, M, key)

ground state energy

a ────┬──── c 
│     b     │ 
├─ d ─┼─ e ─┤ 
│     g     │ 
f ────┴──── h 

"""
function energy_gs_MPO(A, M, E, Ǝ)
    # e = ein"(((adfij,abcij),dgebij),cehij),fghij -> "(E,A,M,Ǝ,conj(A))[]
    # n = ein"abcij,abcij -> "(E,Ǝ)[]
    # e = ein"abcij,abcij -> "(EMmap(EMmap(E, M, A, A), M, A, A), Ǝ)[]
    # n = ein"abcij,abcij -> "(EMmap(E, M, A, A), Ǝ)[]
    # e = ein"abcij,abcij -> "(EMmap(E, M, A, A), Ǝ)[]
    # n = ein"abcij,abcij -> "(E,Ǝ)[]
    e = ein"(((adfij,abcij),dgebij),cehij),fghij -> "(E,A,M,Ǝ,conj(A))[]
    n = ein"abcij,abcij -> "(circshift(E, (0,0,0,0,1)),Ǝ)[]
    n1 = ein"abcij,abcij -> "(E,circshift(Ǝ, (0,0,0,0,1)))[]
    @show e n (e-n)/2 n1

    return e-n
end

# function envir_MPO(A, M)
#     # D, χ, infolder, outfolder = key
#     # Zygote.@ignore begin
#     #     in_chkp_file = joinpath([infolder,"env","MPO_D$(D)_χ$(χ).jld2"]) 
#     #     if isfile(in_chkp_file)
#     #         # println("environment load from $(in_chkp_file)")
#     #         E,Ǝ = load(in_chkp_file)["env"]
#     #     else
#             E = _arraytype(A)(rand(eltype(A), size(A,1), size(M,1), size(A,1)))
#             Ǝ = _arraytype(A)(rand(eltype(A), size(A,3), size(M,3), size(A,3)))
#     #     end 
#     # end
#     _, E = env_E(A, conj(A), M, E)
#     _, Ǝ = env_Ǝ(A, conj(A), M, Ǝ)
    
#     # E /= ein"abc,abc->"(E, Ǝ)[]
    
#     # Zygote.@ignore begin
#     #     out_chkp_file = joinpath([outfolder,"env","MPO_D$(D)_χ$(χ).jld2"]) 
#     #     save(out_chkp_file, "env", (E, Ǝ))
#     # end
#     return E, Ǝ
# end

function envir_MPO(A, M, c, ɔ)
    atype = _arraytype(M)
    χ,Nx,Ny = size(A)[[1,4,5]]
    W       = size(M, 1)

    E = atype == Array ? zeros(ComplexF64, χ,W,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,W,χ,Nx,Ny)
    Ǝ = atype == Array ? zeros(ComplexF64, χ,W,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,W,χ,Nx,Ny)

    E[:,W,:,:,:] = c
    for i in W-1:-1:1
        YL = atype == Array ? zeros(ComplexF64, χ,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,χ,Nx,Ny)
        for j in i+1:W
            YL += ein"(abcij,dbij),(aeij,edfij)->cfij"(A,M[j,:,i,:,:,:],E[:,j,:,:,:],conj(A))
        end
        if i == 1 # if M[i,:,i,:] == I(d)
            bL = YL 
            E[:,i,:,:,:], infoE = linsolve(X->circshift(X, (0,0,0,1)) - ein"abcij,(adij,dbeij)->ceij"(A,X,conj(A)) + ein"(abij,abij),cdij->cdij"(X, ɔ, E[:,W,:,:,:]), bL)
            @assert infoE.converged == 1
        else
            E[:,i,:,:,:] = circshift(YL, (0,0,0,-1))
        end
        # E[:,i,:,:,:] = circshift(YL, (0,0,0,1))
    end

    Ǝ[:,1,:,:,:] = ɔ
    for i in 2:W
        YR = atype == Array ? zeros(ComplexF64, χ,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,χ,Nx,Ny)
        for j in 1:i-1
            YR += ein"((abcij,dbij),cfij),edfij->aeij"(A,M[i,:,j,:,:,:],Ǝ[:,j,:,:,:],conj(A))
        end
        if i == W # if M[i,:,i,:] == I(d)
            bR = YR 
            Ǝ[:,i,:,:,:], infoƎ = linsolve(X->circshift(X, (0,0,0,-1)) - ein"(abcij,ceij),dbeij->adij"(A,X,conj(A)) + ein"(abij,abij),cdij->cdij"(c, X, Ǝ[:,1,:,:,:]), bR)
            @assert infoƎ.converged == 1
        else
            Ǝ[:,i,:,:,:] = circshift(YR, (0,0,0,1))
        end
        # Ǝ[:,i,:,:,:] = circshift(YR, (0,0,0,-1))
    end

    return E, Ǝ
end

function EMmap(E, M, Au, Ad)
    E = ein"((adfij,abcij),dgebij),fghij -> cehij"(E,Au,M,conj(Ad))
    circshift(E, (0,0,0,0,1))
end

function MƎmap(Ǝ, M, Au, Ad)
    Ǝ = ein"((abcij,cehij),dgebij),fghij -> adfij"(Au,Ǝ,M,conj(Ad))
    circshift(Ǝ, (0,0,0,0,-1))
end

"""
    ```
     ┌───B────┬─             a ────┬──── c 
     │   │    │              │     b     │ 
     E───M────s─             ├─ d ─┼─ e ─┤ 
     │   │    │              │     g     │ 
     └───A*───┴─             f ────┴──── h  
    ```
"""
function einLB(k, L, B, A, E, M, Ǝ)
    LB1, info = linsolve(LB->LB - exp(2.0im * k) * EMmap(EMmap(LB, M, A, A), M, A, A) + exp(2.0im * k) * ein"(abcij,abcij),defij->defij"(LB,circshift(Ǝ,(0,0,0,0,-1)),E), EMmap(L, M, B, A))
    @assert info.converged == 1
    LB2, info = linsolve(LB->LB - exp(2.0im * k) * EMmap(EMmap(LB, M, A, A), M, A, A) + exp(2.0im * k) * ein"(abcij,abcij),defij->defij"(LB,circshift(Ǝ,(0,0,0,0,-1)),E), EMmap(EMmap(L, M, B, A), M, A, A))
    @assert info.converged == 1
    return LB1 * exp(1.0im * k) + LB2 * exp(2.0im * k) 
end

"""
    ```
    ─┬───B───┐               a ────┬──── c
     │   │   │               │     b     │
    ─s───M───Ǝ               ├─ d ─┼─ e ─┤
     │   │   │               │     g     │
    ─┴───A*──┘               f ────┴──── h 
    ```
"""
function einRB(k, R, B, A, E, M, Ǝ)
    RB1, info = linsolve(RB->RB - exp(2.0im *-k) *  MƎmap(MƎmap(RB, M, A, A), M, A, A) + exp(2.0im *-k) * ein"(abcij,abcij),defij->defij"(circshift(E,(0,0,0,0,-1)),RB,Ǝ), MƎmap(R, M, B, A))
    @assert info.converged == 1
    RB2, info = linsolve(RB->RB - exp(2.0im *-k) *  MƎmap(MƎmap(RB, M, A, A), M, A, A) + exp(2.0im *-k) * ein"(abcij,abcij),defij->defij"(circshift(E,(0,0,0,0,-1)),RB,Ǝ), MƎmap(MƎmap(R, M, B, A), M, A, A))
    @assert info.converged == 1
    return RB1 * exp(1.0im *-k) + RB2 * exp(2.0im *-k)
end

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
    H_mn = H_eff(k, A, Bu, Bd, H, L_n, R_n, s1, s2, s3)

    get `<Ψₖ(B)|H|Ψₖ(B)>`, including sum graphs form https://arxiv.org/abs/1810.07006 Eq.(268)
    ```
    1. Bu and Bd on the same site of M
        ┌───Bu──┐
        │   │   │
        E───M───Ǝ
        │   │   │
        └───Bd──┘

    2. B and dB on different sites of M
        ┌───Bu──┬───A───┐
        │   │   │   │   │
        E───M──s2───M───Ǝ
        │   │   │   │   │
        └───A*──┴───Bd──┘

        ┌───A───┬───Bu──┐
        │   │   │   │   │
        E───M──s3───M───Ǝ
        │   │   │   │   │
        └───Bd──┴───A*──┘

        s2 = sum of `eⁱᵏ 王` series: 
          ───         ─┬─              ─┬──┬─              ─┬──┬──┬─                 ─┬──┬──┬─...─┬─
                       │                │  │                │  │  │                   │  │  │     │ 
     eⁱ⁰ᵏ ───  +  eⁱ¹ᵏ─┼─    +    eⁱ²ᵏ ─┼──┼─    +    eⁱ³ᵏ ─┼──┼──┼─  + ... +   eⁱⁿᵏ ─┼──┼──┼─...─┼─  + ...
                       │                │  │                │  │  │                   │  │  │     │ 
          ───         ─┴─              ─┴──┴─              ─┴──┴──┴─                 ─┴──┴──┴─...─┴─


    ```

"""
function H_MPO_eff(k, A, Bu, E, M, Ǝ)
    # 1. B and dB on the same site of M
    HB = eindB(Bu, E, M, Ǝ)

    # 2. B and dB on different sites of M
    HB += eindB(A, einLB(k, E, Bu, A, E, M, Ǝ), M, Ǝ) +
          eindB(A, E, M, einRB(k, Ǝ, Bu, A, E, M, Ǝ)) 

    return HB
end

"""
    ```
     ┌───Bu──┐             a──┬──b
     │   │   │             │  │  │ 
     c   │   ɔ             │  c  │ 
     │   │   │             │  │  │ 
     └──   ──┘             d─   ─e
    ```
"""
function N_MPO_eff(Bu, c, ɔ)
    # 1. B, dB on the same site
    NB = ein"(adij,acbij),beij->dceij"(c, Bu, ɔ)

    return NB
end

"""
    excitation_spectrum(k, A, H, n)

find at least `n` smallest excitation gaps 
"""
function excitation_spectrum_MPO(k, A, model, n::Int = 1;
                             infolder = "./data/", outfolder = "./data/")
     infolder = joinpath( infolder, "$model")
    outfolder = joinpath(outfolder, "$model")

    χ, D, _, Ni,Nj = size(A)
    Mo = _arraytype(A)(MPO(model))
    M  = zeros(ComplexF64, (size(Mo)...,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        M[:,:,:,:,i,j] = Mo
    end

    _, c = env_c(A, conj(A))
    _, ɔ = env_ɔ(A, conj(A))
    
    for j in 1:Nj, i in 1:Ni
        jr = j+1 - Nj*(j+1>Nj)
        ɔ[:,:,i,j] ./= ein"ab,ab->"(c[:,:,i,jr],ɔ[:,:,i,j])
    end
    E, Ǝ      = envir_MPO(A, M, c, ɔ)
    sq_c = similar(c)
    sq_ɔ = similar(c)
    inv_sq_c = similar(c)
    inv_sq_ɔ = similar(c)
    VL = zeros(ComplexF64, χ,D,(D-1)*χ,Ni,Nj)
    for j in 1:Nj, i in 1:Ni
        sq_c[:,:,i,j]     = sqrt(c[:,:,i,j])
        sq_ɔ[:,:,i,j]     = sqrt(ɔ[:,:,i,j])
        inv_sq_c[:,:,i,j] = (sq_c[:,:,i,j])^-1 
        inv_sq_ɔ[:,:,i,j] = (sq_ɔ[:,:,i,j])^-1
        VL[:,:,:,i,j]     = initial_VL(A[:,:,:,i,j], c[:,:,i,j])
    end
  
    X = rand(ComplexF64, χ*(D-1), χ, Ni, Nj)
    # X = ones(ComplexF64, χ*(D-1), χ, Ni, Nj)
    # X[1] = 1.0
    # X /= sqrt(ein"ab,ab->"(X,conj(X))[])
    E1 = ein"abcij,abcij -> ij"(EMmap(E, M, A, A), Ǝ)
    # E2 = ein"abcij,abcij -> ij"(EMmap(EMmap(E, M, A, A), M, A, A), Ǝ)
    # E3 = ein"abcij,abcij -> ij"(E, MƎmap(Ǝ, M, A, A))
    # E4 = ein"abcij,abcij -> ij"(E, MƎmap(MƎmap(Ǝ, M, A, A), M, A, A))
    # @show E1 E2 E3 E4
    E0 = E1
    # E0 = ein"abcij,abcij -> ij"(EMmap(E, M, A, A), Ǝ)
    # E0[1,1], E0[1,2] = E0[1,2], E0[1,1]
    # @show energy_gs_MPO(A, M, E, Ǝ)
    function f(X)
        Bu = ein"((baij,bcdij),deij),efij->acfij"(inv_sq_c, VL, X, inv_sq_ɔ)
        HB = H_MPO_eff(k, A, Bu, E, M, Ǝ) - ein"(adij,acbij),beij, ij->dceij"(c, Bu, ɔ, E0)
        # HB = H_MPO_eff(k, A, Bu, E, M, Ǝ)
        # NB = N_MPO_eff(Bu, c, ɔ)
        HB = ein"((baij,bcdij),acfij),deij->feij"(inv_sq_c,HB,conj(VL),inv_sq_ɔ)
        # NB = ein"((baij,bcdij),acfij),deij->feij"(inv_sq_c,NB,conj(VL),inv_sq_ɔ)
        return HB
    end
    Δ, Y, info = eigsolve(x -> f(x), X, n, :SR; ishermitian = false, maxiter = 100)
    @show Δ
    # Δ /= 2
    # Δ, Y, info = geneigsolve(x -> f(x), X, n, :SR, ishermitian = true, isposdef = true)
    # @assert info.converged == 1
    # @show ein"(((adfij,abcij),dgebij),cehij),fghij -> "(E,A,M,Ǝ,conj(A))[]
    # Δ .-= ein"abcij,abcij -> "(EMmap(EMmap(E, M, A, A), M, A, A), Ǝ)
    # E0 = ein"(((adfij,abcij),dgebij),cehij),fghij -> "(E,A,M,Ǝ,conj(A))[]
    # E0 = ein"abcij,abcij -> "(EMmap(E, M, A, A), Ǝ)[]
    # E1 = ein"abcij,abcij -> "(EMmap(EMmap(E, M, A, A), M, A, A), Ǝ)[]
    # @show Δ E0 E1
    # Δ .-= E0
    return Δ, Y, info
end