using TeneT: leftenv, rightenv
export excitation_spectrum_canonical_MPO

function initial_canonical_VL(AL)
    χ,D,_,Ni,Nj = size(AL)
    VL = _arraytype(AL)(randn(ComplexF64, χ, D, χ*(D-1), Ni, Nj))
    for j in 1:Nj, i in 1:Ni
        λL = ein"abc,abd -> cd"(VL[:,:,:,i,j],conj(AL[:,:,:,i,j]))
        VL[:,:,:,i,j] -= ein"abc,dc -> abd"(AL[:,:,:,i,j],λL)
        Q, _ = qrpos(reshape(VL[:,:,:,i,j], χ*D, χ*(D-1)))
        VL[:,:,:,i,j] = reshape(Q, χ, D, χ*(D-1))
    end
    return VL
end

function energy_gs_canonical_MPO(M, AC, C, E, Ǝ)
    e = ein"(((adf,abc),dgeb),ceh),fgh -> "(E,AC,M,Ǝ,conj(AC))[]
    n = ein"((ab,acd),bce),de->"(C,E,Ǝ,conj(C))[]
    @show e n
    return e-n
end

function envir_MPO_exci(U, D, C, M; type = "RL")
    atype = _arraytype(M)
    χ,Nx,Ny = size(U)[[1,4,5]]
    W       = size(M, 1)

    E = atype == Array ? zeros(ComplexF64, χ,W,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,W,χ,Nx,Ny)
    Ǝ = atype == Array ? zeros(ComplexF64, χ,W,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,W,χ,Nx,Ny)

    # if type == "RL"
    #     c = circshift(ein"abij->baij"(C), (0,0,0,1))
    #     ɔ = ein"abij->baij"(conj(C))
    # else
    #     c = circshift(conj(C), (0,0,0,1))
    #     ɔ = C
    # end
    _, c = env_c(U, conj(D))
    _, ɔ = env_ɔ(U, conj(D))
    for y in 1:Ny, x in 1:Nx
        yr = mod1(y+1, Ny) 
        ɔ[:,:,x,y] ./= ein"ab,ab->"(c[:,:,x,yr],ɔ[:,:,x,y])[]
    #     # @show ein"ab,ab->"(c[:,:,x,yr],ɔ[:,:,x,y])[]
    end
    E[:,W,:,:,:] = c
    for i in W-1:-1:1
        YL = atype == Array ? zeros(ComplexF64, χ,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,χ,Nx,Ny)
        for j in i+1:W
            YL += ein"(abcij,dbij),(aeij,edfij)->cfij"(U,M[j,:,i,:,:,:],E[:,j,:,:,:],conj(D))
        end
        if i == 1 #if M[i,:,i,:] == I(d)
            bL = YL
            E[:,i,:,:,:], infoE = linsolve(X->circshift(X, (0,0,0,-1)) - ein"abcij,(adij,dbeij)->ceij"(U,X,conj(D)) + ein"(abij,abij),cdij->cdij"(circshift(X, (0,0,0,-1)), ɔ, circshift(c, (0,0,0,-1))), bL, E[:,i,:,:,:])
            @assert infoE.converged == 1
        else
            E[:,i,:,:,:] = circshift(YL, (0,0,0,1))
        end
    end

    Ǝ[:,1,:,:,:] = ɔ
    for i in 2:W
        YR = atype == Array ? zeros(ComplexF64, χ,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,χ,Nx,Ny)
        for j in 1:i-1
            YR += ein"((abcij,dbij),cfij),edfij->aeij"(U,M[i,:,j,:,:,:],Ǝ[:,j,:,:,:],conj(D))
        end
        if i == W # if M[i,:,i,:] == I(d)
            bR = YR 
            Ǝ[:,i,:,:,:], infoƎ = linsolve(X->circshift(X, (0,0,0,1)) - ein"(abcij,ceij),dbeij->adij"(U,X,conj(D)) + ein"(abij,abij),cdij->cdij"(c, circshift(X, (0,0,0,1)), circshift(ɔ, (0,0,0,-1))), bR, Ǝ[:,i,:,:,:])
            @assert infoƎ.converged == 1
        else
            Ǝ[:,i,:,:,:] = circshift(YR, (0,0,0,-1))
        end
    end

    return E, Ǝ
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
function einEB(k, ELL, B, AL, AR, ERL, ƎRL, M)
    LB, info = linsolve(LB->LB - circshift(ein"ij,abcij->abcij"([1 exp(1.0im * k * 2)], EMmap(LB, M, AR, AL)), (0,0,0,0,1)) + exp(1.0im * k) * ein"(abcij,abcij),defij->defij"(LB,circshift(ƎRL, (0,0,0,0,1)),ERL), circshift(ein"ij,abcij->abcij"([1 exp(1.0im * k * 2)], EMmap(ELL, M, B, AL)), (0,0,0,0,1)))
    # + EMmap(EMmap(ELL, M, B, AL), M, circshift(AR,(0,0,0,0,-1)), circshift(AL,(0,0,0,0,-1)))
    @assert info.converged == 1
    return LB
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
function einBƎ(k, ƎRR, B, AL, AR, ELR, ƎLR, M)
    RB, info = linsolve(RB->RB - circshift(ein"ij,abcij->abcij"([exp(1.0im *-k * 2) 1], MƎmap(RB, M, AL, AR)), (0,0,0,0,-1)) + exp(1.0im *-k) * ein"(abcij,abcij),defij->defij"(circshift(ELR,(0,0,0,0,-1)),RB,ƎLR), circshift(ein"ij,abcij->abcij"([exp(1.0im *-k * 2) 1], MƎmap(ƎRR, M, B, AR)), (0,0,0,0,-1)))
    # + MƎmap(MƎmap(ƎRR, M, B, AR), M, circshift(AL,(0,0,0,0,1)), circshift(AR,(0,0,0,0,1)))
    @assert info.converged == 1
    return RB
end

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
function H_canonical_eff(k, AL, AR, Bu,  M, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR)
    # 1. B and dB on the same site of M
    HB  = eindB(Bu, ELL, M, ƎRR) 

    # # 2. B and dB on different sites of M
    EB = einEB(k, ELL, Bu, AL, AR, ERL, ƎRL, M)
    BƎ = einBƎ(k, ƎRR, Bu, AL, AR, ELR, ƎLR, M)
    HB += eindB(AR, EB, M, ƎRR) +
          eindB(AL, ELL, M, BƎ)
          
    return HB
end

"""
    excitation_spectrum(k, A, H, n)

find at least `n` smallest excitation gaps 
"""
function excitation_spectrum_canonical_MPO(model, k, n::Int = 1;
                                           Ni::Int = 1,
                                           Nj::Int = 1,
                                           χ::Int = 8,
                                           atype = Array,
                                           infolder = "../data/", outfolder = "../data/")
     infolder = joinpath(infolder, "$model")

    Mo = atype(MPO(model))
    D = size(Mo, 2)
    W = size(Mo, 1)
    M = zeros(ComplexF64, (size(Mo)...,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        M[:,:,:,:,i,j] = Mo
    end
    AL, C, AR = init_canonical_mps(;infolder = infolder, 
                                    atype = atype,  
                                    Ni = Ni,
                                    Nj = Nj,      
                                    D = D, 
                                    χ = χ)
    # for j in 1:Nj, i in 1:Ni
    #     C[:,:,i,j] /= tr(C[:,:,i,j])
    # end
    AC = ALCtoAC(AL, C)
    ELL, ƎRR = envir_MPO(AL, AR, C, M)
    ERL, ƎRL = envir_MPO_exci(AR, AL, C, M; type = "RL")
    ELR, ƎLR = envir_MPO_exci(AL, AR, C, M; type = "LR")

    VL= initial_canonical_VL(AL)

    X = atype(rand(ComplexF64, χ*(D-1), χ, Ni, Nj))
    E0 = ein"(((adfij,abcij),dgebij),cehij),fghij -> ij"(ELL,AC,M,ƎRR,conj(AC))
    # @show E0
    function f(X)
        Bu = ein"abcij,cdij->abdij"(VL, X)
        HB = H_canonical_eff(k, AL, AR, Bu, M, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR) - ein"abcij, ij->abcij"(Bu, E0)
        HB = ein"abcij,abdij->dcij"(HB,conj(VL))
        return HB
    end
    Δ, Y, info = eigsolve(x -> f(x), X, n, :SR; ishermitian = false, maxiter = 100)
    info.converged == 1 && @warn("eigsolve not converged")
    return real(Δ), Y, info
end