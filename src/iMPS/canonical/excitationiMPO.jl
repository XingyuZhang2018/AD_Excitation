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

function envir_MPO_UD(U, D, M)
    atype = _arraytype(M)
    χ,d,_ = size(U)
    W     = size(M, 1)

    U = reshape(U, χ,d,χ)
    D = reshape(D, χ,d,χ)

    E = atype == Array ? zeros(ComplexF64, χ,W,χ) : CUDA.zeros(ComplexF64, χ,W,χ)
    Ǝ = atype == Array ? zeros(ComplexF64, χ,W,χ) : CUDA.zeros(ComplexF64, χ,W,χ)
    λ, ɔ = norm_R(U, conj(D))
    _, c = norm_L(U, conj(D))
    # @show λ1 λ2 ein"ab,ab->"(c,ɔ)
    c ./= ein"ab,ab->"(c,ɔ)

    E[:,W,:] = c
    for i in W-1:-1:1
        YL = atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in i+1:W
            YL += ein"(abc,db),(ae,edf)->cf"(U,M[j,:,i,:],E[:,j,:],conj(D)) ./ λ
        end
        if i == 1 #if M[i,:,i,:] == I(d)
            bL = YL - ein"(ab,ab),cd->cd"(YL,ɔ,c) 
            E[:,i,:], infoE = linsolve(E->E - ein"abc,(ad,dbe)->ce"(U,E,conj(D)) ./ λ + ein"(ab,ab),cd->cd"(E,ɔ,c), bL)
            @assert infoE.converged == 1
        else
            E[:,i,:] = YL
        end
    end

    Ǝ[:,1,:] = ɔ
    for i in 2:W
        YR = atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in 1:i-1
            YR += ein"((abc,db),cf),edf->ae"(U,M[i,:,j,:],Ǝ[:,j,:],conj(D)) ./ λ
        end
        if i == W # if M[i,:,i,:] == I(d)
            bR = YR - ein"(ab,ab),cd->cd"(c,YR,ɔ)
            Ǝ[:,i,:], infoƎ = linsolve(Ǝ->Ǝ - ein"(abc,ce),dbe->ad"(U,Ǝ,conj(D)) ./ λ + ein"(ab,ab),cd->cd"(c,Ǝ,ɔ), bR)
            @assert infoƎ.converged == 1
        else
            Ǝ[:,i,:] = YR
        end
    end

    return E, Ǝ, λ
end

"""
    ```
     ┌───B────┬─             a ────┬──── c 
     │   │    │              │     b     │ 
     E───M────s─             ├─ d ─┼─ e ─┤ 
     │   │    │              │     g     │ 
     └───AL*──┴─             f ────┴──── h  
    ```
"""
function einEB(k, B, AL, AR, E, M)
    ALs = circshift(AL, (0,0,0,0,-1))
    ARs = circshift(AR, (0,0,0,0,-1))
    Ms  = circshift(M , (0,0,0,0,0,-1))
    Bs  = circshift(B , (0,0,0,0,-1))
    Es  = circshift(E , (0,0,0,0,-1))
    # @show size.([ ein"((adfij,abcij),dgebij),fghij -> cehij"(Es,Bs,Ms,conj(ALs))])
    EB1, info = linsolve(EB->EB - exp(2.0im * k) * ein"((adfij,abcij),dgebij),fghij -> cehij"(ein"((adfij,abcij),dgebij),fghij -> cehij"(EB,AR,M,conj(AL)),ARs,Ms,conj(ALs)), ein"((adfij,abcij),dgebij),fghij -> cehij"(Es,Bs,Ms,conj(ALs)))
    # @assert info.converged == 1
    EB2, info = linsolve(EB->EB - exp(2.0im * k) * ein"((adfij,abcij),dgebij),fghij -> cehij"(ein"((adfij,abcij),dgebij),fghij -> cehij"(EB,AR,M,conj(AL)),ARs,Ms,conj(ALs)), ein"((adfij,abcij),dgebij),fghij -> cehij"(ein"((adfij,abcij),dgebij),fghij -> cehij"(E,B,M,conj(AL)),ARs,Ms,conj(ALs)))
    # @assert info.converged == 1
    return EB1 + EB2 * exp(1.0im * k)
end

"""
    ```
    ─┬───B───┐               a ────┬──── c
     │   │   │               │     b     │
    ─s───M───Ǝ               ├─ d ─┼─ e ─┤
     │   │   │               │     g     │
    ─┴───AR*─┘               f ────┴──── h 
    ```
"""
function einBƎ(k, B, AL, AR, Ǝ, M)
    ALs = circshift(AL, (0,0,0,0,1))
    ARs = circshift(AR, (0,0,0,0,1))
    Ms  = circshift(M , (0,0,0,0,0,1))
    Bs  = circshift(B , (0,0,0,0,1))
    Ǝs  = circshift(Ǝ , (0,0,0,0,1))
    BƎ1, info = linsolve(BƎ->BƎ - exp(2.0im *-k) * ein"((abcij,cehij),dgebij),fghij -> adfij"(ALs,ein"((abcij,cehij),dgebij),fghij -> adfij"(AL,BƎ,M,conj(AR)),Ms,conj(ARs)), ein"((abcij,cehij),dgebij),fghij -> adfij"(Bs,Ǝs,Ms,conj(ARs)))
    # @assert info.converged == 1
    BƎ2, info = linsolve(BƎ->BƎ - exp(2.0im *-k) * ein"((abcij,cehij),dgebij),fghij -> adfij"(ALs,ein"((abcij,cehij),dgebij),fghij -> adfij"(AL,BƎ,M,conj(AR)),Ms,conj(ARs)), ein"((abcij,cehij),dgebij),fghij -> adfij"(ALs,ein"((abcij,cehij),dgebij),fghij -> adfij"(B,Ǝ,M,conj(AR)),Ms,conj(ARs)))
    # @assert info.converged == 1
    return BƎ1 + BƎ2 * exp(1.0im *-k)
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
function H_canonical_eff(k, AL, AR, Bu, E, M, Ǝ)
    # 1. B and dB on the same site of M
    HB  = eindB(Bu, E, M, Ǝ) 
    # HB  = eindB(Bu, E, M, Ǝ) - ein"(((adf,abc),dgeb),ceh),fgh -> "(E,AC,M,Ǝ,conj(AC))[]  * Bu

    # # 2. B and dB on different sites of M
    EB = einEB(k, Bu, AL, AR, E, M)
    BƎ = einBƎ(k, Bu, AL, AR, Ǝ, M)
    HB += eindB(AR, EB, M, Ǝ) * exp(1.0im * k) +
          eindB(AL, E, M, BƎ) * exp(1.0im *-k)
          
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
                                           infolder = "./data/", outfolder = "./data/")
     infolder = joinpath( infolder, "$model")

    M = atype(MPO(model))
    D = size(M, 2)
    W = size(M, 1)
    MM= zeros(ComplexF64, (size(M)...,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        MM[:,:,:,:,i,j] = M
    end
    AL, C, AR = init_canonical_mps(;infolder = infolder, 
                                    atype = atype,  
                                    Ni = Ni,
                                    Nj = Nj,      
                                    D = D, 
                                    χ = χ)
    AC = ALCtoAC(AL, C)
    E, Ǝ = envir_MPO(AL, AR, MM)

    # AL, AR, AC = map(x->reshape(x, χ,D,χ), (AL, AR, AC ))
    # C = reshape(C, χ, χ)
    # E, Ǝ = map(x->reshape(x, χ,W,χ), (E, Ǝ))

    VL= initial_canonical_VL(AL)

    # X = zeros(ComplexF64, χ*(D-1), χ, Ni, Nj)
    X = atype(randn(ComplexF64, χ*(D-1), χ, Ni, Nj))
    # X[1] = 1.0
    # X = atype(X)
    # X /= sqrt(ein"ab,ab->"(X,conj(X))[])
    function f(X)
        Bu = ein"abcij,cdij->abdij"(VL, X)
        HB = H_canonical_eff(k, AL, AR, Bu, E, MM, Ǝ)
        HB = ein"abcij,abdij->dcij"(HB,conj(VL))
        return HB
    end
    Δ, Y, info = eigsolve(x -> f(x), X, n, :SR; ishermitian = false, maxiter = 100)
    # @assert info.converged == 1
    # Δ .-= Array(ein"(((adfij,abcij),dgebij),cehij),fghij -> "(circshift(ein"((adfij,abcij),dgebij),fghij -> cehij"(E,AL,MM,conj(AL)),(0,0,0,0,1)),AC,MM,Ǝ,conj(AC)))[] 
    return Δ, Y, info
end