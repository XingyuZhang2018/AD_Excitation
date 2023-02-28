using TeneT: leftenv, rightenv
export excitation_spectrum_canonical_MPO

function initial_canonical_VL(AL)
    χ,D,_ = size(AL)
    VL = _arraytype(AL)(randn(χ, D, χ*(D-1)))
    λL = ein"abc,abd -> cd"(VL,conj(AL))
    VL -= ein"abc,dc -> abd"(AL,λL)
    Q, _ = qrpos(reshape(VL, χ*D, χ*(D-1)))
    VL = reshape(Q, χ, D, χ*(D-1))
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
    _, ɔ = norm_R(U, conj(D))
    _, c = norm_L(U, conj(D))
    # @show λ1 λ2 ein"ab,ab->"(c,ɔ)
    # c ./= ein"ab,ab->"(c,ɔ)

    E[:,W,:] = c
    for i in W-1:-1:1
        YL = atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in i+1:W
            YL += ein"(abc,db),(ae,edf)->cf"(U,M[j,:,i,:],E[:,j,:],conj(D))
        end
        if i == 1 #if M[i,:,i,:] == I(d)
            bL = YL
            E[:,i,:], infoE = linsolve(E->E - ein"abc,(ad,dbe)->ce"(U,E,conj(D)) + ein"(ab,ab),cd->cd"(E,ɔ,c), bL)
            @assert infoE.converged == 1
        else
            E[:,i,:] = YL
        end
    end

    Ǝ[:,1,:] = ɔ
    for i in 2:W
        YR = atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in 1:i-1
            YR += ein"((abc,db),cf),edf->ae"(U,M[i,:,j,:],Ǝ[:,j,:],conj(D))
        end
        if i == W # if M[i,:,i,:] == I(d)
            bR = YR
            Ǝ[:,i,:], infoƎ = linsolve(Ǝ->Ǝ - ein"(abc,ce),dbe->ad"(U,Ǝ,conj(D)) + ein"(ab,ab),cd->cd"(c,Ǝ,ɔ), bR)
            @assert infoƎ.converged == 1
        else
            Ǝ[:,i,:] = YR
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
     └───AL*──┴─             f ────┴──── h  
    ```
"""
function einEB(W, k, ELL, B, AL, AR, ERL, ƎRL, M)
    kx, ky = k
    EM = EMmap(ELL, M, B, AL)
    coef = series_coef_L(k, W)
    EMs = sum(collect(Iterators.take(iterated(x->EMmap(x, M, AR, AL), EM), W)) .* coef)
    EB, info = linsolve(EB->EB - exp(1.0im * kx) * nth(iterated(x->EMmap(x, M, AR, AL), EB), W+1) + exp(1.0im * kx) * ein"(abc,abc),def->def"(EB, ƎRL, ERL), EMs)
    @assert info.converged == 1
    return EB
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
function einBƎ(W, k, ƎRR, B, AL, AR, ELR, ƎLR, M)
    kx, ky = k
    MƎ = MƎmap(ƎRR, M, B, AR)
    coef = series_coef_R(k, W)
    MƎs = sum(collect(Iterators.take(iterated(x->MƎmap(x, M, AL, AR), MƎ), W)) .* coef)
    BƎ, info = linsolve(BƎ->BƎ - exp(-1.0im * kx) * nth(iterated(x->MƎmap(x, M, AL, AR), BƎ), W+1) + exp(-1.0im * kx) * ein"(abc,abc),def->def"(ELR, BƎ, ƎLR), MƎs)
    @assert info.converged == 1
    return BƎ
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
function H_canonical_eff(W, k, AL, AR, Bu,  M, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR)
    # 1. B and dB on the same site of M
    HB  = eindB(Bu, ELL, M, ƎRR) 

    # # 2. B and dB on different sites of M
    EB = einEB(W, k, ELL, Bu, AL, AR, ERL, ƎRL, M)
    BƎ = einBƎ(W, k, ƎRR, Bu, AL, AR, ELR, ƎLR, M)
    HB += eindB(AR, EB, M, ƎRR) +
          eindB(AL, ELL, M, BƎ)
          
    return HB
end

"""
    excitation_spectrum(k, A, H, n)

find at least `n` smallest excitation gaps 
"""
function excitation_spectrum_canonical_MPO(model, k, n::Int = 1;
                                           χ::Int = 8,
                                           atype = Array,
                                           infolder = "../data/", outfolder = "../data/")
     infolder = joinpath( infolder, "$model")
    outfolder = joinpath(outfolder, "$model")

    M = atype(MPO(model))
    D = size(M, 2)
    W = size(M, 1)
    AL, C, AR = init_canonical_mps(;infolder = infolder, 
                                    atype = atype,        
                                    D = D, 
                                    χ = χ)
    AC = ALCtoAC(AL, C)
    ELL, ƎRR = envir_MPO(AL, AR, reshape(M, (W,D,W,D,1,1)))
    ERL, ƎRL = envir_MPO_UD(AR, AL, M)
    ELR, ƎLR = envir_MPO_UD(AL, AR, M)

    AL, AR, AC = map(x->reshape(x, χ,D,χ), (AL, AR, AC))
    ELL, ƎRR = map(x->reshape(x, χ,W,χ), (ELL, ƎRR))

    VL = initial_canonical_VL(AL)

    # X = zeros(ComplexF64, χ*(D-1), χ)
    X = atype(rand(ComplexF64, χ*(D-1), χ))
    # X[1] = 1.0
    # X = atype(X)
    # X /= sqrt(ein"ab,ab->"(X,conj(X))[])
    function f(X)
        Bu = ein"abc,cd->abd"(VL, X)
        HB = H_canonical_eff(W, k, AL, AR, Bu, M, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR)
        HB = ein"abc,abd->dc"(HB,conj(VL))
        return HB
    end
    Δ, Y, info = eigsolve(x -> f(x), X, n, :SR; ishermitian = false, maxiter = 100)
    info.converged != 1 && @warn("eigsolve doesn't converged")
    Δ .-= real(Array(ein"(((adf,abc),dgeb),ceh),fgh -> "(ELL,AC,M,ƎRR,conj(AC)))[])
    return Δ, Y, info
end