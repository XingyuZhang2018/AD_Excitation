using TeneT: leftenv, rightenv
using JLD2
export excitation_spectrum_canonical_MPO

function initial_canonical_VL(AL)
    χ,D,_,Ni,Nj = size(AL)
    VL = _arraytype(AL)(randn(ComplexF64, χ, D, χ*(D-1), Ni, Nj))
    for j in 1:Nj, i in 1:Ni
        λL = ein"abc,abd -> cd"(VL[:,:,:,i,j],conj(AL[:,:,:,i,j]))
        VL[:,:,:,i,j] -= ein"abc,dc -> abd"(AL[:,:,:,i,j],λL)
        Q, _ = qrpos(reshape(VL[:,:,:,i,j], χ*D, χ*(D-1)))
        VL[:,:,:,i,j] = reshape(Q, χ, D, χ*(D-1))
        λL = ein"abc,abd -> cd"(VL[:,:,:,i,j],conj(AL[:,:,:,i,j]))
        VL[:,:,:,i,j] -= ein"abc,dc -> abd"(AL[:,:,:,i,j],λL)
    end
    return VL
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
        ɔ[:,:,x,y] ./= Array(ein"ab,ab->"(c[:,:,x,yr],ɔ[:,:,x,y]))[]
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
     └───AL*──┴─             f ────┴──── h  
    ```
"""
function einEB(W, k, ELL, B, AL, AR, ERL, ƎRL, M)
    kx, ky = k
    EM = EMmap(ELL, M, B, AL)
    coef = series_coef_L(k, W)
    EMs = sum(collect(Iterators.take(iterated(x->EMmap(x, M, AR, AL), EM), W)) .* coef)
    EB, info = linsolve(EB->EB - exp(1.0im * kx) * nth(iterated(x->EMmap(x, M, AR, AL), EB), W+1) + exp(1.0im * kx) * ein"(abcij,abcij),defij->defij"(EB, ƎRL, ERL), EMs)
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
    BƎ, info = linsolve(BƎ->BƎ - exp(-1.0im * kx) * nth(iterated(x->MƎmap(x, M, AL, AR), BƎ), W+1) + exp(-1.0im * kx) * ein"(abcij,abcij),defij->defij"(ELR, BƎ, ƎLR), MƎs)
    @assert info.converged == 1
    return BƎ
end

"""
    H_mn = H_canonical_eff(W, k, AL, AR, Bu, M, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR)

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
function H_canonical_eff(W, k, AL, AR, Bu, M, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR)
    # 1. B and dB on the same site of M
    HB  = eindB(Bu, ELL, M, ƎRR) 

    # # 2. B and dB on different sites of M
    EB = einEB(W, k, ELL, Bu, AL, AR, ERL, ƎRL, M)
    BƎ = einBƎ(W, k, ƎRR, Bu, AL, AR, ELR, ƎLR, M)
    HB += eindB(AR, EB, M, ƎRR) +
          eindB(AL, ELL, M, BƎ)
          
    return HB
end

function canonical_exci_env(model, Nj, χ; infolder, atype, ifmerge, if4site)
    infolder = joinpath( infolder, "$model")

    Mo = if4site ? atype(MPO_2x2(model)) : atype(MPO(model))
    D1 = size(Mo, 1)
    D2 = size(Mo, 2)
    if ifmerge
        M = reshape(ein"abcg,cdef->abdegf"(Mo,Mo), (D1, D2^2, D1, D2^2, 1, 1))
    else
        M = atype(zeros(ComplexF64, (size(Mo)...,1,Nj)))
        for j in 1:Nj
            M[:,:,:,:,1,j] = Mo
        end
    end
    AL, C, AR = init_canonical_mps(;infolder = joinpath(infolder, "groundstate"), 
                                    atype = atype,  
                                    Nj = Nj,      
                                    D = D2, 
                                    χ = χ)
    if ifmerge
        AL = reshape(ein"abc,cde->abde"(AL[:,:,:,1,1], AL[:,:,:,1,2]), (χ, D2^2, χ, 1, 1))
        AR = reshape(ein"abc,cde->abde"(AR[:,:,:,1,1], AR[:,:,:,1,2]), (χ, D2^2, χ, 1, 1))
        C = reshape(C[:,:,1,2], (χ, χ, 1, 1))  
    end

    AC = ALCtoAC(AL, C)
    ELL, ƎRR = envir_MPO(AL, AR, M)
    ERL, ƎRL = envir_MPO_exci(AR, AL, C, M; type = "RL")
    ELR, ƎLR = envir_MPO_exci(AL, AR, C, M; type = "LR")

    VL = initial_canonical_VL(AL)

    return M, AL, C, AR, AC, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR, VL
end

"""
    excitation_spectrum(k, A, H, n)

find at least `n` smallest excitation gaps 
"""
function excitation_spectrum_canonical_MPO(model, k, n::Int = 1;
                                           Ni::Int = 1, Nj::Int = 1,
                                           χ::Int = 8,
                                           atype = Array,
                                           ifmerge::Bool = false,
                                           if4site::Bool = false,
                                           infolder = "../data/", outfolder = "../data/")

    M, AL, C, AR, AC, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR, VL = canonical_exci_env(model, Nj, χ;  
            infolder = infolder, atype = atype, ifmerge = ifmerge, if4site = if4site)

    X = atype(rand(ComplexF64, χ*(size(AL, 2)-1), χ, size(AL, 4), size(AL, 5)))
    E0 = ein"(((adfij,abcij),dgebij),cehij),fghij -> ij"(ELL,AC,M,ƎRR,conj(AC))
    W = model.W
    ifmerge && (W = Int(W/2))
    # @show E0
    function f(X)
        Bu = ein"abcij,cdij->abdij"(VL, X)
        HB = H_canonical_eff(W, k, AL, AR, Bu, M, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR) - ein"abcij, ij->abcij"(Bu, E0)
        # HB = H_canonical_eff(W, k, AL, AR, Bu, M, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR)
        HB = ein"abcij,abdij->dcij"(HB,conj(VL))
        return HB
    end
    Δ, Y, info = eigsolve(x -> f(x), X, n, :SR; ishermitian = true, maxiter = 100)
    info.converged != 1 && @warn("eigsolve doesn't converged")
    # if4site && (Δ /= 4)
    save_canonical_excitaion(joinpath(outfolder, "$model"), W, Nj, size(AL,2), χ, k, Δ, VL, Y)
    # Δ .-= E0
    return Δ
end

function save_canonical_excitaion(outfolder, W, Nj, D, χ, k, Δ, VL, X)
    kx, ky = k
    filepath = joinpath(outfolder, "canonical/Nj$(Nj)_D$(D)_χ$(χ)/")
    !(ispath(filepath)) && mkpath(filepath)
    logfile = open("$filepath/kx$(round(Int,kx/pi*W/2))_ky$(round(Int,ky/pi*W/2)).log", "w")
    write(logfile, "$(Δ)")
    close(logfile)

    out_chkp_file = "$filepath/excitaion_VLX_kx$(round(Int,kx/pi*W/2))_ky$(round(Int,ky/pi*W/2)).jld2"
    save(out_chkp_file, "VLX", (Array(VL), map(Array, X)))
    println("excitaion file saved @$logfile")
end