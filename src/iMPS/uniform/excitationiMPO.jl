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
function energy_gs_MPO(A, M)
    E, Ǝ = envir_MPO(A, M)
    e = ein"(((adf,abc),dgeb),ceh),fgh -> "(E,A,M,Ǝ,conj(A))[]
    n = ein"abc,abc -> "(E,Ǝ)[]
    @show e n
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

function envir_MPO(A, c, ɔ, M)
    atype = _arraytype(A)
    χ,d,_,Nx,Ny = size(A)
    W           = size(M, 1)

    E = atype == Array ? zeros(ComplexF64, χ,W,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,W,χ,Nx,Ny)
    Ǝ = atype == Array ? zeros(ComplexF64, χ,W,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,W,χ,Nx,Ny)

    E[:,W,:,:,:] = c
    for i in W-1:-1:1
        YL = atype == Array ? zeros(ComplexF64, χ,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,χ,Nx,Ny)
        for j in i+1:W
            YL += ein"(abcij,dbij),(aeij,edfij)->cfij"(A,M[j,:,i,:,:,:],E[:,j,:,:,:],conj(A))
        end
        if M[i,:,i,:] == I(d)
            bL = YL - ein"abij,abij->"(YL,ɔ)[] * c
            E[:,i,:,:,:], infoE = linsolve(X->circshift(X, (0,0,0,1)) - ein"abcij,(adij,dbeij)->ceij"(A,X,conj(A)) + ein"(abij,abij),cdij->cdij"(X, ɔ, E[:,W,:,:,:]), bL)
            @assert infoE.converged == 1
        else
            E[:,i,:,:,:] = circshift(YL, (0,0,0,-1))
        end
    end

    Ǝ[:,1,:,:,:] = ɔ
    for i in 2:W
        YR = atype == Array ? zeros(ComplexF64, χ,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,χ,Nx,Ny)
        for j in 1:i-1
            YR += ein"((abcij,dbij),cfij),edfij->aeij"(A,M[i,:,j,:,:,:],Ǝ[:,j,:,:,:],conj(A))
        end
        if M[i,:,i,:] == I(d)
            bR = YR - ein"abij,abij->"(c,YR)[] * ɔ
            Ǝ[:,i,:,:,:], infoƎ = linsolve(X->circshift(X, (0,0,0,-1)) - ein"(abcij,ceij),dbeij->adij"(A,X,conj(A)) + ein"(abij,abij),cdij->cdij"(c, X, Ǝ[:,1,:,:,:]), bR)
            @assert infoƎ.converged == 1
        else
            Ǝ[:,i,:,:,:] = circshift(YR, (0,0,0,1))
        end
    end

    # @show ein"ab,ab->"(c,YR)[] ein"ab,ab->"(YL,ɔ)[] ein"ab,ab->"(c,Ǝ[:,3,:])[] ein"ab,ab->"(E[:,1,:],ɔ)[] 
    # @show ein"(abc,ce),(ad,dbe)->"(A,Ǝ[:,3,:],c,conj(A))[]
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
function einLB(k, L, B, A, E, M, Ǝ)
    LB, info = linsolve(LB->LB - exp(1.0im * k) * ein"((adfij,abcij),dgebij),fghij -> cehij"(LB,A,M,conj(A)) + exp(1.0im * k) * ein"abcij,abcij->"(LB,Ǝ)[]*E, ein"((adfij,abcij),dgebij),fghij -> cehij"(L,B,M,conj(A)))
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
function einRB(k, R, B, A, E, M, Ǝ)
    RB, info = linsolve(RB->RB - exp(1.0im *-k) * ein"((abcij,cehij),dgebij),fghij -> adfij"(A,RB,M,conj(A)) + exp(1.0im *-k) * ein"abcij,abcij->"(E,RB)[]*Ǝ, ein"((abcij,cehij),dgebij),fghij -> adfij"(B,R,M,conj(A)))
    @assert info.converged == 1
    return RB
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
    HB  = eindB(Bu, E, M, Ǝ)
 
    # 2. B and dB on different sites of M
    HB += eindB(A, einLB(k, E, Bu, A, E, M, Ǝ), M, Ǝ) * exp(1.0im * k) +
          eindB(A, E, M, einRB(k, Ǝ, Bu, A, E, M, Ǝ)) * exp(1.0im *-k)

    return HB
end

"""
    excitation_spectrum(k, A, H, n)

find at least `n` smallest excitation gaps 
"""
function excitation_spectrum_MPO(k, model, n::Int = 1;
                             gs_from = "c",
                             Ni::Int = 1,
                             Nj::Int = 1,
                             χ::Int,
                             atype = Array,
                             infolder = "./data/")

    infolder = joinpath(infolder, "$model")
    M = atype(MPO(model))
    D = size(M, 2)
    MM= zeros(ComplexF64, (size(M)...,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        MM[:,:,:,:,i,j] = M
    end

    if gs_from == "c"
        A, _ = init_canonical_mps(;infolder = infolder,
                                atype = atype,
                                Ni = Ni,
                                Nj = Nj,
                                D = D,
                                χ = χ)
        println("load canonical mps")
    else
        A = init_mps(D = D, χ = χ,
                     infolder = infolder)
        
        A = reshape(A, χ,D,χ,1,1)
        println("load uniform mps")
    end

    c, ɔ     = env_norm!(A)
    E, Ǝ     = envir_MPO(A, c, ɔ, M)
    sq_c,sq_ɔ,inv_sq_c,inv_sq_ɔ = [similar(c) for _ in 1:4]
    for j in 1:Nj, i in 1:Ni
        sq_c[:,:,i,j]     = sqrt(c[:,:,i,j])
        sq_ɔ[:,:,i,j]     = sqrt(ɔ[:,:,i,j])
        inv_sq_c[:,:,i,j] = sq_c[:,:,i,j]^-1
        inv_sq_ɔ[:,:,i,j] = sq_ɔ[:,:,i,j]^-1
    end

    VL       = initial_VL(A, c)

    # X = zeros(ComplexF64, χ*(D-1), χ)
    X = rand(ComplexF64, χ*(D-1), χ, Ni, Nj)
    # X[1] = 1.0
    # X /= sqrt(ein"ab,ab->"(X,conj(X))[])
    
    function f(X)
        Bu = ein"((baij,bcdij),deij),efij->acfij"(inv_sq_c, VL, X, inv_sq_ɔ)
        HB = H_MPO_eff(k, A, Bu, E, MM, Ǝ)
        HB = ein"((baij,bcdij),acfij),deij->feij"(inv_sq_c,HB,conj(VL),inv_sq_ɔ)
        return HB
    end
    Δ, Y, info = eigsolve(x -> f(x), X, n, :SR; ishermitian = true, maxiter = 100)
    # @assert info.converged == 1
    Δ .-= real(ein"(((adfij,abcij),dgebij),cehij),fghij -> "(E,A,MM,Ǝ,conj(A))[])
    return Δ, Y, info
end