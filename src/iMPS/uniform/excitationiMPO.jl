export excitation_spectrum_MPO
using IterTools

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

function envir_MPO(A, M)
    χ,d,_ = size(A)
    W     = size(M, 1)
    atype = _arraytype(A)
    E = atype == Array ? zeros(ComplexF64, χ,W,χ) : CUDA.zeros(ComplexF64, χ,W,χ)
    Ǝ = atype == Array ? zeros(ComplexF64, χ,W,χ) : CUDA.zeros(ComplexF64, χ,W,χ)
    c,ɔ = env_norm(A)

    E[:,W,:] = c
    for i in W-1:-1:1
        YL = atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in i+1:W
            YL += ein"(abc,db),(ae,edf)->cf"(A,M[j,:,i,:],E[:,j,:],conj(A))
        end
        if i == 1 #if M[i,:,i,:] == I(d)
            bL = YL
            E[:,i,:], infoE = linsolve(E->E - ein"abc,(ad,dbe)->ce"(A,E,conj(A)) + ein"(ab,ab),cd->cd"(E,ɔ,c), bL)
            @assert infoE.converged == 1
        else
            E[:,i,:] = YL
        end
    end

    Ǝ[:,1,:] = ɔ
    for i in 2:W
        YR = atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in 1:i-1
            YR += ein"((abc,db),cf),edf->ae"(A,M[i,:,j,:],Ǝ[:,j,:],conj(A))
        end
        if i == W # if M[i,:,i,:] == I(d)
            bR = YR
            Ǝ[:,i,:], infoƎ = linsolve(Ǝ->Ǝ - ein"(abc,ce),dbe->ad"(A,Ǝ,conj(A)) + ein"(ab,ab),cd->cd"(c,Ǝ,ɔ), bR)
            @assert infoƎ.converged == 1
        else
            Ǝ[:,i,:] = YR
        end
    end

    # @show ein"ab,ab->"(c,YR)[] ein"ab,ab->"(YL,ɔ)[] ein"ab,ab->"(c,Ǝ[:,3,:])[] ein"ab,ab->"(E[:,1,:],ɔ)[] 
    # @show ein"(abc,ce),(ad,dbe)->"(A,Ǝ[:,3,:],c,conj(A))[]
    return E, Ǝ
end

function series_coef_L(k, W)
    kx, ky = k
    coef = zeros(ComplexF64, W)
    W_half = Int(ceil(W/2))
    for i in 1:W
        if i < W_half
            coef[i] = exp(i*1.0im*ky)
        elseif i > W_half
            coef[i] = exp(i*1.0im*ky + 1.0im*kx)
        else
            coef[i] = (1 + exp(1.0im*kx))/2 * exp(i*1.0im*ky)
        end
    end
    # no approximation
    # coef = [(W-i + i * exp(1.0im * kx))/W * exp(1.0im * ky * i) for i in 1:W]
    # bad approximation
    # coef = [exp(1.0im * ky * i) for i in 1:W]
    return coef
end

series_coef_R(k, W) = series_coef_L(map(-,k), W)

"""
    ```
     ┌───B────┬─             a ────┬──── c 
     │   │    │              │     b     │ 
     E───M────s─             ├─ d ─┼─ e ─┤ 
     │   │    │              │     g     │ 
     └───A*───┴─             f ────┴──── h  
    ```
"""
function einLB(W, k, L, B, A, E, M, Ǝ)
    kx, ky = k
    EM = EMmap(L, M, B, A)
    coef = series_coef_L(k, W)
    EMs = sum(collect(Iterators.take(iterated(x->EMmap(x, M, A, A), EM), W)) .* coef)
    LB, info = linsolve(LB->LB - exp(1.0im * kx) * nth(iterated(x->EMmap(x, M, A, A), LB), 2*W+1) + exp(1.0im * kx) * ein"(abc,abc),def->def"(LB,Ǝ,E), EMs)
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
function einRB(W, k, R, B, A, E, M, Ǝ)
    kx, ky = k
    MƎ = MƎmap(R, M, B, A)
    coef = series_coef_R(k, W)
    MƎs = sum(collect(Iterators.take(iterated(x->MƎmap(x, M, A, A), MƎ), W)) .* coef)
    RB, info = linsolve(RB->RB - exp(-1.0im * kx) * nth(iterated(x->MƎmap(x, M, A, A), RB), 2*W+1) + exp(-1.0im * kx) * ein"(abc,abc),def->def"(E,RB,Ǝ), MƎs)
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
function H_MPO_eff(W, k, A, Bu, E, M, Ǝ)
    # 1. B and dB on the same site of M
    HB  = eindB(Bu, E, M, Ǝ)
 
    # 2. B and dB on different sites of M
    HB += eindB(A, einLB(W, k, E, Bu, A, E, M, Ǝ), M, Ǝ) +
          eindB(A, E, M, einRB(W, k, Ǝ, Bu, A, E, M, Ǝ))

    return HB
end

"""
    excitation_spectrum(k, A, H, n)
find at least `n` smallest excitation gaps 
"""
function excitation_spectrum_MPO(k, model, n::Int = 1; atype = Array, χ::Int,
                             infolder = "../data/", outfolder = "../data/")
     infolder = joinpath( infolder, "$model")
    outfolder = joinpath(outfolder, "$model")

    W = model.W
    M = atype(MPO(model))
    D = size(M, 2)

    AL, _, _ = init_canonical_mps(;infolder = infolder, 
                                   atype = atype, 
                                   Ni=1,Nj=1,       
                                   D = D,  
                                   χ = χ)
    A = AL[:,:,:,1,1]

    E, Ǝ      = envir_MPO(A, M)
    Ln, Rn    = env_norm(A)
    sq_Ln     = sqrt(Array(Ln))
    sq_Rn     = sqrt(Array(Rn))
    inv_sq_Ln = atype(sq_Ln^-1)
    inv_sq_Rn = atype(sq_Rn^-1)
    VL        = atype(initial_VL(Array(A), Array(Ln)))

    X = atype(randn(ComplexF64, χ*(D-1), χ))
    # X ./= norm(X)
    E0 = real(Array(ein"(((adf,abc),dgeb),ceh),fgh -> "(E,A,M,Ǝ,conj(A)))[])
    function f(X)
        Bu = ein"((ba,bcd),de),ef->acf"(inv_sq_Ln, VL, X, inv_sq_Rn)
        HB = H_MPO_eff(W, k, A, Bu, E, M, Ǝ) - ein"(ad,acb),be ->dce"(Ln, Bu, Rn) * E0
        # HB = H_MPO_eff(W, k, A, Bu, E, M, Ǝ)
        HB = ein"((ba,bcd),acf),de->fe"(inv_sq_Ln,HB,conj(VL),inv_sq_Rn)
        return HB
    end
    Δ, Y, info = eigsolve(x -> f(x), X, n, :SR; ishermitian = true, maxiter = 100)
    info.converged != 1 && @warn("eigsolve doesn't converged")
    # Δ .-= E0
    save_uniform_excitaion(outfolder, W, χ, k, Δ, X)
    # @show Δ
    return Δ, Y, info
end

function save_uniform_excitaion(outfolder, W, χ, k, Δ, X)
    kx, ky = k
    filepath = joinpath(outfolder, "uniform/χ$(χ)/")
    !(ispath(filepath)) && mkpath(filepath)
    logfile = open("$filepath/kx$((kx/pi*W/2))_ky$((ky/pi*W/2)).log", "w")
    write(logfile, "$(Δ)")
    close(logfile)

    out_chkp_file = "$filepath/excitaion_X_kx$((kx/pi*W/2))_ky$((ky/pi*W/2)).jld2"
    save(out_chkp_file, "X", Array(X))
    println("excitaion file saved @$logfile")
end