export load_canonical_excitaion
export energy_gs_canonical_MPO
export spectral_weight

function load_canonical_excitaion(infolder, model, Nj, D, χ, k)
    kx, ky = k
    W = model.W
    infolder = joinpath(infolder, "$model")
    filepath = joinpath(infolder, "canonical/Nj$(Nj)_D$(D)_χ$(χ)/")
    chkp_file = "$filepath/excitaion_VLX_kx$(round(Int,kx/pi*W/2))_ky$(round(Int,ky/pi*W/2)).jld2"
    println("load canonical excitaion from $chkp_file")
    load(chkp_file)["VLX"]
end

function energy_gs_canonical_MPO(model; Nj, χ, infolder, atype, ifmerge, if4site)
    M, AL, C, AR, AC, ELL, ƎRR, ERL, ƎRL, ELR, ƎLR, VL = 
        canonical_exci_env(model, Nj,χ; 
                           infolder=infolder, atype=atype, 
                           ifmerge=ifmerge, if4site=if4site)
    e2 = ein"(((adfij,abcij),dgebij),cehij),fghij -> "(ELL,AC,M,ƎRR,conj(AC))[]
    e1 = ein"((abij,acdij),bceij),deij->"(C,ELL,ƎRR,conj(C))[]
    e = e2-e1
    ifmerge && (e /= 2)
    if4site && (e /= 4)
    return e
end

"""
    ω_m = ω(W, k, AL, AR, S, Bu)

    get the spectral weight `|<Ψₖ(B)|Sₖ|Ψₖ(A)>|²`
    ```
    1. AS and B on the same site
        ┌───AC──┐
        │   │   │
        │   S   │
        │   │   │
        └───B*──┘
    
    2. AS and B on different sites
        ┌───AL──┬───AC──┐
        │   │   │   │   │
        │   S   s2  │   │
        │   │   │   │   │
        └───AL*─┴───B*──┘

        ┌───AC──┬───AR──┐
        │   │   │   │   │
        │   │   s3  S   │
        │   │   │   │   │
        └───B*──┴───AR*─┘

        s2 = sum of `eⁱᵏ 工` series: 
                                                                             
             ───         ─┬─             ─┬─┬─           ─┬─┬─┬─               ─┬─┬─...─┬─ 
        eⁱ⁰ᵏ      +  eⁱ¹ᵏ │    +    eⁱ²ᵏ  │ │    +  eⁱ³ᵏ  │ │ │   ... +   eⁱⁿᵏ  │ │     │   ...
             ───         ─┴─             ─┴─┴─           ─┴─┴─┴─               ─┴─┴─...─┴─ 

    ```

"""
function ω(W, k, AC, AL, AR, S, B, ƆLL, CRR)
    # 1. AS and B on the same site
    ωk = ein"(abdij,cb),acdij->"(AC,S,conj(B))

    # 2. AS and B on different sites of M
    CS = einCS(W, k, AL, S, ƆLL)
    SƆ = einSƆ(W, k, AR, S, CRR)
    ωk += ein"(abcij,deij),abdij->"(AC,SƆ,conj(B)) + 
          ein"(abcij,adij),dbcij->"(AC,CS,conj(B))
          
    return abs2(Array(ωk)[])
end

"""
    ```
     ┌───AL───┬─           ┌────┬──── c
     │   │    │            │    b    
     │   S    s2           a    │    
     │   │    │            │    d    
     └───AL*──┴─           └────┴──── e 
    ```
"""
function einCS(W, k, AL, S, Ɔ)
    kx, ky = k
    C工 = ein"(abcij,db),adeij->ceij"(AL, S, conj(AL))
    coef = series_coef_L(k, W)
    C工s = sum(collect(Iterators.take(iterated(x->C工map(x, AL, AL), C工), W)) .* coef)
    χ = size(AL,1)
    C = _arraytype(AL)(reshape(I(χ), (χ,χ,1,1)))
    CS, info = linsolve(CS->CS - exp(1.0im * kx) * nth(iterated(x->C工map(x, AL, AL), CS), W+1) + exp(1.0im * kx) * ein"(abij,abij),cdij->cdij"(CS, Ɔ, C), C工s)
    @assert info.converged == 1
    return CS
end

"""
    ```
    ─┬───AR──┐          a ────┬────┐
     │   │   │                b    │
     s3  S   │                │    c
     │   │   │                d    │
    ─┴───AR*─┘          e ────┴────┘
    ```
"""
function einSƆ(W, k, AR, S, C)
    kx, ky = k
    工Ɔ = ein"(abcij,db),edcij->aeij"(AR, S, conj(AR))
    coef = series_coef_R(k, W)
    工Ɔs = sum(collect(Iterators.take(iterated(x->工Ɔmap(x, AR, AR), 工Ɔ), W)) .* coef)
    χ = size(AR,1)
    Ɔ = _arraytype(AR)(reshape(I(χ), (χ,χ,1,1)))
    SƆ, info = linsolve(SƆ->SƆ - exp(-1.0im * kx) * nth(iterated(x->工Ɔmap(x, AR, AR), SƆ), W+1) + exp(-1.0im * kx) * ein"(abij,abij),cdij->cdij"(C, SƆ, Ɔ), 工Ɔs)
    @assert info.converged == 1
    return SƆ
end

function S_4site(model, k)
    S = model.S
    kx, ky = k
    Sα = const_Sx(S), const_Sy(S), const_Sz(S)
    d = Int(2*S + 1)
    Id = I(d)
    [(contract4([S,Id,Id,Id]) + exp(1.0im * ky) * contract4([Id,S,Id,Id]) + exp(1.0im * kx) *contract4([Id,Id,S,Id]) + exp(1.0im * kx + 1.0im * ky) * contract4([Id,Id,Id,S]))/4 for S in Sα]
end

function spectral_weight(model, k, m; Nj, χ, infolder, outfolder, atype, ifmerge, if4site)
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
    AL, C, AR = init_canonical_mps(;infolder = joinpath(infolder, "$model", "groundstate"), 
                                    atype = atype,  
                                    Nj = Nj,      
                                    D = D2, 
                                    χ = χ)
    if ifmerge
        AL = reshape(ein"abc,cde->abde"(AL[:,:,:,1,1], AL[:,:,:,1,2]), (χ, D2^2, χ, 1, 1))
        AR = reshape(ein"abc,cde->abde"(AR[:,:,:,1,1], AR[:,:,:,1,2]), (χ, D2^2, χ, 1, 1))
        C = reshape(C[:,:,1,2], (χ, χ, 1, 1))
    end

    _, CRR = env_c(AR, conj(AR))
    _, ƆLL = env_ɔ(AL, conj(AL))

    AC = ALCtoAC(AL, C)
    S_4s = atype.(S_4site(model, k))
    kx, ky = k
    W = model.W
    k_config = [k[1], k[2]]
    if if4site
        k_config[1] > pi/2 && (k_config[1] = pi-k_config[1])
        k_config[2] > pi/2 && (k_config[2] = pi-k_config[2])
        k_config *= 2
    end
    VL, X = load_canonical_excitaion(infolder, model, Nj, D2, χ, k_config)
    VL = atype(VL)
    X  = atype.(X)
    ωk = zeros(Float64, m)
    for i in 1:m
        B = ein"abcij,cdij->abdij"(VL, X[i])
        ωk[i] = sum([ω(W, k_config, AC, AL, AR, S, B, ƆLL, CRR) for S in S_4s])
    end
    filepath = joinpath(outfolder, "$model", "canonical/Nj$(Nj)_D$(D2)_χ$(χ)/")
    if4site && (W *= 2)
    logfile = open("$filepath/spectral_weight_kx$(round(Int,kx/pi*W/2))_ky$(round(Int,ky/pi*W/2)).log", "w")
    write(logfile, "$(ωk)")
    close(logfile)
    println("save spectral weight to $logfile")

    return ωk
end