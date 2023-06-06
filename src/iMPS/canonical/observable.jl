export load_canonical_excitaion
export energy_gs_canonical_MPO
export spectral_weight, correlation_length, spin_config, S2_total

function load_canonical_excitaion(infolder, model, Nj, D, χ, k)
    kx, ky = k
    W = model.W
    infolder = joinpath(infolder, "$model")
    filepath = joinpath(infolder, "canonical/Nj$(Nj)_D$(D)_χ$(χ)/")
    chkp_file = W==1 ? "$filepath/excitaion_VLX_k$(round(kx/pi, digits=8)).jld2" : "$filepath/excitaion_VLX_kx$(round(Int,kx/pi*W/2))_ky$(round(Int,ky/pi*W/2)).jld2"
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
        │   │   s3  S   │       (removed by gauge)
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
    ωk += ein"(abcij,adij),dbcij->"(AC,CS,conj(B))
          
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
    │  │ 
   ─3──4─
    │  │  
   ─1──2─
    │  │ 
```
"""
function S_4site(model, k)
    S = model.S
    kx, ky = k
    Sα = const_Sx(S), const_Sy(S), const_Sz(S)
    d = Int(2*S + 1)
    Id = I(d)
    [(contract4([S,Id,Id,Id]) + exp(1.0im * kx) * contract4([Id,S,Id,Id]) + exp(1.0im * ky) *contract4([Id,Id,S,Id]) + exp(1.0im * kx + 1.0im * ky) * contract4([Id,Id,Id,S]))/4 for S in Sα]
end

"""

    get the spectral weight `|<Ψₖ(B)|Sₖ|Ψₖ(A)>|²`
"""
function spectral_weight(model, k, m; Nj, χ, infolder, outfolder, atype, ifmerge, if4site)
    Mo = if4site ? atype(MPO_2x2(model)) : atype(MPO(model))
    D2 = size(Mo, 2)

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

function S2_1site(model)
    S = model.S
    Sα = const_Sx(S), const_Sy(S), const_Sz(S)
    # sum([ein"ab,bc->ca"(S,S) for S in Sα])
    # ein"ab,bc->ac"(Sα[3],Sα[3])
    Sα[3]
end

function S2_total(model, k, m; Nj, χ, infolder, outfolder, atype, ifmerge, if4site)
    Mo = if4site ? atype(MPO_2x2(model)) : atype(MPO(model))
    D2 = size(Mo, 2)
    
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
    S2 = if4site ? atype(S2_4site(model, k)) : atype(S2_1site(model))
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
    S2_t = zeros(Float64, m)
    for i in 1:m
        B = ein"abcij,cdij->abdij"(VL, X[i])
        S2_t[i] = ω(W, k_config, AC, AL, AR, S2, B, ƆLL, CRR)
    end
    filepath = joinpath(outfolder, "$model", "canonical/Nj$(Nj)_D$(D2)_χ$(χ)/")
    if4site && (W *= 2)
    logfile = W==1 ? open("$filepath/S2_total_k$(round(kx/pi, digits=8)).log", "w") : open("$filepath/S2_total_kx$(round(Int,kx/pi*W/2))_ky$(round(Int,ky/pi*W/2)).log", "w")
    write(logfile, "$(S2_t)")
    close(logfile)
    println("save S2 total to $logfile")

    return S2_t
end

function correlation_length(model; Nj, χ, infolder, outfolder, atype, ifmerge, if4site)
    Mo = if4site ? atype(MPO_2x2(model)) : atype(MPO(model))
    D2 = size(Mo, 2)

    groundstate_folder = joinpath(infolder, "$model", "groundstate")
    AL, C, AR = init_canonical_mps(;infolder = groundstate_folder, 
                                    atype = atype,  
                                    Nj = Nj,      
                                    D = D2, 
                                    χ = χ)

    if ifmerge
        AL = reshape(ein"abc,cde->abde"(AL[:,:,:,1,1], AL[:,:,:,1,2]), (χ, D2^2, χ, 1, 1))
        AR = reshape(ein"abc,cde->abde"(AR[:,:,:,1,1], AR[:,:,:,1,2]), (χ, D2^2, χ, 1, 1))
        C = reshape(C[:,:,1,2], (χ, χ, 1, 1))
    end

    outfolder = joinpath(groundstate_folder,"1x$(Nj)_D$(D2)_χ$χ")
    !isdir(outfolder) && mkpath(outfolder)
    env_c(AL, conj(AL); 
          ifcor_len=true, 
          outfolder=outfolder)
    # env_ɔ(AR, conj(AR);
    #       ifcor_len=true, 
    #       outfolder=outfolder)
end

"""
    spin_config(spin_config(model; 
                Nj = 1, χ, 
                infolder = Defaults.infolder,
                atype = Defaults.atype,
                ifmerge = false, 
                if4site = true
                )
site label
```
    │  │ 
   ─3──4─
    │  │  
   ─1──2─
    │  │ 
```
```
"""
function spin_config(model; 
                     Nj = 1, χ, 
                     infolder = Defaults.infolder,
                     atype = Defaults.atype,
                     ifmerge = false, 
                     if4site = true
                     )

    Mo = if4site ? atype(MPO_2x2(model)) : atype(MPO(model))
    D2 = size(Mo, 2)

    groundstate_folder = joinpath(infolder, "$model", "groundstate")
    AL, C, AR = init_canonical_mps(;infolder = groundstate_folder, 
                                    atype = atype,  
                                    Nj = Nj,      
                                    D = D2, 
                                    χ = χ)

    if ifmerge
        AL = reshape(ein"abc,cde->abde"(AL[:,:,:,1,1], AL[:,:,:,1,2]), (χ, D2^2, χ, 1, 1))
        AR = reshape(ein"abc,cde->abde"(AR[:,:,:,1,1], AR[:,:,:,1,2]), (χ, D2^2, χ, 1, 1))
        C = reshape(C[:,:,1,2], (χ, χ, 1, 1))
    end

    S = model.S                       
    Sx, Sy, Sz = const_Sx(S), const_Sy(S), const_Sz(S)

    if if4site             
        ISx, ISy, ISz = atype.(I_S(Sx)), atype.(I_S(Sy)), atype.(I_S(Sz))

        Sx_s = [real(Array(ein"(((abcij,cdij),eb),aefij),fdij ->"(AL,C,ISx[i],conj(AL),conj(C))))[] for i in 1:4]
        Sy_s = [real(Array(ein"(((abcij,cdij),eb),aefij),fdij ->"(AL,C,ISy[i],conj(AL),conj(C))))[] for i in 1:4]
        Sz_s = [real(Array(ein"(((abcij,cdij),eb),aefij),fdij ->"(AL,C,ISz[i],conj(AL),conj(C))))[] for i in 1:4]

        outfolder = joinpath(groundstate_folder,"1x$(Nj)_D$(D2)_χ$χ")
        !isdir(outfolder) && mkpath(outfolder)
        logfile = open("$outfolder/spin_config.log", "w")
        message = 
"
Sx:
$(Sx_s[3]) $(Sx_s[4])
$(Sx_s[1]) $(Sx_s[2])
Sy:
$(Sy_s[3]) $(Sy_s[4])
$(Sy_s[1]) $(Sy_s[2])
Sz:
$(Sz_s[3]) $(Sz_s[4])
$(Sz_s[1]) $(Sz_s[2])
"
        write(logfile, message)
        close(logfile)
        println("save spectral weight to $logfile")
        println(message)
    end
end
