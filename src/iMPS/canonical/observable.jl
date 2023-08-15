export load_canonical_excitaion
export energy_gs_canonical_MPO
export spectral_weight, spectral_weight_dimer, correlation_length, spin_config, SS_correlation, DD_correlation, dimer_order

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
function ω(W, k, AC, AL, AR, S, B, ƆLL)
    # 1. AS and B on the same site
    ωk = ein"(abdij,cb),acdij->"(AC,S,conj(B))

    # 2. AS and B on different sites of M
    CS = einCS(W, k, AL, S, ƆLL)
    ωk += ein"(abcij,adij),dbcij->"(AC,CS,conj(B))
          
    return abs2(Array(ωk)[])
end

"""
    ```
     ┌───AL───┬─           ┌────┬────c
     │   │    │            │    b    
     │   S    s2           a    │    
     │   │    │            │    d    
     └───AL*──┴─           └────┴────e 
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
    ω_m = ω2(W, k, AL, AR, S, Bu)

    get the spectral weight `|<Ψₖ(B)|SₖSₖ|Ψₖ(A)>|²`
    ```
    1. S1 and S2 are next to each other and on the same two sites
        ┌───AC──AR──┐     ┌───AC──AR──┐           ┌───┬─c─┬───┐
        │   │   │   │     │   │   │   │           │   b   d   │
        │   S1  S2  │  +  │   S1  S2  │           a   │   │   e
        │   │   │   │     │   │   │   │           │   f   h   │
        └───B*──AR──┘     └───AL──B*──┘           └───┴─g─┴───┘
    
    2. S1 and S2 next to each other and on the different sites
        ┌───AL──AL──┬───AC──┐
        │   │   │   │   │   │
        │   S1  S2  s2  │   │
        │   │   │   │   │   │
        └───AL*─AL*─┴───B*──┘

        s2 = sum of `eⁱᵏ 工` series: 
                                                                             
             ───         ─┬─             ─┬─┬─           ─┬─┬─┬─               ─┬─┬─...─┬─ 
        eⁱ⁰ᵏ      +  eⁱ¹ᵏ │    +    eⁱ²ᵏ  │ │    +  eⁱ³ᵏ  │ │ │   ... +   eⁱⁿᵏ  │ │     │   ...
             ───         ─┴─             ─┴─┴─           ─┴─┴─┴─               ─┴─┴─...─┴─ 

    ```

"""
function ω2(W, k, AC, AL, AR, S, B, ƆLL)
    S1, S2 = S
    kx, ky = k
    # 1. S1 and S2 are next to each other and on the same two sites
    ωk = ein"((((abcij,fb),afgij),cdeij),hd),gheij->"(AC,S1,conj(B),AR,S2,conj(AR)) +
         ein"((((abcij,fb),afgij),cdeij),hd),gheij->"(AC,S1,conj(AL),AR,S2,conj(B)) * exp(1.0im * ky)

    # S1 and S2 next to each other and on the different sites
    CSS = einCSS(W, k, AL, S, ƆLL)
    ωk += ein"(abcij,adij),dbcij->"(AC,CSS,conj(B))
          
    return abs2(Array(ωk)[])
end

"""
    ```
     ┌───AL──AL──┬─        ┌───┬─d─┬──f
     │   │   │   │         │   b   e 
     │   S1  S2  s2        a   │   │ 
     │   │   │   │         │   g   k 
     └───AL*─AL*─┴─        └───┴─h─┴──l 
    ```
"""
function einCSS(W, k, AL, S, Ɔ)
    kx, ky = k
    S1, S2 = S
    C工工  = ein"((((abdij,gb),aghij),defij),ke),hklij->flij"(AL, S1, conj(AL), AL, S2, conj(AL))
    coef = series_coef_L(k, W)
    C工工s = sum(collect(Iterators.take(iterated(x->C工map(x, AL, AL), C工工), W)) .* circshift(coef,-1))
    χ = size(AL,1)
    C = _arraytype(AL)(reshape(I(χ), (χ,χ,1,1)))
    CSS, info = linsolve(CSS->CSS - exp(1.0im * kx) * nth(iterated(x->C工map(x, AL, AL), CSS), W+1) + exp(1.0im * kx) * ein"(abij,abij),cdij->cdij"(CSS, Ɔ, C), C工工s)
    @assert info.converged == 1
    return CSS
end

"""
    ω_m = ω3(W, k, AL, AR, S, Bu)

    get the spectral weight `|<Ψₖ(B)|SₖSₖ|Ψₖ(A)>|²`
    ```
    1. S1 and S2 keep away from W site next to each other and on the different sites

        ┌───AC─...─AR───┐
        │   │      │    │
        │   S1 ... S2   │   (W terms)
        │   │      │    │
        └───B*─...─AR*──┘

    2. S1 and S2 next to each other and on the different sites

        ┌───AL──...─AL──┬───AC──┐
        │   │       │   │   │   │
        │   S1  ... S2  s2  │   │
        │   │       │   │   │   │
        └───AL*─...─AL*─┴───B*──┘    

        s2 = sum of `eⁱᵏ 工` series: 
                                                                             
             ───         ─┬─             ─┬─┬─           ─┬─┬─┬─               ─┬─┬─...─┬─ 
        eⁱ⁰ᵏ      +  eⁱ¹ᵏ │    +    eⁱ²ᵏ  │ │    +  eⁱ³ᵏ  │ │ │   ... +   eⁱⁿᵏ  │ │     │   ...
             ───         ─┴─             ─┴─┴─           ─┴─┴─┴─               ─┴─┴─...─┴─ 

    ```

"""
function ω3(W, k, AC, AL, AR, S, B, ƆLL)
    S1, S2 = S
    coef = series_coef_L(k, W)

    # 1. S1 and S2 keep away from W site next to each other and on the different sites
    ωk_r = ein"(abcij,db),edcij->aeij"(AR,S2,conj(AR))
    ωk_r = nth(iterated(x->工Ɔmap(x, AR, AR), ωk_r), W-1)
    ωk = ein"((abcij,db),adeij),ceij->"(AC,S1,conj(B),ωk_r)
    for i in 2:W-1
        ωk_l = ein"(abcij,db),adeij->ceij"(AL,S1,conj(AL))
        ωk_l = nth(iterated(x->C工map(x, AL, AL), ωk_l), i-1)
        ωk_r = ein"(abcij,db),edcij->aeij"(AR,S2,conj(AR))
        ωk_r = nth(iterated(x->工Ɔmap(x, AR, AR), ωk_r), W-i)
        ωk += ein"((adij,abcij),dbeij),ceij->"(ωk_l,AC,conj(B),ωk_r) * coef[i-1]
    end
    ωk_l = ein"(abcij,db),adeij->ceij"(AL,S1,conj(AL))
    ωk_l = nth(iterated(x->C工map(x, AL, AL), ωk_l), W-1)
    ωk += ein"((aeij,abcij),db),edcij->"(ωk_l,AC,S2,conj(B)) * coef[W-1]

    # 2. S1 and S2 next to each other and on the different sites
    CSAS = einCSAS(W, k, AL, S, ƆLL)
    ωk += ein"(abcij,adij),dbcij->"(AC,CSAS,conj(B))
          
    return abs2(Array(ωk)[])
end

"""
    ```
     ┌───AL──...─AL──┬──         a────┬────c
     │   │       │   │           │    b    
     │   S1  ... S2  s2          │    │     
     │   │       │   │           │    e      
     └───AL*─...─AL*─┴──         d────┴────f  
    ```
"""
function einCSAS(W, k, AL, S, Ɔ)
    kx, ky = k
    S1, S2 = S
    C工 = ein"(abcij,db),adeij->ceij"(AL,S1,conj(AL))
    C工 = nth(iterated(x->C工map(x, AL, AL), C工), W-1)
    C工  = ein"((adij,abcij),eb),defij->cfij"(C工,AL,S2,conj(AL))
    coef = series_coef_L(k, W)
    C工s = sum(collect(Iterators.take(iterated(x->C工map(x, AL, AL), C工), W)) .* coef) * exp(1.0im * kx)
    χ = size(AL,1)
    C = _arraytype(AL)(reshape(I(χ), (χ,χ,1,1)))
    CSAS, info = linsolve(CSAS->CSAS - exp(1.0im * kx) * nth(iterated(x->C工map(x, AL, AL), CSAS), W+1) + exp(1.0im * kx) * ein"(abij,abij),cdij->cdij"(CSAS, Ɔ, C), C工s)
    @assert info.converged == 1
    return CSAS
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

    get the spectral weight `|<Ψₖ(B)|Sₖ|Ψ(A)>|²`
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

    _, ƆLL = env_ɔ(AL, conj(AL))

    AC = ALCtoAC(AL, C)
    S_4s = atype.(S_4site(model, k))
    kx, ky = k
    W = model.W
    k_config = Float64.([kx, ky])
    if if4site
        k_config[1] > pi/2 && (k_config[1] = pi-k_config[1])
        k_config[2] > pi/2 && (k_config[2] = pi-k_config[2])
        k_config *= 2
    end
    VL, X = load_canonical_excitaion(infolder, model, Nj, D2, χ, k_config)
    VL = atype(VL)
    norm(ein"abcij,abdij->cdij"(VL, conj(AL))) < 1e-10 || error("VL and AL are not orthogonal")
    X  = atype.(X)
    m = min(m, length(X))
    ωk = zeros(Float64, m, 3)
    for i in 1:m
        B = ein"abcij,cdij->abdij"(VL, X[i])
        ωk[i, :] = [ω(W, k_config, AC, AL, AR, S, B, ƆLL) for S in S_4s]
    end
    filepath = joinpath(outfolder, "$model", "canonical/Nj$(Nj)_D$(D2)_χ$(χ)/")
    if4site && (W *= 2)
    logfile = open("$filepath/spectral_weight_kx$(round(Int,kx/pi*W/2))_ky$(round(Int,ky/pi*W/2)).log", "w")
    write(logfile, "$(ωk)")
    close(logfile)
    println("save spectral weight to $logfile")

    return ωk
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
function SS_4site(model, k)
    S = model.S
    kx, ky = k
    Sα = const_Sx(S), const_Sy(S), const_Sz(S)
    d = Int(2*S + 1)
    Id = I(d)
    [(contract4([S,S,Id,Id]) + contract4([S,Id,S,Id]) + exp(1.0im * kx) *contract4([Id,S,Id,S]) + exp(1.0im * ky) * contract4([Id,Id,S,S])) for S in Sα]
end

function SS_4site_v(model, k)
    S = model.S
    kx, ky = k
    Sα = const_Sx(S), const_Sy(S), const_Sz(S)
    d = Int(2*S + 1)
    Id = I(d)
    [[exp(1.0im * ky) * contract4([Id,Id,S,Id])/2, exp(2.0im * ky) * contract4([S,Id,Id,Id])/2] for S in Sα], 
    [[exp(1.0im * kx + 1.0im * ky) * contract4([Id,Id,Id,S])/2, exp(1.0im * kx + 2.0im * ky) * contract4([Id,S,Id,Id])/2] for S in Sα]
end

function SS_4site_h(model, k)
    S = model.S
    kx, ky = k
    Sα = const_Sx(S), const_Sy(S), const_Sz(S)
    d = Int(2*S + 1)
    Id = I(d)
    [[exp(1.0im * kx) * contract4([Id,S,Id,Id])/2, contract4([S,Id,Id,Id])/2] for S in Sα], 
    [[exp(1.0im * kx + 1.0im * ky) * contract4([Id,Id,Id,S])/2, contract4([Id,Id,S,Id])/2] for S in Sα]
end

"""

    get the dimer spectral weight `|<Ψₖ(B)|SₖSₖ|Ψ(A)>|²`
"""
function spectral_weight_dimer(model, k, m; Nj, χ, infolder, outfolder, atype, ifmerge, if4site)
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

    _, ƆLL = env_ɔ(AL, conj(AL))

    AC = ALCtoAC(AL, C)
    S_4s = atype.(SS_4site(model, k))
    S_4v1s, S_4v2s = SS_4site_v(model, k)
    # S_4h1s, S_4h2s = SS_4site_h(model, k)

    kx, ky = k
    W = model.W
    k_config = Float64.([kx, ky])
    if if4site
        k_config[1] > pi/2 && (k_config[1] = pi-k_config[1])
        k_config[2] > pi/2 && (k_config[2] = pi-k_config[2])
        k_config *= 2
    end
    VL, X = load_canonical_excitaion(infolder, model, Nj, D2, χ, k_config)
    VL = atype(VL)
    norm(ein"abcij,abdij->cdij"(VL, conj(AL))) < 1e-10 || error("VL and AL are not orthogonal")
    X  = atype.(X)
    m = min(m, length(X))
    ωk = zeros(Float64, m, 3)
    for i in 1:m
        B = ein"abcij,cdij->abdij"(VL, X[i])
        ωk[i, :] = [ω(W, k_config, AC, AL, AR, S, B, ƆLL) for S in S_4s]
                #    ([ω2(W, k_config, AC, AL, AR, atype.(S), B, ƆLL) for S in S_4v1s] + 
                #    [ω2(W, k_config, AC, AL, AR, atype.(S), B, ƆLL) for S in S_4v2s])
                #    [ω3(W, k_config, AC, AL, AR, atype.(S), B, ƆLL) for S in S_4h1s] + 
                #    [ω3(W, k_config, AC, AL, AR, atype.(S), B, ƆLL) for S in S_4h2s]  ## discard the horizontal part because it's too small for small 2D correlation length
    end
    filepath = joinpath(outfolder, "$model", "canonical/Nj$(Nj)_D$(D2)_χ$(χ)/")
    if4site && (W *= 2)
    logfile = open("$filepath/spectral_weight_dimer_kx$(round(Int,kx/pi*W/2))_ky$(round(Int,ky/pi*W/2)).log", "w")
    write(logfile, "$(ωk)")
    close(logfile)
    println("save dimer spectral weight to $logfile")

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

    AC = ALCtoAC(AL, C)
    outfolder = joinpath(groundstate_folder,"1x$(Nj)_D$(D2)_χ$χ")
    !isdir(outfolder) && mkpath(outfolder)
    if if4site             
        ISx, ISy, ISz = atype.(I_S(Sx)), atype.(I_S(Sy)), atype.(I_S(Sz))

        Sx_s = [real(Array(ein"abcij,db,adcij ->"(AC,ISx[i],conj(AC))))[] for i in 1:4]
        Sy_s = [real(Array(ein"abcij,db,adcij ->"(AC,ISy[i],conj(AC))))[] for i in 1:4]
        Sz_s = [real(Array(ein"abcij,db,adcij ->"(AC,ISz[i],conj(AC))))[] for i in 1:4]


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
    else
        Sx_s = real(Array(ein"abcij,db,adcij ->ij"(AC,atype(Sx),conj(AC))))
        Sy_s = real(Array(ein"abcij,db,adcij ->ij"(AC,atype(Sy),conj(AC))))
        Sz_s = real(Array(ein"abcij,db,adcij ->ij"(AC,atype(Sz),conj(AC))))

        logfile = open("$outfolder/spin_config.log", "w")
        message =
"
Sx:
$(Sx_s)
Sy:
$(Sy_s)
Sz:
$(Sz_s)
"
    end
            
    write(logfile, message)
    close(logfile)
    println("save spin_config to $logfile")
    println(message)
end

"""
    mag2(model, k;
       Nj = 1, χ, 
       infolder = Defaults.infolder,
       atype = Defaults.atype,
       ifmerge = false, 
       if4site = true
       )

    mag2 = <Si⋅Sj>exp(ik⋅(i-j))

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
function mag2(model, k; 
              Nj = 1, χ, 
              infolder = Defaults.infolder,
              atype = Defaults.atype,
              ifmerge = false, 
              if4site = true
              )

    # Todo: this is not correct because ij is the whole site index, not the nearest neighbor index
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
                  
    S_4s = atype.(SS_4site(model, k))
    S_4v1s, S_4v2s = SS_4site_v(model, k)
    S_4h1s, S_4h2s = SS_4site_h(model, k)

    AC = ALCtoAC(AL, C)
    Sij(S) = real(Array(ein"abcij,db,adcij ->"(AC,S,conj(AC))))[]
    function Sij2(S)
        ωk_l = ein"(abcij,db),adeij->ceij"(AL,atype(S[1]),conj(AL))
        real(Array(ein"((aeij,abcij),db),edcij->"(ωk_l,AC,atype(S[2]),conj(AC)))[])
    end
    function SijW(S)
        ωk_l = ein"(abcij,db),adeij->ceij"(AL,atype(S[1]),conj(AL))
        ωk_l = nth(iterated(x->C工map(x, AL, AL), ωk_l), model.W-1)
        real(Array(ein"((aeij,abcij),db),edcij->"(ωk_l,AC,atype(S[2]),conj(AC)))[])
    end

    if if4site             
        S_onsite = [Sij(S_4s[i]) for i in 1:3]
        # S_v1     = [Sij2(S_4v1s[i]) for i in 1:3] 
        # S_v2     = [Sij2(S_4v2s[i]) for i in 1:3] 
        # S_h1     = [SijW(S_4h1s[i]) for i in 1:3]
        # S_h2     = [SijW(S_4h2s[i]) for i in 1:3]

        mag2_tol = S_onsite 
        outfolder = joinpath(groundstate_folder,"1x$(Nj)_D$(D2)_χ$χ")
        !isdir(outfolder) && mkpath(outfolder)
        logfile = open("$outfolder/m2.log", "w")
        message = 
"
Sx:
$(mag2_tol[1])
Sy:
$(mag2_tol[2])
Sz:
$(mag2_tol[3])
"
        write(logfile, message)
        close(logfile)
        println("save mag2 to $logfile")
        println(message)

        return mag2_tol
    end
end

"""
    SS_correlation(model, k;
                   Nj = 1, χ, 
                   infolder = Defaults.infolder,
                   atype = Defaults.atype,
                   ifmerge = false, 
                   if4site = true
                   )

    SS = <S₀⋅Sᵣ>
"""
function SS_correlation(model, k, r;
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

    kx, ky = k
    AC = ALCtoAC(AL, C)
    W, S = model.W, model.S
    Id = I(Int(2*S + 1))
    Sα = const_Sx(S), const_Sy(S), const_Sz(S)


    SS = zeros(Float64, r)
    # SS[1] on the same column
    # S_4s = exp(1.0im * kx) * contract4([Sα[1],Sα[1],Id,Id])
    # SS[1] = real(Array(ein"abcij,db,adcij ->"(AC,atype(S_4s),conj(AC)))[])

    # S12 = atype.([contract4([Sα[1],Id,Id,Id]), 
    #               exp(1.0im * kx) * contract4([Id,Sα[1],Id,Id])])
    # SS_l12 = ein"(abcij,db),adeij->ceij"(AL,S12[1],conj(AL))
    # SS_l12 = collect(Iterators.take(iterated(x->C工map(x, AL, AL), SS_l12), floor(Int,r/2) * W))
    
    # for i in 2:r
    #     S2 = i%2==0 ? S12[1] : S12[2]
    #     SS[i] = real(Array(ein"((aeij,abcij),db),edcij->"(SS_l12[floor(Int,i/2) * W],AC,S2,conj(AC)))[])
    # end

    # SS[1] on the different column
    S12 = atype.([contract4([Sα[1],Id,Id,Id]), 
                  exp(1.0im * kx) * contract4([Id,Sα[1],Id,Id])])
    SS_l12 = ein"(abcij,db),adeij->ceij"(AL,S12[2],conj(AL))
    SS_l12 = collect(Iterators.take(iterated(x->C工map(x, AL, AL), SS_l12), floor(Int,(r+1)/2) * W))
    
    for i in 1:r
        S2 = i%2==0 ? S12[2] : S12[1]
        SS[i] = real(Array(ein"((aeij,abcij),db),edcij->"(SS_l12[floor(Int,(i+1)/2) * W],AC,S2,conj(AC)))[])
    end
    return SS
end

"""
    DD_correlation(model, k;
                   Nj = 1, χ, 
                   infolder = Defaults.infolder,
                   atype = Defaults.atype,
                   ifmerge = false, 
                   if4site = true
                   )

    DD = <D₀⋅Dᵣ>-<D₀>⋅<Dᵣ>
"""
function DD_correlation(model, k, r;
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

    kx, ky = k
    AC = ALCtoAC(AL, C)
    W, S = model.W, model.S
    Id = I(Int(2*S + 1))
    Sα = const_Sx(S), const_Sy(S), const_Sz(S)

    DD = zeros(Float64, r)
    D12 = atype(contract4([Sα[1],Sα[1],Id,Id]))
    DD12 = real(Array(ein"abcij,db,adcij ->"(AC,D12,conj(AC))))[]^2

    DD_l12 = ein"(abcij,db),adeij->ceij"(AL,D12,conj(AL))
    DD_l12 = collect(Iterators.take(iterated(x->C工map(x, AL, AL), DD_l12), floor(Int,r) * W))
    
    for i in 1:r
        DD[i] = real(Array(ein"((aeij,abcij),db),edcij->"(DD_l12[floor(Int,i) * W],AC,D12,conj(AC)))[])
    end
    return DD.-DD12
end

"""
    dimer_order(model, k;
                   Nj = 1, χ, 
                   infolder = Defaults.infolder,
                   atype = Defaults.atype,
                   ifmerge = false, 
                   if4site = true
                   )
site label
```
    │  │  │  │ 
   ─3──4──3──4─
    │  │  │  │     ←--- A
   ─1──2──1──2─         |
    │  │  │  │          |
   ─3──4──3──4─         ↓
    │  │  │  │     
   ─1──2──1──2─
    │  │  │  │
```
"""
function dimer_order(model;
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

    AC = ALCtoAC(AL, C)
    W, S = model.W, model.S
    Id = I(Int(2*S + 1))
    Sα = const_Sx(S), const_Sy(S), const_Sz(S)

    S21 = atype(sum([contract4([S,S,Id,Id]) for S in Sα]))
    S31 = atype(sum([contract4([S,Id,S,Id]) for S in Sα]))
    S1 = [atype(contract4([S,Id,Id,Id])) for S in Sα]
    S2 = [atype(contract4([Id,S,Id,Id])) for S in Sα]
    S3 = [atype(contract4([Id,Id,S,Id])) for S in Sα]
    # S24 = atype(sum([contract4([Id,S,Id,S]) for S in Sα]))
    # S34 = atype(sum([contract4([Id,Id,S,S]) for S in Sα]))


    SS21 = real(Array(ein"abcij,db,adcij ->"(AC,S21,conj(AC))))[]
    SS31 = real(Array(ein"abcij,db,adcij ->"(AC,S31,conj(AC))))[]

    SS_l12 = [ein"(abcij,db),adeij->ceij"(AL,S,conj(AL)) for S in S1]
    SS_l12 = [nth(iterated(x->C工map(x, AL, AL), SS), W) for SS in SS_l12]
    SS12 = sum([real(Array(ein"((aeij,abcij),db),edcij->"(SS,AC,S,conj(AC)))[]) for (SS,S) in zip(SS_l12, S2)])

    SS_l13 = [ein"(abcij,db),adeij->ceij"(AL,S,conj(AL)) for S in S1]
    SS13 = sum([real(Array(ein"((aeij,abcij),db),edcij->"(SS,AC,S,conj(AC)))[]) for (SS,S) in zip(SS_l13, S3)])
    outfolder = joinpath(groundstate_folder,"1x$(Nj)_D$(D2)_χ$χ")
    !isdir(outfolder) && mkpath(outfolder)
    logfile = open("$outfolder/dimer_order.log", "w")
    message = 
"
SS21: $SS21
SS31: $SS31
SS12: $SS12
SS13: $SS13
"
    write(logfile, message)
    close(logfile)
    println("save dimer_order to $logfile")
    println(message)
end