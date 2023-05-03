using CUDA
CUDA.allowscalar(false)
using Random
using JSON

using TeneT: leftorth, rightorth, LRtoC, ALCtoAC, ACCtoALAR
export vumps


function init_canonical_mps(;infolder = "../data/", 
                             atype = Array, 
                             verbose::Bool = true,  
                             Ni::Int = 1,
                             Nj::Int = 1,      
                             D::Int = 2, 
                             χ::Int = 5,
                             targχ = χ)

    in_chkp_file = joinpath(infolder,"canonical_mps_$(Ni)x$(Nj)_D$(D)_χ$(χ).jld2")
    if isfile(in_chkp_file)
        AL = atype(rand(ComplexF64, targχ,D,targχ,Ni,Nj)) * 1e-6
        AR = atype(rand(ComplexF64, targχ,D,targχ,Ni,Nj)) * 1e-6
            C = atype(rand(ComplexF64, targχ,  targχ,Ni,Nj)) * 1e-6
        AL[1:χ,:,1:χ,:,:], C[1:χ,1:χ,:,:], AR[1:χ,:,1:χ,:,:] = map(atype, load(in_chkp_file)["ALCAR"])
        verbose && println("load canonical mps from $in_chkp_file")
        targχ > χ && println("and increase χ from $(χ) to $(targχ)")
    else
        A = atype(rand(ComplexF64, χ,D,χ,Ni,Nj))
        AL, L, _ = leftorth(A)
        R, AR, _ = rightorth(AL)
        C = LRtoC(L,R)
        verbose && println("random initial canonical mps $in_chkp_file")
    end
    return AL, C, AR
end

function envir_MPO(AL, AR, M)
    atype = _arraytype(M)
    χ,Nx,Ny = size(AL)[[1,4,5]]
    W       = size(M, 1)

    E = atype == Array ? zeros(ComplexF64, χ,W,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,W,χ,Nx,Ny)
    Ǝ = atype == Array ? zeros(ComplexF64, χ,W,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,W,χ,Nx,Ny)
    _, c = env_c(AR, conj(AR))
    _, ɔ = env_ɔ(AL, conj(AL))

    # for y in 1:Ny, x in 1:Nx
    #     ɔ[:,:,x,y] ./= tr(ɔ[:,:,x,y])
    #     c[:,:,x,y] ./= tr(c[:,:,x,y])
    # end

    Iχ = atype(I(χ))
    for y in 1:Ny, x in 1:Nx
        E[:,W,:,x,y] = Iχ
    end
    for i in W-1:-1:1
        YL = atype == Array ? zeros(ComplexF64, χ,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,χ,Nx,Ny)
        for j in i+1:W
            YL += ein"(abcij,dbij),(aeij,edfij)->cfij"(AL,M[j,:,i,:,:,:],E[:,j,:,:,:],conj(AL))
        end
        if i == 1 # if M[i,:,i,:] == I(d)
            # bL = YL - ein"(abij,abij),cdij->cdij"(YL,ɔ,E[:,W,:,:,:]) 
            bL = YL
            E[:,i,:,:,:], infoE = linsolve(X->circshift(X, (0,0,0,1)) - ein"abcij,(adij,dbeij)->ceij"(AL,X,conj(AL)) + ein"(abij,abij),cdij->cdij"(X, ɔ, E[:,W,:,:,:]), bL)
            @assert infoE.converged == 1
        else
            E[:,i,:,:,:] = circshift(YL, (0,0,0,-1))
        end
        # E[:,i,:,:,:] = circshift(YL, (0,0,0,1))
    end

    for y in 1:Ny, x in 1:Nx
        Ǝ[:,1,:,x,y] = Iχ
    end
    for i in 2:W
        YR = atype == Array ? zeros(ComplexF64, χ,χ,Nx,Ny) : CUDA.zeros(ComplexF64, χ,χ,Nx,Ny)
        for j in 1:i-1
            YR += ein"((abcij,dbij),cfij),edfij->aeij"(AR,M[i,:,j,:,:,:],Ǝ[:,j,:,:,:],conj(AR))
        end
        if i == W # if M[i,:,i,:] == I(d)
            # bR = YR - ein"(abij,abij),cdij->cdij"(c,YR,Ǝ[:,1,:,:,:])
            bR = YR
            Ǝ[:,i,:,:,:], infoƎ = linsolve(X->circshift(X, (0,0,0,-1)) - ein"(abcij,ceij),dbeij->adij"(AR,X,conj(AR)) + ein"(abij,abij),cdij->cdij"(c, X, Ǝ[:,1,:,:,:]), bR)
            @assert infoƎ.converged == 1
        else
            Ǝ[:,i,:,:,:] = circshift(YR, (0,0,0,1))
        end
        # Ǝ[:,i,:,:,:] = circshift(YR, (0,0,0,-1))
    end

    return E, Ǝ
end

"""

"""
function vumps(config_file)
    config = JSON.parsefile(config_file)  
    Random.seed!(config["seed"])  
    model = eval(Meta.parse(config["model"]))
    Ni = config["vumps"]["Ni"]
    Nj = config["vumps"]["Nj"]
    χ = config["vumps"]["χ"]
    targχ = config["vumps"]["targχ"]
    iters = config["vumps"]["iters"]
    tol = config["vumps"]["tol"]
    show_every = config["vumps"]["show_every"]
    atype = eval(Meta.parse(config["vumps"]["atype"]))
    if4site = config["vumps"]["if4site"]
    infolder = config["data"]["infolder"]
    outfolder = config["data"]["outfolder"]
    
    M = if4site ? atype(MPO_2x2(model)) : atype(MPO(model))
    D = size(M,2)
    MM= atype(zeros(ComplexF64, (size(M)...,Ni,Nj)))
    for j in 1:Nj, i in 1:Ni
        MM[:,:,:,:,i,j] = M
    end
    
     infolder = joinpath( infolder, "$model", "groundstate")
    outfolder = joinpath(outfolder, "$model", "groundstate")
    out_chkp_file = joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D)_χ$(targχ).jld2")
    out_log_file = joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D)_χ$(targχ).log")

    AL, C, AR = init_canonical_mps(;infolder = infolder, 
                                    atype = atype,   
                                    Ni = Ni,
                                    Nj = Nj,       
                                    D = D, 
                                    χ = χ,
                                    targχ = targχ)
    err = Inf
    i = 0
    energy = 0
    while err > tol && i < iters
        i += 1
        E, Ǝ = envir_MPO(AL, AR, MM)
        AC = ALCtoAC(AL,C)
        λAC, AC = ACenv(AC, E, MM, Ǝ)
         λC,  C =  Cenv( C, E,     Ǝ)
        energy = sum(λAC - λC)/Nj
        if4site && (energy /= 4)
        AL, AR, errL, errR = ACCtoALAR(AC, C)
        err = errL + errR
        message = "vumps@$i err = $err energy = $energy\n"
        (i % show_every) == 0 && print(message)
        save(out_chkp_file, "ALCAR", map(Array, (AL, C, AR)))
        logfile = open(out_log_file, "a")
        write(logfile, message)
        close(logfile)
    end

    println("vumps done@$i err = $err energy = $energy")
    return energy
end