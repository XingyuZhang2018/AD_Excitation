using CUDA
CUDA.allowscalar(false)
using Random
   
using TeneT: leftorth, rightorth, LRtoC, ALCtoAC, ACCtoALAR

@with_kw struct VUMPS <: Algorithm
    show_every::Int = 100
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

function init_canonical_mps(;Ni = 1, Nj = 1, D, χ, targχ = χ,
                             verbose = Defaults.verbose, 
                             atype = Defaults.atype, 
                             infolder = Defaults.infolder, 
                             ifADinit = false
                            )

    in_chkp_file = joinpath(infolder, "groundstate", "canonical_mps_$(Ni)x$(Nj)_D$(D)_χ$(χ).jld2")
    if isfile(in_chkp_file) && !ifADinit
        AL = atype(rand(ComplexF64, targχ,D,targχ,Ni,Nj)) * 1e-6
        AR = atype(rand(ComplexF64, targχ,D,targχ,Ni,Nj)) * 1e-6
            C = atype(rand(ComplexF64, targχ,  targχ,Ni,Nj)) * 1e-6
        AL[1:χ,:,1:χ,:,:], C[1:χ,1:χ,:,:], AR[1:χ,:,1:χ,:,:] = map(atype, load(in_chkp_file)["ALCAR"])
        verbose && println("load canonical mps from $in_chkp_file")
        verbose && targχ > χ && println("and increase χ from $(χ) to $(targχ)")
    else
        A = atype(rand(ComplexF64, χ,D,χ,Ni,Nj))
        if ifADinit
            in_chkp_file = joinpath(infolder, "groundstate", "uniform_mps_D$(D)_χ$(χ).jld2")
            A = reshape(atype(load(in_chkp_file)["A"]), χ, D, χ, Ni, Nj)
            verbose && println("load mps from $in_chkp_file")
        end
        AL, L, _ = leftorth(A)
        R, AR, _ = rightorth(AL)
        C = LRtoC(L,R)
        verbose && !ifADinit &&println("random initial canonical mps $in_chkp_file")
    end
    return AL, C, AR
end

function find_groundstate(model::HamiltonianModel, 
                          alg::VUMPS;
                          Ni::Int = 1, Nj::Int = 1,
                          χ::Int = 16, targχ::Int = χ,
                          atype = Defaults.atype,
                          infolder::String = Defaults.infolder,
                          outfolder::String = Defaults.outfolder,
                          verbose::Bool = Defaults.verbose,
                          ifADinit = false,
                          if4site = false
                          )

    Mo = if4site ? atype(MPO_2x2(model)) : atype(MPO(model))
    D = size(Mo,2)
    M = atype(zeros(ComplexF64, (size(Mo)...,Ni,Nj)))
    for j in 1:Nj, i in 1:Ni
        M[:,:,:,:,i,j] = Mo
    end

    AL, C, AR = init_canonical_mps(;Ni = Ni, Nj = Nj, D = D, χ = χ, targχ = targχ,
                                    atype = atype,
                                    infolder = joinpath(infolder, "$model"),
                                    ifADinit = ifADinit,
                                    verbose = verbose
                                   )

    
    outfolder = joinpath(outfolder, "$model", "groundstate")
    out_chkp_file = joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D)_χ$(targχ).jld2")
    out_log_file = joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D)_χ$(targχ).log")
    
    err = Inf
    i = 0
    energy = 0
    while err > alg.tol && i < alg.maxiter
        i += 1
        E, Ǝ = envir_MPO(AL, AR, M)
        AC = ALCtoAC(AL,C)
        λAC, AC = ACenv(AC, E, M, Ǝ)
         λC,  C =  Cenv( C, E,    Ǝ)
        energy = sum(λAC - λC)/Nj
        if4site && (energy /= 4)
        AL, AR, errL, errR = ACCtoALAR(AC, C)
        err = errL + errR
        message = "vumps@$i err = $err energy = $energy\n"
        verbose && (i % alg.show_every) == 0 && print(message)
        save(out_chkp_file, "ALCAR", map(Array, (AL, C, AR)))
        logfile = open(out_log_file, "a")
        write(logfile, message)
        close(logfile)
    end

    verbose && println("vumps done@$i err = $err energy = $energy")
    return energy
end