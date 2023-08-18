using TeneT: FLmap, FRmap, qrpos, lqpos

@with_kw struct IDMRG1 <: Algorithm
    show_every::Int = 1
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

@with_kw struct IDMRG2 <: Algorithm
    show_every::Int = 1
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

function find_groundstate(model::HamiltonianModel, 
                          alg::IDMRG1;
                          Ni::Int = 1, Nj::Int = 1,
                          χ::Int = 16, targχ::Int = χ,
                          atype = Defaults.atype,
                          save_period = 10,
                          infolder::String = Defaults.infolder,
                          outfolder::String = Defaults.outfolder,
                          verbose::Bool = Defaults.verbose,
                          ifADinit = false,
                          if2site = false,
                          if4site = false
                          )

    Nj == 1 || throw(ArgumentError("Nj must be 1 for IDMRG1 for now")) # TODO: implement Nj > 1

    Mo = if4site ? atype(MPO_2x2(model)) : atype(MPO(model))
    D1 = size(Mo,1)
    D2 = size(Mo,2)
    if if2site
        D2 = D2^2
        M = reshape(ein"abcg,cdef->abdegf"(Mo,Mo), (D1, D2, D1, D2, 1, 1))
    else
        M = atype(zeros(ComplexF64, (size(Mo)...,1,Nj)))
        for j in 1:Nj
            M[:,:,:,:,1,j] = Mo
        end
    end

    AL, C, AR = init_canonical_mps(;Ni = Ni, Nj = Nj, D = D2, χ = χ, targχ = targχ,
                                    atype = atype,
                                    infolder = joinpath(infolder, "$model", "groundstate"),
                                    ifADinit = ifADinit,
                                    verbose = verbose
                                   )

    outfolder = joinpath(outfolder, "$model", "groundstate")
    out_chkp_file = joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D2)_χ$(targχ).jld2")
    out_log_file = joinpath(outfolder,"idmrg_mps_$(Ni)x$(Nj)_D$(D2)_χ$(targχ).log")
    
    err = Inf
    i = 0
    energy = 0
    t0 = time()  # total time
    E, Ǝ = envir_MPO(AL, AR, M)
    normalize!(C)
    AC = ALCtoAC(AL, C)
    while err > alg.tol && i < alg.maxiter
        i += 1
        
        λAC, AC = ACenv(AC, E, M, Ǝ)
        AL, _ = qrpos(reshape(AC, targχ*D2, targχ))
        AL = reshape(AL, targχ,D2,targχ,1,1)
        E[:,:,:,1,:] = FLmap(AL[:,:,:,1,:], conj(AL[:,:,:,1,:]), M[:,:,:,:,1,:], E[:,:,:,1,:])

        λAC, AC = ACenv(AC, E, M, Ǝ)
        Cp, AR = lqpos(reshape(AC, targχ,targχ*D2))
        Cp = reshape(Cp,targχ,targχ,1,1)
        AR = reshape(AR, targχ,D2,targχ,1,1)
        Ǝ[:,:,:,1,:] = FRmap(AR[:,:,:,1,:], conj(AR[:,:,:,1,:]), M[:,:,:,:,1,:], Ǝ[:,:,:,1,:])

        err = norm(C-Cp)
        C = Cp

        i % save_period == 0 && save(out_chkp_file, "ALCAR", map(Array, (AL, C, AR)))
        t = round(time() - t0, digits = 2)

        AC = ALCtoAC(AL, C)
        # energy = ein"((adfij,abcij),gdbij),fgcij -> "(E,AC,M[D2,:,:,:,:,:],conj(AC))[]
        # energy = ein"((abcij,cehij),egbij),aghij -> "(AC,Ǝ,M[:,:,1,:,:,:],conj(AC))[]
        e2 = Array(ein"(((adfij,abcij),dgebij),cehij),fghij -> "(E,AC,M,Ǝ,conj(AC)))[]
        e1 = Array(ein"((abij,acdij),bceij),deij->"(C,E,Ǝ,conj(C)))[]
        energy = real(e2 - e1)
        if4site && (energy /= 4)
        if2site && (energy /= 2)
        message = "$t idmrg@$i err = $err energy = $(energy)\n "
        verbose && (i % alg.show_every) == 0 && print(message)
        logfile = open(out_log_file, "a")
        write(logfile, message)
        close(logfile)
    end

    save(out_chkp_file, "ALCAR", map(Array, (AL, C, AR)))
    verbose && println("idmrg done@$i err = $err energy = $(energy)")
    return energy
end

function find_groundstate(model::HamiltonianModel, 
                          alg::IDMRG2;
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
    D1 = size(Mo,1)
    D2 = size(Mo,2)
    M = atype(zeros(ComplexF64, (size(Mo)...,Ni,Nj)))
    for j in 1:Nj, i in 1:Ni
        M[:,:,:,:,i,j] = Mo
    end

    AL, C, AR = init_canonical_mps(;Ni = Ni, Nj = Nj, D = D2, χ = χ, targχ = targχ,
                                    atype = atype,
                                    infolder = joinpath(infolder, "$model", "groundstate"),
                                    ifADinit = ifADinit,
                                    verbose = verbose
                                   )

    @show norm(C)
    outfolder = joinpath(outfolder, "$model", "groundstate")
    out_chkp_file = joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D2)_χ$(targχ).jld2")
    out_log_file = joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D2)_χ$(targχ).log")
    
    err = Inf
    i = 0
    energy = 0
    t0 = time()  # total time
    # E = atype==Array ? rand(ComplexF64, (χ,D1,χ)) : CUDA.rand(ComplexF64, (χ,D1,χ))
    # Ǝ = atype==Array ? rand(ComplexF64, (χ,D1,χ)) : CUDA.rand(ComplexF64, (χ,D1,χ))
    E, Ǝ = envir_MPO(AL, AR, M)
    normalize!(C)
    while err > alg.tol && i < alg.maxiter
        i += 1

        Cinv = inv(C[:,:,1,1])
        AC2 = ein"(((abij,bcdij),de),efgij),ghij->acfhij"(C, AR, Cinv, AL, C)
        λAC2, AC2 = ACenv2(AC2, E, M, Ǝ)
        F = svd(reshape(AC2, χ*D2, χ*D2))
        AL = reshape(F.U[:,1:χ],   χ,D2,χ,1,1)
        AR = reshape(F.Vt[1:χ,:],  χ,D2,χ,1,1)
        Cp = reshape(diagm(F.S[1:χ]), χ,χ,1,1)

        normalize!(Cp)
        err = norm(C-Cp)
        C = complex(Cp)

        # E, Ǝ = envir_MPO(AL, AR, M)
        # E[:,:,:,1,:] = FLmap(AL[:,:,:,1,:], conj(AL[:,:,:,1,:]), M[:,:,:,:,1,:], E[:,:,:,1,:])
        # Ǝ[:,:,:,1,:] = FRmap(AR[:,:,:,1,:], conj(AR[:,:,:,1,:]), M[:,:,:,:,1,:], Ǝ[:,:,:,1,:])
        # if4site && (energy /= 4)
        save(out_chkp_file, "ALCAR", map(Array, (AL, Cp, AR)))
        t = round(time() - t0, digits = 2)

        ELL, ƎRR = envir_MPO(AL, AR, M)
        AC = ALCtoAC(AL, C)
        e2 = ein"(((adfij,abcij),dgebij),cehij),fghij -> "(ELL,AC,M,ƎRR,conj(AC))[]
        e1 = ein"((abij,acdij),bceij),deij->"(C,ELL,ƎRR,conj(C))[]

        message = "$t idmrg@$i err = $err λ = $(λAC2) energy = $(real(e2-e1))\n "
        verbose && (i % alg.show_every) == 0 && print(message)
        # logfile = open(out_log_file, "a")
        # write(logfile, message)
        # close(logfile)
    end

    verbose && println("idmrg done@$i err = $err")
    return energy
end