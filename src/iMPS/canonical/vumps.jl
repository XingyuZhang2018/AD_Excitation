using TeneT: leftorth, rightorth, LRtoC, ALCtoAC, ACCtoALAR

export vumps

function init_canonical_mps(;infolder = "./data/", 
                             atype = Array, 
                             verbose::Bool = true,        
                             D::Int = 2, 
                             χ::Int = 5)

    in_chkp_file = joinpath(infolder,"canonical_mps_D$(D)_χ$(χ).jld2")
    if isfile(in_chkp_file)
        AL, C, AR = map(atype, load(in_chkp_file)["ALCAR"])
        verbose && println("load canonical mps from $in_chkp_file")
    else
        A = atype(rand(ComplexF64, χ,D,χ,1,1))
        AL, L, _ = leftorth(A)
        R, AR, _ = rightorth(AL)
        C = LRtoC(L,R)
        verbose && println("random initial canonical mps $in_chkp_file")
    end
    return AL, C, AR
end

function envir_MPO(AL, AR, M)
    atype = _arraytype(M)
    χ,d,_ = size(AL)
    W     = size(M, 1)

    AL = reshape(AL, χ,d,χ)
    AR = reshape(AR, χ,d,χ)
    M  = reshape(M,  W,d,W,d)

    E = atype == Array ? zeros(ComplexF64, χ,W,χ) : CUDA.zeros(ComplexF64, χ,W,χ)
    Ǝ = atype == Array ? zeros(ComplexF64, χ,W,χ) : CUDA.zeros(ComplexF64, χ,W,χ)
    _, ɔ = norm_R(AL, conj(AL))
    _, c = norm_L(AR, conj(AR))

    Iχ = atype(I(χ))
    E[:,W,:] = Iχ
    for i in W-1:-1:1
        YL = atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in i+1:W
            YL += ein"(abc,db),(ae,edf)->cf"(AL,M[j,:,i,:],E[:,j,:],conj(AL))
        end
        if M[i,:,i,:] == I(d)
            bL = YL - Array(ein"ab,ab->"(YL,ɔ))[] * Iχ
            E[:,i,:], infoE = linsolve(E->E - ein"abc,(ad,dbe)->ce"(AL,E,conj(AL)) + ein"ab,ab->"(E, ɔ)[] * Iχ, bL)
            @assert infoE.converged == 1
        else
            E[:,i,:] = YL
        end
    end

    Ǝ[:,1,:] = Iχ
    for i in 2:W
        YR = atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in 1:i-1
            YR += ein"((abc,db),cf),edf->ae"(AR,M[i,:,j,:],Ǝ[:,j,:],conj(AR))
        end
        if M[i,:,i,:] == I(d)
            bR = YR - Array(ein"ab,ab->"(c,YR))[] * Iχ
            Ǝ[:,i,:], infoƎ = linsolve(Ǝ->Ǝ - ein"(abc,ce),dbe->ad"(AR,Ǝ,conj(AR)) + ein"ab,ab->"(c, Ǝ)[] * Iχ, bR)
            @assert infoƎ.converged == 1
        else
            Ǝ[:,i,:] = YR
        end
    end

    E = reshape(E, χ,W,χ,1,1)
    Ǝ = reshape(Ǝ, χ,W,χ,1,1)
    return E, Ǝ
end

"""

"""
function vumps(model;
               χ::Int = 10, 
               iters::Int = 100,
               tol::Float64 = 1e-10,
               infolder = "./data/", outfolder = "./data/",
               show_every = Inf,
               atype = Array)
               
    M = atype(MPO(model))
    D = size(M,2)
    W = size(M,1)
    M  = reshape(M,  W,D,W,D,1,1)
    
     infolder = joinpath( infolder, "$model")
    outfolder = joinpath(outfolder, "$model")
    out_chkp_file = joinpath(outfolder,"canonical_mps_D$(D)_χ$(χ).jld2")


    AL, C, AR = init_canonical_mps(;infolder = infolder, 
                                    atype = atype,        
                                    D = D, 
                                    χ = χ)
    err = Inf
    i = 0
    energy = 0
    while err > tol && i < iters
        i += 1
        E, Ǝ = envir_MPO(AL, AR, M)
        AC = ALCtoAC(AL,C)
        λAC, AC = ACenv(AC, E, M, Ǝ)
         λC,  C =  Cenv( C, E,    Ǝ)
        energy = λAC - λC
        AL, AR, errL, errR = ACCtoALAR(AC, C)
        err = errL + errR
        (i % show_every) == 0 && println("vumps@$i err = $err energy = $energy")
        save(out_chkp_file, "ALCAR", map(Array, (AL, C, AR)))
    end

    println("vumps done@$i err = $err energy = $energy")
    return energy
end