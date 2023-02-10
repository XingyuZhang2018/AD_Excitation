using TeneT: leftorth, rightorth, LRtoC, ALCtoAC, ACCtoALAR
using LinearAlgebra: norm

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
        AL = atype(zeros(ComplexF64, targχ,D,targχ,Ni,Nj))
        AR = atype(zeros(ComplexF64, targχ,D,targχ,Ni,Nj))
            C = atype(zeros(ComplexF64, targχ,  targχ,Ni,Nj))
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

function c工map(c, Au, Ad, H)
    ein"((ad,abc),eb),def->cf"(c,Au,H,conj(Ad))
end

function 工ɔmap(ɔ, Au, Ad, H)
    ein"((abc,cf),db),edf->ae"(Au,ɔ,H,conj(Ad))
end

function left_cyclethrough!(col::Int, E, M, AL)
    len = size(M,6)
    for i = 1:len
        ir = mod1(i+1, len)
        E[:,col,:,1,ir] .= 0
        for j = col:-1:1
            if reduce(|, M[j,:,col,:,1,i] .!= 0)
                E[:,col,:,1,ir] .+= c工map(E[:,j,:,1,i], AL[:,:,:,1,i], AL[:,:,:,1,i], M[j,:,col,:,1,i])
            end
        end
    end
end

function right_cyclethrough!(col::Int, Ǝ, M, AR)
    len = size(M,6)
    for i = len:-1:1
        ir = mod1(i-1, len)
        Ǝ[:,col,:,1,ir] .= 0 
        for j = col:size(M,1)
            if reduce(|, M[col,:,j,:,1,i] .!= 0)
                Ǝ[:,col,:,1,ir] .+= 工ɔmap(Ǝ[:,j,:,1,i], AR[:,:,:,1,i], AR[:,:,:,1,i], M[col,:,j,:,1,i])
            end
        end
    end
end

function cmap_through(c, Au, Ad)
    cm = copy(c)
    for i in 1:size(Au,5)
        cm .= cmap(cm, Au[:,:,:,1,i], Ad[:,:,:,1,i])
    end
    return cm
end

function ɔmap_through(ɔ, Au, Ad)
    ɔm = copy(ɔ)
    for i in size(Au,5):-1:1
        ɔm .= ɔmap(ɔm, Au[:,:,:,1,i], Ad[:,:,:,1,i])
    end
    return ɔm
end

function canonical_envir_MPO!(E, Ǝ, AL, AR, C, M)
    atype = _arraytype(M)
    χ,Ni,Nj = size(AL)[[1,4,5]]
    W       = size(M, 1)

    _, c = env_c(AR, conj(AR))
    _, ɔ = env_ɔ(AL, conj(AL))
    for y in 1:Nj, x in 1:Ni
        ɔ[:,:,x,y] ./= tr(ɔ[:,:,x,y])
        c[:,:,x,y] ./= tr(c[:,:,x,y])
    end

    # c = circshift(ein"abij,acij->bcij"(conj(C),C), (0,0,0,1)) 
    # ɔ = ein"abij,cbij->acij"(C,conj(C))

    Iχ = atype(I(χ))
    E[:,1,:,1,1] .= Iχ
    Nj>1 && left_cyclethrough!(1, E, M, AL)
    for i in 2:W
        prev = copy(E[:,i,:,1,1]);
        E[:,i,:,1,1] .= 0
        left_cyclethrough!(i, E, M, AL)        
        if i == W
            E[:,i,:,1,1], infoE = linsolve(X->X - cmap_through(X, AL, AL) + ein"(ab,ab),cd->cd"(X, ɔ[:,:,1,end], Iχ), E[:,i,:,1,1], prev)
            @assert infoE.converged == 1
            Nj>1 && left_cyclethrough!(i, E, M, AL)

            for j in 1:Nj
                jr = mod1(j-1, Nj)
                E[:,i,:,1,j] .-= ein"(ab,ab),cd->cd"(E[:,i,:,1,j], ɔ[:,:,1,jr], Iχ)
            end
        else
            ## To do: M contain I 
            Nj>1 && left_cyclethrough!(i, E, M, AL)
        end
    end

    Ǝ[:,end,:,1,end] .= Iχ
    Nj>1 && right_cyclethrough!(W, Ǝ, M, AR)
    for i in W-1:-1:1
        prev = copy(Ǝ[:,i,:,1,end])
        Ǝ[:,i,:,1,end] .= 0
        right_cyclethrough!(i, Ǝ, M, AR)    
        if i == 1 # if M[i,:,i,:] == I(d)
            Ǝ[:,i,:,1,end], infoƎ = linsolve(X->X - ɔmap_through(X, AR, AR) + ein"(ab,ab),cd->cd"(c[:,:,1,1], X, Iχ), Ǝ[:,i,:,1,end], prev)
            @assert infoƎ.converged == 1
            Nj>1 && right_cyclethrough!(i, Ǝ, M, AR)

            for j in 1:Nj
                jr = mod1(j+1, Nj)
                Ǝ[:,i,:,1,j] .-= ein"(ab,ab),cd->cd"(c[:,:,1,jr], Ǝ[:,i,:,1,j], Iχ)
            end
        else
            ## To do: M contain I 
            Nj>1 && right_cyclethrough!(i, Ǝ, M, AR)
        end
    end
end

"""

"""
function vumps(model;
               Ni::Int = 1,
               Nj::Int = 1,
               χ::Int = 10, 
               targχ = χ,
               iters::Int = 100,
               tol::Float64 = 1e-8,
               infolder = "../data/", outfolder = infolder,
               show_every = Inf,
               atype = Array)
               
    Mo = atype(MPO(model))
    D = size(Mo,2)
    W = size(Mo,1)
    M = zeros(ComplexF64, (size(Mo)...,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        M[:,:,:,:,i,j] = Mo
    end
    
     infolder = joinpath( infolder, "$model")
    outfolder = joinpath(outfolder, "$model")
    out_chkp_file = joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D)_χ$(targχ).jld2")

    AL, C, AR = init_canonical_mps(;infolder = infolder, 
                                    atype = atype,   
                                    Ni = Ni,
                                    Nj = Nj,       
                                    D = D, 
                                    χ = χ,
                                    targχ = targχ)

    E = atype == Array ? zeros(ComplexF64, χ,W,χ,Ni,Nj) : CUDA.zeros(ComplexF64, χ,W,χ,Ni,Nj)
    Ǝ = atype == Array ? zeros(ComplexF64, χ,W,χ,Ni,Nj) : CUDA.zeros(ComplexF64, χ,W,χ,Ni,Nj)
    err = Inf
    i = 0
    energy = 0
    while err > tol && i < iters
        i += 1
        canonical_envir_MPO!(E, Ǝ, AL, AR, C, M)
        AC = ALCtoAC(AL,C)
        λAC, AC = ACenv(AC, E, M, Ǝ)
         λC,  C =  Cenv( C, E,    Ǝ)
         @show λAC λC
        # for y in 1:Nj, x in 1:Ni
        #     C[:,:,x,y] ./= sqrt(ein"ab,ab->"(C[:,:,x,y], conj(C[:,:,x,y]))[])
        # end
        energy = sum(λAC - λC)/Nj
        AL, AR, errL, errR = ACCtoALAR(AC, C)
        err = errL + errR
        (i % show_every) == 0 && println("vumps@$i err = $err energy = $energy")
        save(out_chkp_file, "ALCAR", map(Array, (AL, C, AR)))
    end

    println("vumps done@$i err = $err energy = $energy")
    return energy
end