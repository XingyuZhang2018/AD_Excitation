function C工linear(T, C, Ɔ, Cb)
    x, info = linsolve(x->x - ein"abc,(ad,dbe)->ce"(T,x,conj(T)) + ein"(ab,ab),cd->cd"(x,Ɔ,C), Cb)
    @assert info.converged == 1
    return x
end

function 工Ɔlinear(T, C, Ɔ, Ɔb)
    x, info = linsolve(x->x - ein"(abc,ce),dbe->ad"(T,x,conj(T)) + ein"(ab,ab),cd->cd"(C,x,Ɔ), Ɔb)
    @assert info.converged == 1
    return x
end

function envir_MPO(A, M, c, ɔ)
    χ = size(A, 1)
    W = size(M, 1)
    atype = _arraytype(A)
    E = Zygote.Buffer(A, χ,W,χ)
    Ǝ = Zygote.Buffer(A, χ,W,χ)
    # c,ɔ = env_norm(A)

    E[:,W,:] = c
    for i in W-1:-1:1
        YL = Zygote.@ignore atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in i+1:W
            YL += ein"(abc,db),(ae,edf)->cf"(A,M[j,:,i,:],E[:,j,:],conj(A))
        end
        if i == 1 #if M[i,:,i,:] == I(d)
            bL = YL
            E[:,i,:] = C工linear(A, c, ɔ, bL)
        else
            E[:,i,:] = YL
        end
    end

    Ǝ[:,1,:] = ɔ
    for i in 2:W
        YR = Zygote.@ignore atype == Array ? zeros(ComplexF64, χ,χ) : CUDA.zeros(ComplexF64, χ,χ)
        for j in 1:i-1
            YR += ein"((abc,db),cf),edf->ae"(A,M[i,:,j,:],Ǝ[:,j,:],conj(A))
        end
        if i == W # if M[i,:,i,:] == I(d)
            bR = YR
            Ǝ[:,i,:] = 工Ɔlinear(A, c, ɔ, bR)
        else
            Ǝ[:,i,:] = YR
        end
    end

    # @show ein"ab,ab->"(c,YR)[] ein"ab,ab->"(YL,ɔ)[] ein"ab,ab->"(c,Ǝ[:,3,:])[] ein"ab,ab->"(E[:,1,:],ɔ)[] 
    # @show ein"(abc,ce),(ad,dbe)->"(A,Ǝ[:,3,:],c,conj(A))[]
    return copy(E), copy(Ǝ)
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