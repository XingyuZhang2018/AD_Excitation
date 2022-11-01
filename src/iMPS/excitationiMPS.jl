export excitation_spectrum

"""
    L_n, R_n = env_norm(A)

    get normalized environment of A

    ```  
                            a──────┬──────b
    ┌───┐                   │      │      │
    L   R  = 1              │      c      │
    └───┘                   │      │      │
                            d──────┴──────e      
                                                        
    ┌─ A ─      ┌─            ─ A──┐       ─┐
    L  │   =    L               │  R  =     R  
    └─ A*─      └─            ─ A*─┘       ─┘
    ```  
"""
function env_norm(A)
    _, L_n = norm_L(A, conj(A))
    _, R_n = norm_R(A, conj(A))
    n = ein"((ad,acb),dce),be->"(L_n,A,conj(A),R_n)[]
    L_n /= n
    return L_n, R_n
end

"""
    B = initial_excitation(A)

    Construct a `B` tensor that is orthogonal to A:    
    ```
                                                        a──────┬──────b
    ┌──B──┐                                             │      │      │
    L  │  R  = 0                                        │      c      │
    └──A*─┘                                             │      │      │
                                                        d──────┴──────e                                      
    ```
    Fix gauge by
    ```
    ┌──B─      
    L  │   = 0 
    └──A*      
    ```

    B is get from https://arxiv.org/abs/1810.07006 Eq.(73)
"""
function initial_excitation(A, L_n, R_n)
    χ,D,_ = size(A)
    Bs = []

    VL = randn(χ, D, χ*(D-1))
    sq_L_n = sqrt(L_n)
    sq_R_n = sqrt(R_n)
    inv_sq_L_n = sq_L_n^-1
    inv_sq_R_n = sq_R_n^-1
    λL = ein"(ad,acb),dce -> eb"(sq_L_n,VL,conj(A))
    VL -= ein"(ba,bcd),ed,ef ->acf"(sq_L_n,A,L_n^-1,λL)
    Q, _ = qrpos(reshape(VL, χ*D, χ*(D-1)))
    VL = reshape(Q, χ, D, χ*(D-1))
    for i in 1:(D-1)*χ^2
        X = zeros(ComplexF64, χ*(D-1), χ)
        X[i] = 1.0
        X /= sqrt(ein"ab,ab->"(X,conj(X))[])
        B = ein"((ba,bcd),de),ef->acf"(inv_sq_L_n,VL,X,inv_sq_R_n)
        push!(Bs, B)
    end
    return Bs
end

function initial_VL(A, L_n)
    χ,D,_ = size(A)
    VL = randn(χ, D, χ*(D-1))
    sq_L_n = sqrt(L_n)
    λL = ein"(ad,acb),dce -> eb"(sq_L_n,VL,conj(A))
    VL -= ein"(ba,bcd),ed,ef ->acf"(sq_L_n,A,L_n^-1,λL)
    Q, _ = qrpos(reshape(VL, χ*D, χ*(D-1)))
    VL = reshape(Q, χ, D, χ*(D-1))
    return VL
end

function initial_excitation_U(A)
    _, L_n = norm_L(A, conj(A))
    _, R_n = norm_R(A, conj(A))
    x = 1.5707963267948966  + 0.5351372028447591im # χ=2
    U = [cos(x) -sin(x); sin(x) cos(x)]
    B = ein"abc,bd->adc"(A, U)
    # env = ein"((ad,acb),dfe),be->cf"(L_n,A,conj(A),R_n)
    @assert norm(ein"((ad,acb),dce),be->"(L_n,B,conj(A),R_n)[]) < 1e-8
    return B
end

"""
    s1 = sum_series(A, L_n, R_n)

    s1 = sum of `    工` series:
    ```
        ─          ┬           ┬┬          ┬┬┬               ┬┬┬...┬
           +       │   +       ││  +       │││  + ... +      │││...│ + ...
        ─          ┴           ┴┴          ┴┴┴               ┴┴┴...┴
    ```
"""
function sum_series(A, L_n, R_n)
    χ = size(A, 1)
    工_I = ein"ab,cd->acbd"(I(χ), I(χ))
    工 = ein"acb,dce->adbe"(A, conj(A))
    rl = ein"ab,cd->abcd"(R_n, L_n)
    a1 = 工_I - (工 - rl)
    b1 = 工_I
    s1, info1 = linsolve(x->ein"adbe,becf->adcf"(a1,x), b1)
    @assert info1.converged == 1
    return s1
end

"""
    s2, s3 = sum_series_k(k, A, L_n, R_n)

    s2 = sum of `eⁱᵏ 工` series: 
    ```
         ─       ┬           ┬┬          ┬┬┬               ┬┬┬...┬
    eⁱ⁰ᵏ   + eⁱ¹ᵏ│   +  eⁱ²ᵏ ││  +  eⁱ³ᵏ │││  + ... + eⁱⁿᵏ │││...│ + ...
         ─       ┴           ┴┴          ┴┴┴               ┴┴┴...┴
    ```

    s3 = sum of `eⁱ⁻ᵏ 工` series:
"""
function sum_series_k(k, A, L_n, R_n)
    χ = size(A, 1)
    工_I = ein"ab,cd->acbd"(I(χ), I(χ))
    工 = ein"acb,dce->adbe"(A, conj(A))
    rl = ein"ab,cd->abcd"(R_n, L_n)

    a2 = 工_I - exp(1.0im * k) * (工 - rl)
    b2 = 工_I
    s2, info2 = linsolve(x->ein"adbe,becf->adcf"(a2,x), b2)
    @assert info2.converged == 1

    a3 = 工_I - exp(1.0im *-k) * (工 - rl)
    b3 = 工_I
    s3, info3 = linsolve(x->ein"adbe,becf->adcf"(a3,x), b3)
    @assert info3.converged == 1
    return s2, s3
end

"""
    ```
     ┌───A───┐             a──┬──b
     │   │   │             │  │  │ 
     L   │   R             │  c  │ 
     │   │   │             │  │  │ 
     └──   ──┘             d─   ─e
    ```
"""
einLAR(L,A,R) = ein"(ad,acb),be->dce"(L, A, R)

"""
    ```
     ┌───B────┬─            a──┬──b──┬──c
     │   │    │             │  │     │  
    L_n  │    s             │  d     │  
     │   │    │             │  │     │  
     └───A*───┴─            e──┴──f──┴──g 
    ```
"""
# einLB(L_n, B, A, s2) = ein"((ae,adb),edf),bfcg->cg"(L_n, B, conj(A), s2)
function einLB(k, L, B, A, Ln, Rn)
    LB, info = linsolve(LB->LB - exp(1.0im * k) * ein"(ad,abc),dbe->ce"(LB,A,conj(A)) + exp(1.0im * k) * ein"ab,ab->"(LB,Rn)[]*Ln, ein"(ad,abc),dbe->ce"(L,B,conj(A)))
    @assert info.converged == 1
    return LB
end


"""
    ```
    ─┬───B───┐              a──┬──b──┬──c
     │   │   │                 │     │  │
     s3  │  R_n                │     d  │
     │   │   │                 │     │  │
    ─┴───A*──┘              e──┴──f──┴──g 
    ```
"""
# einRB(s3, B, A, R_n) = ein"aebf,(bdc,(fdg,cg))->ae"(s3, B, conj(A), R_n)
function einRB(k, R, B, A, Ln, Rn)
    RB, info = linsolve(RB->RB - exp(1.0im *-k) * ein"(abc,ce),dbe->ad"(A,RB,conj(A)) + exp(1.0im *-k) * ein"ab,ab->"(Ln,RB)[]*Rn, ein"(abc,ce),dbe->ad"(B,R,conj(A)))
    @assert info.converged == 1
    return RB
end

"""
    ```
     ┌───A1────A2───┬──           a──┬──c──┬──e──┬──f
     │   │     │    │             │  b     d     │   
    L_n  ├─ H ─┤    s1            │  ├─────┤     │   
     │   │     │    │             │  g     h     │   
     └───A3*───A4*──┴──           i──┴──j──┴──k──┴──l 
    ```
"""
# einLH(A1, A2, A3, A4, s1, L_n, H) = ein"(((((ai,abc),cde),bgdh),igj),jhk),ekfl->fl"(L_n,A1,A2,H,conj(A3),conj(A4),s1)
function einLH(k, L, A1, A2, A3, A4, A, H, Ln, Rn)
    LH, info = linsolve(LH->LH - exp(1.0im * k) * ein"(ad,abc),dbe->ce"(LH,A,conj(A)) + exp(1.0im * k) * ein"ab,ab->"(LH,Rn)[]*Ln, ein"(((ah,abc),cde),(hfi,igj)),bfdg->ej"(L,A1,A2,conj(A3),conj(A4),H))
    @assert info.converged == 1
    return LH
end

"""
    ```
     ──┬────A1────A2──┐          a──┬──b──┬──d──┬──f
       │    │     │   │             │     c     e  │ 
       s1   ├─ H ─┤  R_n            │     ├─────┤  │ 
       │    │     │   │             │     g     h  │ 
     ──┴────A3*───A4*─┘          i──┴──j──┴──k──┴──l 
    ```
"""
# einRH(A1, A2, A3, A4, s1, R_n, H) = ein"(((bcd,(def,fl),cgeh),khl),jgk),aibj->ai"(A1,A2,R_n,H,conj(A4),conj(A3),s1)
function einRH(k, R, A1, A2, A3, A4, A, H, Ln, Rn)
    RB, info = linsolve(RB->RB - exp(1.0im *-k) * ein"(abc,ce),dbe->ad"(A,RB,conj(A)) + exp(1.0im *-k) * ein"ab,ab->"(Ln,RB)[]*Rn, ein"((abc,cde),((hfi,igj),ej)),bfdg->ah"(A1,A2,conj(A3),conj(A4),R,H))
    @assert info.converged == 1
    return RB
end

"""
    ```
     ┌───A1────A2──┐            a──┬──c──┬──e        
     │   │     │   │            │  b     d  │        
     L   ├─ H ─┤   R            │  ├─────┤  │        
     │   │     │   │            │  f     g  │        
     └──   ────A4*─┘            h─   ─i──┴──j        
    ```
"""
einH_dL(A1,A2,A4,L,H,R) = ein"((((ah,abc),cde),bfdg),ej),igj->hfi"(L,A1,A2,H,R,conj(A4))

"""
    ```
     ┌───A1────A2──┐            a──┬──c──┬──e        
     │   │     │   │            │  b     d  │        
     L   ├─ H ─┤   R            │  ├─────┤  │        
     │   │     │   │            │  f     g  │        
     └───A3*──   ──┘            h──┴──i─   ─j        
    ```
"""
einH_dR(A1,A2,A3,L,H,R) = ein"((((ah,abc),cde),bfdg),ej),hfi->igj"(L,A1,A2,H,R,conj(A3))  

"""
    N_mn = N_eff(k, A, Bu, Bd, L_n, R_n, s2, s3)
    get `<Ψₖ(B)|Ψₖ(B)>`, including sum graphs: 
    ```
    1. Bu, Bd* on the same site
     ┌───Bu──┐             a─────┬─────b  
     │   │   │             │     │     │ 
    L_n  │  R_n            │     c     │ 
     │   │   │             │     │     │ 
     └───Bd*─┘             d─────┴─────e 
    2. Bu, Bd* on the different sites
     ┌───Bu──┬───A───┐ 
     │   │   │   │   │ 
    L_n  │  s2   │  R_n  (removed cased by gauge)
     │   │   │   │   │ 
     └───A*──┴───Bd*─┘ 
    ```
"""
function N_eff(k, A, Bu, Ln, Rn, s2, s3)
    # 1. B, dB on the same site
    NB = einLAR(Ln, Bu, Rn)

    # 2. B, dB on the different sites 
    # LB = einLB(Ln, Bu, A, s2)
    # RB = einRB(s3, Bu, A, Rn)
    # NB += einLAR(LB, A, Rn) * exp(1.0im * k)
    # NB += einLAR(Ln, A, RB) * exp(1.0im *-k)
    return NB
end

"""
    H_mn = H_eff(k, A, Bu, Bd, H, L_n, R_n, s1, s2, s3)

    get `<Ψₖ(B)|H|Ψₖ(B)>`, including sum graphs form https://arxiv.org/abs/1810.07006 Eq.(193)
"""
function H_eff(k, A, Bu, H, Ln, Rn, LH, RH)
    # LB = einLB(Ln, Bu, A, s2)
    RB = einRB(k, Rn, Bu, A, Ln, Rn)

    # 1. B and dB on the same site but are away from H
    HB  = einLAR(LH, Bu, Rn) +
          einLAR(Ln, Bu, RH)

    # 2. dB on the same site of H
    HB += einH_dL(Bu,A ,A,Ln,H,Rn)                  + 
          einH_dL(A ,Bu,A,Ln,H,Rn) * exp(1.0im *-k) + 
        #   einH_dL(A ,A ,A,LB,H,Rn) * exp(1.0im * k) +
          einH_dL(A ,A ,A,Ln,H,RB) * exp(2.0im *-k) +
          einH_dR(Bu,A ,A,Ln,H,Rn) * exp(1.0im * k) +
          einH_dR(A ,Bu,A,Ln,H,Rn)                  +
        #   einH_dR(A, A ,A,LB,H,Rn) * exp(2.0im * k) +
          einH_dR(A, A ,A,Ln,H,RB) * exp(1.0im *-k)

    # 3. LB and RH on the different sites of A
    HB += einLAR(LH, A, RB) * exp(1.0im *-k) 
        #   einLAR(LB, A, RH) * exp(1.0im * k)
          
    
    # 4. LB and RH on the same site of A
    L1 = einLB(k, LH, Bu,A, Ln, Rn)        * exp(1.0im * k) +
         einLH(k, Ln, Bu, A, A, A, A, H, Ln, Rn) * exp(2.0im * k) +
         einLH(k, Ln, A, Bu, A, A, A, H, Ln, Rn) * exp(1.0im * k)  
        #  einLH(A,  A, A, A, s2, LB, H) * exp(3.0im * k)
    # R1 = einRB(s3, Bu,A, RH          ) * exp(1.0im *-k) +
    #      einRH(Bu, A, A, A, s3, Rn, H) * exp(1.0im *-k) +
    #      einRH(A,  Bu,A, A, s3, Rn, H) * exp(2.0im *-k) + 
    #      einRH(A,  A, A, A, s3, RB, H) * exp(3.0im *-k) 
    HB += einLAR(L1, A, Rn) 
        #   einLAR(Ln, A, R1) 
    # for generalized eigsolve to ordinary eigsolve
    # HB = ein"(ad,acb),be->dce"(L_n^-1, HB, R_n^-1)      
    # Y = ein"((ba,bcd),acf),de->fe"(sq_L_n,HB,conj(VL),sq_R_n)

    # Y = ein"((ba,bcd),acf),de->fe"(inv_sq_L_n,HB,conj(VL),inv_sq_R_n)
    # @show norm(HB-ein"((ba,bcd),de),ef->acf"(inv_sq_L_n,VL,Y,inv_sq_R_n))
    return HB
end

"""
    excitation_spectrum(k, A, H, n)

find at least `n` smallest excitation gaps 
"""
function excitation_spectrum(k, A, model, n::Int = 1;
                             infolder = "./data/", outfolder = "./data/")
     infolder = joinpath( infolder, "$model")
    outfolder = joinpath(outfolder, "$model")

    χ, D, _ = size(A)
    key = D, χ, infolder, outfolder
    H = _arraytype(A)(hamiltonian(model))
    H -= energy_gs(A, H, key) * ein"ab,cd->abcd"(I(D),I(D)) # critical subtraction for the energy gap

    Ln, Rn    = env_norm(A)
    sq_Ln     = sqrt(Ln)
    sq_Rn     = sqrt(Rn)
    inv_sq_Ln = sq_Ln^-1
    inv_sq_Rn = sq_Rn^-1
    # s1        = sum_series(     A, Ln, Rn)
    # s2, s3    = sum_series_k(k, A, Ln, Rn)
    VL        = initial_VL(A, Ln)
    LH        = einLH(0, Ln, A, A, A, A, A, H, Ln, Rn) 
    RH        = einRH(0, Rn, A, A, A, A, A, H, Ln, Rn)

    X = zeros(ComplexF64, χ*(D-1), χ)
    # X = rand(ComplexF64, χ, D, χ)
    X[1] = 1.0
    # X /= sqrt(ein"ab,ab->"(X,conj(X))[])
    
    function f(X)
        Bu = ein"((ba,bcd),de),ef->acf"(inv_sq_Ln, VL, X, inv_sq_Rn)
        HB = H_eff(k, A, Bu, H, Ln, Rn, LH, RH)
        # NB = N_eff(k, A, Bu, Ln, Rn, s2, s3)
        HB = ein"((ba,bcd),acf),de->fe"(inv_sq_Ln,HB,conj(VL),inv_sq_Rn)
        # NB = ein"((ba,bcd),acf),de->fe"(inv_sq_Ln,NB,conj(VL),inv_sq_Rn)
        return HB
    end
    # Δ, Y, info = geneigsolve(x -> f(x), X, n, :SR, ishermitian = true, isposdef = true)
    Δ, Y, info = eigsolve(x -> f(x), X, n, :SR; ishermitian = true, maxiter = 100)
    # @assert info.converged == 1
    return Δ, Y, info
end