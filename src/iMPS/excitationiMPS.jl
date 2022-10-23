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
    L_A_s = einL_A_s(L_n, A1, A2, s)

    ```
     ┌───A1───┬─            a──┬──b──┬──c
     │   │    │             │  │     │  
    L_n  │    s             │  d     │  
     │   │    │             │  │     │  
     └───A2*──┴─            e──┴──f──┴──g 
    ```
"""
einL_A_s(L_n, A1, A2, s) = ein"((ae,adb),edf),bfcg->cg"(L_n, A1, conj(A2), s)

"""
    L_A_s = einL_A_s(L_n, A, A2, s)

    ```
     ┌───A────┬───┐             a──┬──b──┬──c
     │   │    │   │             │  │     │  │
    L_n  │    s  R_n            │  d     │  │
     │   │    │   │             │  │     │  │
     └──   ───┴───┘             e─   ─f──┴──g 
    ```
"""
einL_A_s_R(L_n, A, s, R_n) = ein"(ae,adb),(bfcg,cg)->edf"(L_n, A, s, R_n)


"""
    s_A_R = eins_A_R(s, A1, A2, R_n)

    ```
    ─┬───A1──┐           a──┬──b──┬──c
     │   │   │              │     │  │
     s   │  R_n             │     d  │
     │   │   │              │     │  │
    ─┴───A2*─┘           e──┴──f──┴──g 
    ```
"""
eins_A_R(s, A1, A2, R_n) = ein"aebf,(bdc,(fdg,cg))->ae"(s, A1, conj(A2), R_n)

"""
    s_A_R = eins_A_R(s, A1, A2, R_n)

    ```
     ┌───┬───A───┐              a──┬──b──┬──c
     │   │   │   │              │  │     │  │
    L_n  s   │  R_n             │  │     d  │
     │   │   │   │              │  │     │  │
     └───┴──   ──┘              e──┴──f─   ─g 
    ```
"""
einL_s_A_R(L_n, s, A, R_n) = ein"((ae,aebf),bdc),cg->fdg"(L_n, s, A, R_n)

"""
    H_L = einH_L(A1, A2, A3, A4, L_n, H)

    ```
     ┌───A1────A2──             a──┬──c──┬──e
     │   │     │                │  b     d  │ 
    L_n  ├─ H ─┤                │  ├─────┤  │ 
     │   │     │                │  f     g  │ 
     └───A3*───A4*─             h──┴──i──┴──j 
    ```
"""
einH_L(A1, A2, A3, A4, L_n, H) = ein"(((ah,abc),cde),(hfi,igj)),bfdg->ej"(L_n,A1,A2,conj(A3),conj(A4),H)

"""
    H_R = einH_R(A1, A2, A3, A4, R_n, H)

    ```
    ───A1────A2──┐              a──┬──c──┬──e
       │     │   │              │  b     d  │ 
       ├─ H ─┤  R_n             │  ├─────┤  │ 
       │     │   │              │  f     g  │ 
    ───A3*───A4*─┘              h──┴──i──┴──j 
    ```
"""
einH_R(A1, A2, A3, A4, R_n, H) = ein"((abc,cde),((hfi,igj),ej)),bfdg->ah"(A1,A2,conj(A3),conj(A4),R_n,H)

"""
    H_mn = H_eff(k, A, Bu, Bd, H, L_n, R_n, s1, s2, s3)

    get `<Ψₖ(B)|H|Ψₖ(B)>`, including sum graphs: 
    ```
    1. Bu, Bd* and H on the same site:
     ┌───Bu────A───┐ 
     │   │     │   │ 
    L_n  ├─ H ─┤  R_n 
     │   │     │   │ 
     └───A*────Bd*─┘ 

    2. Bu and Bd* are on the same site but away from the site of H
     ┌───Bu──┬───A─────A───┐ 
     │   │   │   │     │   │ 
    L_n  │  s1   ├─ H ─┤  R_n 
     │   │   │   │     │   │ 
     └───Bd*─┴───A*────A*──┘ 

    3. one of Bu and Bd* on the same site of H
     ┌───Bu──┬───A─────A───┐ 
     │   │   │   │     │   │ 
    L_n  │  s2   ├─ H ─┤  R_n (removed cased by gauge)
     │   │   │   │     │   │ 
     └───A*──┴───Bd*───A*──┘ 

     ┌───A─────A───┬──Bu───┐ 
     │   │     │   │  │    │ 
    L_n  ├─ H ─┤  s2  │   R_n
     │   │     │   │  │    │ 
     └───Bd*───A*──┴──A*───┘ 

    4. Bu and Bd* are on the different sites and away from the site of H on the same side
     ┌───Bu──┬───A───┬───A─────A───┐ 
     │   │   │   │   │   │     │   │ 
    L_n  │  s2   │  s1   ├─ H ─┤  R_n  (removed cased by gauge)
     │   │   │   │   │   │     │   │ 
     └───A*──┴───Bd*─┴───A*────A*──┘ 

     ┌───A─────A───┬───Bu──┬───A───┐ 
     │   │     │   │   │   │   │   │ 
    L_n  ├─ H ─┤  s1   │  s2   │  R_n
     │   │     │   │   │   │   │   │ 
     └───A*────A*──┴───A*──┴───Bd*─┘ 

    5. Bu and Bd* are on the different sites and away from the site of H on the different side
     ┌───Bu──┬───A─────A───┬───A───┐ 
     │   │   │   │     │   │   │   │ 
    L_n  │  s2   ├─ H ─┤  s2   │  R_n (removed cased by gauge)
     │   │   │   │     │   │   │   │ 
     └───A*──┴───A*────A*──┴───Bd*─┘ 

    ```
"""
function H_eff(k, A, X, VL, H, L_n, R_n, s1, s2, s3)
    sq_L_n = sqrt(L_n)
    sq_R_n = sqrt(R_n)
    inv_sq_L_n = sq_L_n^-1
    inv_sq_R_n = sq_R_n^-1
    Bu = ein"((ba,bcd),de),ef->acf"(inv_sq_L_n,VL,X,inv_sq_R_n)

    # 1. B, B* and H on the same site
    """
        a──┬──c──┬──e         a─────┬─────b  
        │  b     d  │         │     │     │  
        │  ├─────┤  │         │     c     │  
        │  f     g  │         │     │     │  
        h──┴──i──┴──j         d─────┴─────e  
    """
    H_L_AA = einH_L(A, A, A, A, L_n, H)

    HB = ein"((((ah,abc),cde),bfdg),ej),igj->hfi"(L_n,Bu,A,H,R_n,conj(A))                  + 
         ein"((((ah,abc),cde),bfdg),ej),hfi->igj"(L_n,Bu,A,H,R_n,conj(A)) * exp(1.0im * k) + 
         ein"((((ah,abc),cde),bfdg),ej),igj->hfi"(L_n,A,Bu,H,R_n,conj(A)) * exp(1.0im *-k) + 
         ein"((((ah,abc),cde),bfdg),ej),hfi->igj"(L_n,A,Bu,H,R_n,conj(A))                  
    
    # 2. B and B* are on the same site but away from the site of H
    H_R_AA = einH_R(A, A, A, A, R_n, H)
    HB += einL_s_A_R(H_L_AA, s1, Bu, R_n)
    HB += einL_A_s_R(L_n, Bu, s1, H_R_AA)
    
    # 3. one of B and B* on the same site of H
    """
        a──┬──c──┬──e
        │  b     d  │ 
        │  ├─────┤  │ 
        │  f     g  │ 
        h──┴──i──┴──j 
    """
    s3_Bu_A_R = eins_A_R(s3, Bu, A, R_n)
    HB += einL_s_A_R(einH_L(Bu, A, A, A, L_n, H), s2, A, R_n) * exp(2.0im * k)
    HB += einL_s_A_R(einH_L(A, Bu, A, A, L_n, H), s2, A, R_n) * exp(1.0im * k)
    HB += ein"((((ah,abc),cde),bfdg),ej),igj->hfi"(L_n,A,A,H,s3_Bu_A_R,conj(A)) * exp(2.0im *-k)
    HB += ein"((((ah,abc),cde),bfdg),ej),hfi->igj"(L_n,A,A,H,s3_Bu_A_R,conj(A)) * exp(1.0im *-k)
    
    # 4. B and B* are on the different sites and away from the site of H on the same side
    """
        a──┬──b──┬──c
        │  │     │  
        │  │     d  
        │  │     │  
        e──┴──f──┴──g 
    """
    HB += einL_s_A_R(ein"((ae,aebf),bdc),fdg->cg"(H_L_AA,s1,Bu,conj(A)), s2, A, R_n) * exp(1.0im * k)
    HB += einL_s_A_R(H_L_AA, s1, A, s3_Bu_A_R) * exp(1.0im *-k)

    HB = ein"(ad,acb),be->dce"(L_n^-1, HB, R_n^-1)
    Y = ein"((ba,bcd),acf),de->fe"(sq_L_n,HB,conj(VL),sq_R_n)
    # @show norm(HB-ein"((ba,bcd),de),ef->acf"(inv_sq_L_n,VL,Y,inv_sq_R_n))

    return Y
end

function excitation_spectrum(k, A, H)
    χ, D, _ = size(A)
    
    L_n, R_n = env_norm(A)
    s1       = sum_series(     A, L_n, R_n)
    s2, s3   = sum_series_k(k, A, L_n, R_n)
    VL       = initial_VL(A, L_n)

    X = zeros(ComplexF64, χ*(D-1), χ)
    X[1] = 1.0
    X /= sqrt(ein"ab,ab->"(X,conj(X))[])
    H -= energy_gs(A, H) * ein"ab,cd->abcd"(I(D),I(D)) # critical subtraction for the energy gap

    X = H_eff(k, A, X, VL, H, L_n, R_n, s1, s2, s3)
    Δ, Y, info = eigsolve(x -> H_eff(k, A, x, VL, H, L_n, R_n, s1, s2, s3), X, 10, :SR; ishermitian = false, maxiter = 100)
    # @assert info.converged == 1
    return Δ, Y, info
end