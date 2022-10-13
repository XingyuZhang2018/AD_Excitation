"""
    B = initial_excitation(A)

    Construct a `B` tensor that is orthogonal to A:    
    ````
                                                        a──────┬──────b
    ┌──B──┐                                             │      │      │
    L  │  R  = 0                                        │      c      │
    └──A*─┘                                             │      │      │
                                                        d──────┴──────e                                         
    ````
"""
function initial_excitation(A)
    _, L_n = norm_L(A, conj(A))
    _, R_n = norm_R(A, conj(A))
    χ,D,_ = size(A)
    Bs = []
    for i in 1:D*χ^2-1
        B = zero(A)
        B[i] = 1
        B -= ein"((ad,acb),dce),be->"(L_n,B,conj(A),R_n)[] * A / ein"((ad,acb),dce),be->"(L_n,A,conj(A),R_n)[]
        @assert norm(ein"((ad,acb),dce),be->"(L_n,B,conj(A),R_n)[]) < 1e-8
        push!(Bs, B)
    end
    return Bs
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
    s1 = sum_series(A, L_n, R_n)

    s1 = sum of `    工` series:
    ````
        ─       ┬           ┬┬          ┬┬┬               ┬┬┬...┬
        +       │   +       ││  +       │││  + ... +      │││...│ + ...
        ─       ┴           ┴┴          ┴┴┴               ┴┴┴...┴
    ````
"""
function sum_series(A, L_n, R_n)
    χ = size(A, 1)
    工_I = ein"ab,cd->acbd"(I(χ), I(χ))
    工 = ein"acb,dce->adbe"(A, conj(A))
    rl = ein"ab,cd->abcd"(R_n, L_n)
    a1 = 工_I - (工 - rl)
    b1 = 工_I - rl
    s1, info1 = linsolve(x->ein"adbe,becf->adcf"(a1,x), b1)
    @assert info1.converged == 1
    return s1
end

"""
    s2, s3 = sum_series_k(k, A)

    s2 = sum of `eⁱᵏ 工` series: 
    ````
         ─       ┬           ┬┬          ┬┬┬               ┬┬┬...┬
    eⁱ⁰ᵏ   + eⁱ¹ᵏ│   +  eⁱ²ᵏ ││  +  eⁱ³ᵏ │││  + ... + eⁱⁿᵏ │││...│ + ...
         ─       ┴           ┴┴          ┴┴┴               ┴┴┴...┴
    ````

    s3 = sum of `eⁱ⁻ᵏ 工` series:
"""
function sum_series_k(k, A, R_n, L_n)
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
    H_L = einH_L(A1, A2, A3, A4, L_n, H)

    ````
     ┌───A1────A2──             a──┬──c──┬──e
     │   │     │                │  b     d  │ 
    L_n  ├─ H ─┤                │  ├─────┤  │ 
     │   │     │                │  f     g  │ 
     └───A3*───A4*─             h──┴──i──┴──j 
    ````
"""
einH_L(A1, A2, A3, A4, L_n, H) = ein"(((ah,abc),cde),(hfi,igj)),bfdg->ej"(L_n,A1,A2,conj(A3),conj(A4),H)

"""
    H_R = einH_R(A1, A2, A3, A4, R_n, H)

    ````
    ───A1────A2──┐              a──┬──c──┬──e
       │     │   │              │  b     d  │ 
       ├─ H ─┤  R_n             │  ├─────┤  │ 
       │     │   │              │  f     g  │ 
    ───A3*───A4*─┘              h──┴──i──┴──j 
    ````
"""
einH_R(A1, A2, A3, A4, R_n, H) = ein"((abc,cde),((hfi,igj),ej)),bfdg->ah"(A1,A2,conj(A3),conj(A4),R_n,H)

"""
    L_A_s = einL_A_s(L_n, A2, A2, s)

    ````
     ┌───A1───┬─            a──┬──b──┬──c
     │   │    │             │  │     │  
    L_n  │    s             │  d     │  
     │   │    │             │  │     │  
     └───A2*──┴─            e──┴──f──┴──g 
    ````
"""
einL_A_s(L_n, A1, A2, s) = ein"((ae,adb),edf),bfcg->cg"(L_n, A1, conj(A2), s)

"""
    s_A_R = eins_A_R(s, A1, A2, R_n)

    ````
    ─┬───A1──┐           a──┬──b──┬──c
     │   │   │              │     │  │
     s   │  R_n             │     d  │
     │   │   │              │     │  │
    ─┴───A2*─┘           e──┴──f──┴──g 
    ````
"""
eins_A_R(s, A1, A2, R_n) = ein"aebf,(bdc,(fdg,cg))->ae"(s, A1, conj(A2), R_n)

"""
    N_mn = N_eff(k, A, Bu, Bd, L_n, R_n, s2, s3)

    get `<Ψₖ(B)|Ψₖ(B)>`, including sum graphs: 
    ````
    1. Bu, Bd* on the same site
     ┌───Bu──┐             a─────┬─────b  
     │   │   │             │     │     │ 
    L_n  │  R_n            │     c     │ 
     │   │   │             │     │     │ 
     └───Bd*─┘             d─────┴─────e 

    2. Bu, Bd* on the different sites
     ┌───Bu──┬───A───┐ 
     │   │   │   │   │ 
    L_n  │  s2   │  R_n 
     │   │   │   │   │ 
     └───A*──┴───Bd*─┘ 

    ````
"""
function N_eff(k, A, Bu, Bd, L_n, R_n, s2, s3)
    N_mn = 0

    # 1. B, B* on the same site
    N_mn += ein"((ad,acb),dce),be->"(L_n, Bu, conj(Bd), R_n)[]

    # 2. B, B* on the different sites
    N_mn += ein"((ad,acb),dce),be->"(einL_A_s(L_n, Bu, A, s2), A, conj(Bd), R_n)[] * exp(1.0im * k)
    N_mn += ein"((ad,acb),dce),be->"(einL_A_s(L_n, A, Bd, s3), Bu, conj(A), R_n)[] * exp(1.0im *-k)

    return N_mn
end

"""
    H_mn = H_eff(k, A, Bu, Bd, H, L_n, R_n, s1, s2, s3)

    get `<Ψₖ(B)|H|Ψₖ(B)>`, including sum graphs: 
    ````
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
    L_n  │  s2   ├─ H ─┤  R_n 
     │   │   │   │     │   │ 
     └───A*──┴───Bd*───A*──┘ 

    4. Bu and Bd* are on the different sites and away from the site of H on the same side
     ┌───Bu──┬───A───┬───A─────A───┐ 
     │   │   │   │   │   │     │   │ 
    L_n  │  s2   │  s1   ├─ H ─┤  R_n 
     │   │   │   │   │   │     │   │ 
     └───A*──┴───Bd*─┴───A*────A*──┘ 

    5. Bu and Bd* are on the different sites and away from the site of H on the different side
     ┌───Bu──┬───A─────A───┬───A───┐ 
     │   │   │   │     │   │   │   │ 
    L_n  │  s2   ├─ H ─┤  s2   │  R_n 
     │   │   │   │     │   │   │   │ 
     └───A*──┴───A*────A*──┴───Bd*─┘ 

    ````
"""
function H_eff(k, A, Bu, Bd, H, L_n, R_n, s1, s2, s3)
    H_mn = 0

    # 1. B, B* and H on the same site
    H_L = einH_L(Bu,A,Bd,A,L_n,H)                  + 
          einH_L(Bu,A,A,Bd,L_n,H) * exp(1.0im * k) + 
          einH_L(A,Bu,Bd,A,L_n,H) * exp(1.0im *-k) + 
          einH_L(A,Bu,A,Bd,L_n,H)
    H_mn += ein"ab,ab->"(H_L, R_n)[]

    # 2. B and B* are on the same site but away from the site of H
    H_L_AA = einH_L(A, A, A, A, L_n, H)
    H_R_AA = einH_R(A, A, A, A, R_n, H)
    H_mn += ein"ab,ab->"(H_L_AA, eins_A_R(s1, Bu, Bd, R_n))[]
    H_mn += ein"ab,ab->"(einL_A_s(L_n, Bu, Bd, s1), H_R_AA)[]
    
    # 3. one of B and B* on the same site of H
    H_mn += ein"ab,ab->"(einH_L(Bu, A, A, A, L_n, H), eins_A_R(s2, A, Bd, R_n))[] * exp(2.0im * k)
    H_mn += ein"ab,ab->"(einH_L(A, Bu, A, A, L_n, H), eins_A_R(s2, A, Bd, R_n))[] * exp(1.0im * k)
    H_mn += ein"ab,ab->"(einH_L(A, A, Bd, A, L_n, H), eins_A_R(s3, Bu, A, R_n))[] * exp(2.0im *-k)
    H_mn += ein"ab,ab->"(einH_L(A, A, A, Bd, L_n, H), eins_A_R(s3, Bu, A, R_n))[] * exp(1.0im *-k)

    H_mn += ein"ab,ab->"(einL_A_s(L_n, Bu, A, s2), einH_R(A, A, Bd, A, R_n, H))[] * exp(1.0im * k)
    H_mn += ein"ab,ab->"(einL_A_s(L_n, Bu, A, s2), einH_R(A, A, A, Bd, R_n, H))[] * exp(2.0im * k)
    H_mn += ein"ab,ab->"(einL_A_s(L_n, A, Bd, s3), einH_R(Bu, A, A, A, R_n, H))[] * exp(1.0im *-k)
    H_mn += ein"ab,ab->"(einL_A_s(L_n, A, Bd, s3), einH_R(A, Bu, A, A, R_n, H))[] * exp(2.0im *-k)

    # 4. B and B* are on the different sites and away from the site of H on the same side
    H_mn += ein"ab,ab->"(H_L_AA, eins_A_R(s1, Bu, A, eins_A_R(s2, A, Bd, R_n)))[] * exp(1.0im * k)
    H_mn += ein"ab,ab->"(H_L_AA, eins_A_R(s1, A, Bd, eins_A_R(s3, Bu, A, R_n)))[] * exp(1.0im *-k)

    H_mn += ein"ab,ab->"(einL_A_s(einL_A_s(L_n, Bu, A, s2), A, Bd, s1), H_R_AA)[] * exp(1.0im * k)
    H_mn += ein"ab,ab->"(einL_A_s(einL_A_s(L_n, A, Bd, s3), Bu, A, s1), H_R_AA)[] * exp(1.0im *-k)

    # 5. B and B* are on the different sites and away from the site of H on the different sides
    H_mn += ein"ab,ab->"(einH_L(A, A, A, A, einL_A_s(L_n, Bu, A, s2), H), eins_A_R(s2, A, Bd, R_n))[] * exp(3.0im * k)
    H_mn += ein"ab,ab->"(einH_L(A, A, A, A, einL_A_s(L_n, A, Bd, s3), H), eins_A_R(s3, Bu, A, R_n))[] * exp(3.0im *-k)

    return H_mn
end

function excitation_spectrum(k, A, H)
    χ, D, _ = size(A)
    M = χ^2*D-1
    L_n, R_n = env_norm(A)
    Bs = initial_excitation(A)

    s1       = sum_series(     A, L_n, R_n)
    s2, s3   = sum_series_k(k, A, R_n, L_n)

    H_mn = zeros(ComplexF64, M, M)
    N_mn = zeros(ComplexF64, M, M)

    for i in 1:M, j in 1:M
        H_mn[i,j] = H_eff(k, A, Bs[i], Bs[j], H, L_n, R_n, s1, s2, s3)
        N_mn[i,j] = N_eff(k, A, Bs[i], Bs[j],    L_n, R_n,     s2, s3)
    end

    @show size(H_mn) size(N_mn)
    F = eigen(H_mn, N_mn)
    return F
end