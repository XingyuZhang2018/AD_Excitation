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

