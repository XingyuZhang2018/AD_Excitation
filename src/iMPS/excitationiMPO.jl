export excitation_spectrum_MPO

"""
    energy_gs_MPO(A, M, key)

ground state energy

a ────┬──── c 
│     b     │ 
├─ d ─┼─ e ─┤ 
│     g     │ 
f ────┴──── h 

"""
function energy_gs_MPO(A, M, key)
    E, Ǝ = envir_MPO(A, M, key)
    e = ein"(((adf,abc),dgeb),ceh),fgh -> "(E,A,M,Ǝ,conj(A))[]
    n = ein"abc,abc -> "(E,Ǝ)[]
    @show n
    return e/n
end

# function envir_MPO(A, M, key)
#     D, χ, infolder, outfolder = key
#     Zygote.@ignore begin
#         in_chkp_file = joinpath([infolder,"env","MPO_D$(D)_χ$(χ).jld2"]) 
#         if isfile(in_chkp_file)
#             # println("environment load from $(in_chkp_file)")
#             E,Ǝ = load(in_chkp_file)["env"]
#         else
#             E = _arraytype(A)(rand(eltype(A), size(A,1), size(M,1), size(A,1)))
#             Ǝ = _arraytype(A)(rand(eltype(A), size(A,3), size(M,3), size(A,3)))
#         end 
#     end
#     _, E = env_E(A, conj(A), M, E)
#     _, Ǝ = env_Ǝ(A, conj(A), M, Ǝ)
    
#     E /= ein"abc,abc->"(E, Ǝ)[]
#     # Zygote.@ignore begin
#     #     out_chkp_file = joinpath([outfolder,"env","MPO_D$(D)_χ$(χ).jld2"]) 
#     #     save(out_chkp_file, "env", (E, Ǝ))
#     # end
#     # c,ɔ = env_norm(A)
#     # σx, σz = 2*const_Sx(1/2), 2*const_Sz(1/2)
#     # Eo /= ein"abc,abc->"(Eo,Ǝo)[]
#     # c = Eo[:,3,:]
#     # ɔ = Ǝo[:,1,:]
#     # c /= ein"ab,ab->"(c, ɔ)[]
#     # YR = ein"((abc,bd),cf),edf->ae"(A,σx,ɔ,conj(A)) + ein"(((((cde,dg),ej),igj),abc),bf),hfi->ah"(A,σz,ɔ,conj(A),A,σz,conj(A))
#     # @show ein"ab,ab->"(c,YR)[] 
#     # E = zero(Eo)
#     # E[:,5,:] = Eo[:,5,:]
#     # Ǝ = zero(Ǝo)
#     # Ǝ[:,1,:] = Ǝo[:,1,:]
#     return E, Ǝ
# end

function envir_MPO(A, M, key)
    D, χ, infolder, outfolder = key
    σx, σz = 2*const_Sx(1/2), 2*const_Sz(1/2)
    E = zeros(ComplexF64, χ,3,χ)
    Ǝ = zeros(ComplexF64, χ,3,χ)
    c,ɔ = env_norm(A)

    YL = -ein"(abc,bd),(ae,edf)->cf"(A,σx,c,conj(A)) - ein"(((((abc,bf),ah),hfi),cde),dg),igj->ej"(A,σz,c,conj(A),A,σz,conj(A))
    bL = YL - ein"ab,ab->"(YL,ɔ)[] * c
    E[:,1,:], info1 = linsolve(E->E - ein"abc,(ad,dbe)->ce"(A,E,conj(A)) + ein"ab,ab->"(E, ɔ)[] * c, bL)
    @assert info1.converged == 1
    E[:,2,:] = ein"(abc,bd),(ae,edf)->cf"(A,σz,c,conj(A))
    E[:,3,:] = c

    YR = -ein"((abc,bd),cf),edf->ae"(A,σx,ɔ,conj(A)) - ein"(((((cde,dg),ej),igj),abc),bf),hfi->ah"(A,σz,ɔ,conj(A),A,σz,conj(A))
    bR = YR - ein"ab,ab->"(c,YR)[] * ɔ
    Ǝ[:,1,:] = ɔ
    Ǝ[:,2,:] = -ein"((abc,bd),cf),edf->ae"(A,σz,ɔ,conj(A))
    Ǝ[:,3,:], info2 = linsolve(Ǝ->Ǝ - ein"(abc,ce),dbe->ad"(A,Ǝ,conj(A)) + ein"ab,ab->"(c, Ǝ)[] * ɔ, bR)
    @assert info2.converged == 1

    # @show norm(ein"((abc,ceh),dgeb),fgh -> adf"(A,Ǝ,M,conj(A))[:,3,:] - bR)
    # @show ein"ab,ab->"(c,YR)[] ein"ab,ab->"(YL,ɔ)[] ein"ab,ab->"(c,Ǝ[:,3,:])[] ein"ab,ab->"(E[:,1,:],ɔ)[] 
    # @show ein"(abc,ce),(ad,dbe)->"(A,Ǝ[:,3,:],c,conj(A))[]

    # Eo = zero(E)
    # Eo[:,1,:] = E[:,3,:]
    # Eo[:,2,:] = E[:,3,:]
    # Ǝo = zero(Ǝ)
    # Ǝo[:,1,:] = Ǝ[:,1,:]
    # Ǝo[:,2,:] = Ǝ[:,2,:]
    # Ǝ[:,2,:] = ɔ
    # Ǝ[:,3,:] = ɔ
    # E /= ein"(((adf,abc),dgeb),fgh),ceh -> "(E,A,M,conj(A),Ǝ)[]
    return E, Ǝ
end

"""
    ```
     ┌───B────┬─             a ────┬──── c 
     │   │    │              │     b     │ 
     E   M    s              ├─ d ─┼─ e ─┤ 
     │   │    │              │     g     │ 
     └───A*───┴─             f ────┴──── h  
    ```
"""
function einLB(k, L, B, A, E, M, Ǝ)
    LB, info = linsolve(LB->LB - exp(1.0im * k) * ein"((adf,abc),dgeb),fgh -> ceh"(LB,A,M,conj(A)) + exp(1.0im * k) * ein"abc,abc->"(LB,Ǝ)[]*E, ein"((adf,abc),dgeb),fgh -> ceh"(L,B,M,conj(A)))
    @assert info.converged == 1
    return LB
end

"""
    ```
    ─┬───B───┐               a ────┬──── c
     │   │   │               │     b     │
     s   M   Ǝ               ├─ d ─┼─ e ─┤
     │   │   │               │     g     │
    ─┴───A*──┘               f ────┴──── h 
    ```
"""
function einRB(k, R, B, A, E, M, Ǝ)
    RB, info = linsolve(RB->RB - exp(1.0im *-k) * ein"((abc,ceh),dgeb),fgh -> adf"(A,RB,M,conj(A)) + exp(1.0im *-k) * ein"abc,abc->"(E,RB)[]*Ǝ, ein"((abc,ceh),dgeb),fgh -> adf"(B,R,M,conj(A)))
    @assert info.converged == 1
    return RB
end

"""
    ```
     ┌───A───┐               a ────┬──── c
     │   │   │               │     b     │
     E───M───Ǝ               ├─ d ─┼─ e ─┤
     │   │   │               │     g     │
     └──   ──┘               f ────┴──── h 
    ```
"""
eindB(A, E, M, Ǝ) = ein"((adf,abc),dgeb),ceh->fgh"(E,A,M,Ǝ)

"""
    H_mn = H_eff(k, A, Bu, Bd, H, L_n, R_n, s1, s2, s3)

    get `<Ψₖ(B)|H|Ψₖ(B)>`, including sum graphs form https://arxiv.org/abs/1810.07006 Eq.(268)
"""
function H_eff(k, A, Bu, E, M, Ǝ)
    # 1. B and dB on the same site of M
    HB  = eindB(Bu, E, M, Ǝ)
 
    # 2. B and dB on different sites of M
    HB += eindB(A, einLB(k, E, Bu, A, E, M, Ǝ), M, Ǝ) * exp(1.0im * k) +
          eindB(A, E, M, einRB(k, Ǝ, Bu, A, E, M, Ǝ)) * exp(1.0im *-k)

    return HB
end

"""
    excitation_spectrum(k, A, H, n)

find at least `n` smallest excitation gaps 
"""
function excitation_spectrum_MPO(k, A, model, n::Int = 1;
                             infolder = "./data/", outfolder = "./data/")
     infolder = joinpath( infolder, "$model")
    outfolder = joinpath(outfolder, "$model")

    χ, D, _ = size(A)
    key = D, χ, infolder, outfolder
    M = _arraytype(A)(MPO(model))

    E, Ǝ      = envir_MPO(A, M, key)
    Ln, Rn    = env_norm(A)
    sq_Ln     = sqrt(Ln)
    sq_Rn     = sqrt(Rn)
    inv_sq_Ln = sq_Ln^-1
    inv_sq_Rn = sq_Rn^-1
    VL        = initial_VL(A, Ln)

    X = zeros(ComplexF64, χ*(D-1), χ)
    # X = rand(ComplexF64, χ, D, χ)
    X[1] = 1.0
    # X /= sqrt(ein"ab,ab->"(X,conj(X))[])
    
    function f(X)
        Bu = ein"((ba,bcd),de),ef->acf"(inv_sq_Ln, VL, X, inv_sq_Rn)
        HB = H_eff(k, A, Bu, E, M, Ǝ)
        HB = ein"((ba,bcd),acf),de->fe"(inv_sq_Ln,HB,conj(VL),inv_sq_Rn)
        return HB
    end
    Δ, Y, info = eigsolve(x -> f(x), X, n, :SR; ishermitian = true, maxiter = 100)
    # @assert info.converged == 1
    return Δ, Y, info
end