using FileIO
using JLD2
using Optim, LineSearches
using Zygote

@with_kw struct ADMPS <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    optimmethod = LBFGS(m = 20)
    ifcheckpoint::Bool = false
end

function init_uniform_mps(;D, χ, 
                           atype = Defaults.atype, 
                           infolder = Defaults.infolder,
                           verbose::Bool = Defaults.verbose,
                           if_vumps_init = false
                          )

    if if_vumps_init
        in_chkp_file = joinpath(infolder, "canonical_mps_1x1_D$(D)_χ$(χ).jld2")
        A, = load(in_chkp_file)["ALCAR"]
        A = atype(reshape(A, χ,D,χ))
        verbose && println("load canonical mps from $in_chkp_file")
    else
        in_chkp_file = joinpath(infolder, "uniform_mps_D$(D)_χ$(χ).jld2")
        if isfile(in_chkp_file)
            A = atype(load(in_chkp_file)["A"])
            verbose && println("load mps from $in_chkp_file")
        else
            A = atype(randn(ComplexF64, χ,D,χ))
            verbose && println("random initial mps $in_chkp_file")
        end
    end

    _, L_n = norm_L(A, conj(A))
    _, R_n = norm_R(A, conj(A))
    n = Array(ein"(ad,acb),(dce,be) ->"(L_n,A,conj(A),R_n))[]/Array(ein"ab,ab ->"(L_n,R_n))[]
    A /= sqrt(n)
    return A
end

"""
    e = energy_gs(A, H, key)

ground state energy
````
    ┌───A─────A───┐          a───┬──c──┬───e
    │   │     │   │          │   b     d   │  
    L   ├─ H ─┤   R          │   ├─────┤   │  
    │   │     │   │          │   f     g   │  
    └───A*────A*──┘          h───┴──i──┴───j 
````
"""
function energy_gs(A, H; infolder = Defaults.infolder, outfolder = Defaults.outfolder)
    L_n, R_n = envir(A; infolder=infolder, outfolder=outfolder)
    env = ein"((ah,abc),cde),((hfi,igj),ej)->bfdg"(L_n,A,A,conj(A),conj(A),R_n)
    e   = ein"abcd,abcd->"(env,H)[]
    n   = ein"aabb->"(env)[]
    return e/n
end

"""
    energy_gs_MPO(A, M, key)
ground state energy
````
    ┌────A────┐        a ────┬──── c 
    │    │    │        │     b     │ 
    E────M────Ǝ        ├─ d ─┼─ e ─┤ 
    │    │    │        │     g     │ 
    └────A*───┘        f ────┴──── h 
````
"""
function energy_gs_MPO(A, M; ifcheckpoint = false, infolder = Defaults.infolder, outfolder = Defaults.outfolder)
    L_n, R_n = envir(A; infolder=infolder, outfolder=outfolder)
    n = Array(ein"(ad,acb),(dce,be) ->"(L_n,A,conj(A),R_n))[]/Array(ein"ab,ab ->"(L_n,R_n))[]
    A /= sqrt(n)
    n = Array(ein"((ad,acb),dce),be->"(L_n,A,conj(A),R_n))[]
    L_n /= n

    E, Ǝ = envir_MPO(A, M, L_n, R_n; ifcheckpoint = ifcheckpoint)
    e = Array(ein"(((adf,abc),dgeb),ceh),fgh -> "(E,A,M,Ǝ,conj(A)))[]
    n = Array(ein"abc,abc -> "(E,Ǝ))[]
    # @show e n
    return e-n
end

function find_groundstate(model, alg::ADMPS;
                          Ni::Int = 1, Nj::Int = 1,
                          χ::Int = 16,
                          atype = Defaults.atype,
                          infolder::String = Defaults.infolder,
                          outfolder::String = Defaults.outfolder,
                          verbose::Bool = Defaults.verbose,
                          ifsave = true,
                          ifMPO = false, 
                          if4site = false,
                          if_vumps_init = false
                          )

     infolder = joinpath( infolder, "$model", "groundstate")
    outfolder = joinpath(outfolder, "$model", "groundstate")

    if if4site
        H = atype(MPO_2x2(model))
    else
        H = ifMPO ? atype(MPO(model)) : atype(hamiltonian(model))
    end

    D = size(H,2)
    f(A) = ifMPO ? (if4site ? real(energy_gs_MPO(atype(A), H; ifcheckpoint=alg.ifcheckpoint, infolder=infolder, outfolder=outfolder))/4 : real(energy_gs_MPO(atype(A), H; ifcheckpoint=alg.ifcheckpoint, infolder=infolder, outfolder=outfolder))) : real(energy_gs(atype(A), H; infolder=infolder, outfolder=outfolder))
    A = init_uniform_mps(;D, χ, 
                          atype = atype, 
                          infolder = infolder,
                          verbose = verbose,
                          if_vumps_init = if_vumps_init
                         )
    
    g(A) = Zygote.gradient(f,atype(A))[1]
    res = optimize(f, g, 
                   A, alg.optimmethod, inplace = false,
                   Optim.Options(f_tol = alg.tol, iterations = alg.maxiter,
                   extended_trace=true,
                   callback=os->writelog(os, outfolder, D, χ, ifsave, verbose)),
    )
    e = Optim.minimum(res)
    return e
end

"""
    writelog(os::OptimizationState, key=nothing)
return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_χ_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_χ_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, outfolder, D, χ, ifsave, verbose)
    message = "$(round(os.metadata["time"],digits=1))    $(os.iteration)    $(round(os.value,digits=15))    $(round(os.g_norm,digits=8))\n"

    if verbose
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end

    if ifsave 
        !(isdir(outfolder)) && mkpath(outfolder)
        logfile = open(outfolder*"/uniform_mps_D$(D)_χ$(χ).log", "a")
        write(logfile, message)
        close(logfile)
        save(outfolder*"/uniform_mps_D$(D)_χ$(χ).jld2", "A", Array(os.metadata["x"]))
    end
    return false
end