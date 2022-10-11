using FileIO
using JLD2
using Optim, LineSearches
using Zygote

function init_mps(;infolder = "./data/", 
                    atype = Array, 
                    verbose::Bool = true,        
                    D::Int = 2, 
                    χ::Int = 5)

    in_chkp_file = infolder*"/D$(D)_χ$(χ).jld2"
    if isfile(in_chkp_file)
        A = atype(load(in_chkp_file)["A"])
        verbose && println("load mps from $in_chkp_file")
    else
        A = atype(rand(ComplexF64,χ,D,χ))
        verbose && println("random initial mps $in_chkp_file")
    end
    _, L_n = norm_L(A, conj(A))
    _, R_n = norm_R(A, conj(A))
    n = ein"(ad,acb),(dce,be) ->"(L_n,A,conj(A),R_n)[]/ein"ab,ab ->"(L_n,R_n)[]
    A /= sqrt(n)
    return A, L_n, R_n
end

"""
    e = energy_gs(T,H)

ground state energy
┌───A─────A───┐          a───┬──c──┬───e
│   │     │   │          │   b     d   │  
L   ├─ H ─┤   R          │   ├─────┤   │  
│   │     │   │          │   f     g   │  
└───A*────A*──┘          h───┴──i──┴───j 
"""
function energy_gs(A, H, L_n, R_n)
    _, L_n = norm_L(A, conj(A), L_n)
    _, R_n = norm_R(A, conj(A), R_n)
    env = ein"((ah,abc),cde),((hfi,igj),ej)->bfdg"(L_n,A,A,conj(A),conj(A),R_n)
    e   = ein"abcd,abcd->"(env,H)[]
    n   = ein"aabb->"(env)[]
    return e/n
end

function optimizeiMPS(A, L_n, R_n; 
        model = Heisenberg(), 
        infolder = "./data/", outfolder = "./data/", 
        optimmethod = LBFGS(m = 20), 
        verbose= true, savefile = true,
        f_tol::Real = 1e-6, 
        opiter::Int = 100)

     infolder = joinpath( infolder, "$model")
    outfolder = joinpath(outfolder, "$model")

    χ, D, _ = size(A)
    H = _arraytype(A)(hamiltonian(model))
    f(A) = real(energy_gs(A, H, L_n, R_n))
    g(A) = Zygote.gradient(f,A)[1]
    res = optimize(f, g, 
        A, optimmethod,inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, outfolder, D, χ, savefile, verbose)),
    )
    A = Optim.minimizer(res)
    e = Optim.minimum(res)
    return A, e
end

"""
    writelog(os::OptimizationState, key=nothing)
return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_χ_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_χ_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, outfolder, D, χ, savefile, verbose)
    message = "$(round(os.metadata["time"],digits=1))    $(os.iteration)    $(round(os.value,digits=15))    $(round(os.g_norm,digits=8))\n"

    if verbose
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end

    if savefile 
        !(isdir(outfolder)) && mkpath(outfolder)
        logfile = open(outfolder*"/D$(D)_χ$(χ).log", "a")
        write(logfile, message)
        close(logfile)
        save(outfolder*"/D$(D)_χ$(χ).jld2", "A", Array(os.metadata["x"]))
    end
    return false
end