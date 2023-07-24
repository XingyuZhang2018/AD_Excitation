using AD_Excitation
using ArgParse
using CUDA
using Random

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--seed"
            help = "seed"
            arg_type = Int
            default = 100
        "--alg"
            help = "algorithm"
            arg_type = String
            default = "VUMPS"
        "--model"
            help = "model"
            arg_type = String
        "--maxiter"
            help = "number of iterations"
            arg_type = Int
            default = 1000
        "--tol"
            help = "tolerance"
            arg_type = Float64
            default = 1e-8
        "--Ni"
            help = "Ni"
            arg_type = Int
            default = 1
        "--Nj"
            help = "Nj"
            arg_type = Int
            default = 1
        "--chi"
            help = "vumps virtual bond dimension"
            arg_type = Int
        "--atype"
            help = "atype"
            arg_type = String
            default = "CuArray"
        "--infolder"
            help = "infolder"
            arg_type = String
            default = "../data/"
        "--outfolder"
            help = "outfolder"
            arg_type = String
            default = "../data/"
        "--verbose"
            help = "verbose"
            arg_type = Bool
            default = true
        "--if2site"
            help = "if2site"
            arg_type = Bool
            default = false
        "--if4site"
            help = "if4site"
            arg_type = Bool
            default = false
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    alg = eval(Meta.parse(parsed_args["alg"]))
    model = eval(Meta.parse(parsed_args["model"]))
    atype = eval(Meta.parse(parsed_args["atype"]))
    seed = parsed_args["seed"]
    maxiter = parsed_args["maxiter"]
    tol = parsed_args["tol"]
    Ni = parsed_args["Ni"]
    Nj = parsed_args["Nj"]
    χ = parsed_args["chi"]
    infolder = parsed_args["infolder"]
    outfolder = parsed_args["outfolder"]
    verbose = parsed_args["verbose"]
    if2site = parsed_args["if2site"]
    if4site = parsed_args["if4site"]

    Random.seed!(seed)
    find_groundstate(model, alg(maxiter = maxiter, tol = tol);
                     Ni = Ni, Nj = Nj,
                     χ = χ,
                     atype = atype,
                     infolder = infolder,
                     outfolder = outfolder,
                     verbose = verbose,
                     if2site = if2site,
                     if4site = if4site
                     );
end

main()