using AD_Excitation
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--if_json_config"
            arg_type = Bool
            default = false
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
        "--if4site"
            help = "if4site"
            arg_type = Bool
            default = false
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    json_string = JSON.json(parsed_args)

    outfolder = joinpath(outfolder, "$model", "groundstate")
    !isdir(outfolder) && mkpath(outfolder)
    config_file = joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D)_χ$(targχ).json")

    open(config_file, "w") do file
        write(file, json_string)
    end

    if_json_config = parsed_args["if_json_config"]
    if if_json_config
        config = JSON.parsefile(config_file)
        model = eval(Meta.parse(config["model"]))
        maxiter = config["maxiter"]
        tol = config["tol"]
        Ni = config["Ni"]
        Nj = config["Nj"]
        χ = config["χ"]
        atype = eval(Meta.parse(config["atype"]))
        infolder = config["infolder"]
        outfolder = config["outfolder"]
        verbose = config["verbose"]
        if4site = config["if4site"]
    else
        model = eval(Meta.parse(parsed_args["model"]))
        maxiter = parsed_args["maxiter"]
        tol = parsed_args["tol"]
        Ni = parsed_args["Ni"]
        Nj = parsed_args["Nj"]
        χ = parsed_args["chi"]
        atype = eval(Meta.parse(parsed_args["atype"]))
        infolder = parsed_args["infolder"]
        outfolder = parsed_args["outfolder"]
        verbose = parsed_args["verbose"]
        if4site = parsed_args["if4site"]
    end

    find_groundstate(model, alg(maxiter = maxiter, tol = tol);
                     Ni = Ni, Nj = Nj,
                     χ = χ,
                     atype = atype,
                     infolder = infolder,
                     outfolder = outfolder,
                     verbose = verbose,
                     if4site = if4site
                     );
end

main()