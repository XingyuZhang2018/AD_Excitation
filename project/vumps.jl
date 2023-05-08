using AD_Excitation
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--config"
            help = "config file"
            arg_type = String
            default = "./project/config.json"
    end

    return parse_args(s)
end

function main(config_file)
    config = JSON.parsefile(config_file)  
    
    model = eval(Meta.parse(config["model"]))

    Ni = config["mps"]["Ni"]
    Nj = config["mps"]["Nj"]
    χ = config["mps"]["χ"]
    targχ = config["mps"]["targχ"]
    if4site = config["mps"]["if4site"]

    iters = config["vumps"]["iters"]
    tol = config["vumps"]["tol"]
    show_every = config["vumps"]["show_every"]
    
    Random.seed!(config["data"]["seed"]) 
    atype = eval(Meta.parse(config["data"]["atype"])) 
    infolder = config["data"]["infolder"]
    outfolder = config["data"]["outfolder"]

    outfolder = joinpath(outfolder, "$model", "groundstate")
    !isdir(outfolder) && mkpath(outfolder)
    cp(config_file, joinpath(outfolder,"canonical_mps_$(Ni)x$(Nj)_D$(D)_χ$(targχ).json"); force=true)
    
    parsed_args = parse_commandline()
    config_file = parsed_args["config"]
    @time vumps(config_file)
end

main()