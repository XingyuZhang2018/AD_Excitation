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
        "--model"
            help = "model"
            arg_type = String
        "--atype"
            help = "atype"
            arg_type = String
            default = "CuArray"
        "--kx"
            arg_type = Float64
            required = true
        "--ky"
            arg_type = Float64
            required = true
        "--n"
            help = "howmany state"
            arg_type = Int
            default = 1
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
        "--ifmerge"
            help = "ifmerge"
            arg_type = Bool
            default = false
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    model = eval(Meta.parse(parsed_args["model"]))
    atype = eval(Meta.parse(parsed_args["atype"]))
    seed = parsed_args["seed"]
    kx = parsed_args["kx"]
    ky = parsed_args["ky"]
    n = parsed_args["n"]
    Ni = parsed_args["Ni"]
    Nj = parsed_args["Nj"]
    χ = parsed_args["chi"]
    infolder = parsed_args["infolder"]
    outfolder = parsed_args["outfolder"]
    verbose = parsed_args["verbose"]
    if4site = parsed_args["if4site"]
    ifmerge = parsed_args["ifmerge"]

    Random.seed!(seed)
    @time excitation_spectrum_canonical_MPO(model, (kx*2*pi/model.W,ky*2*pi/model.W), n;
                                            Ni = Ni, Nj = Nj,
                                            χ = χ,
                                            atype = atype,
                                            ifmerge = ifmerge,
                                            if4site = if4site,
                                            infolder = infolder, outfolder = outfolder)
end

main()