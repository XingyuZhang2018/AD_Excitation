using ArgParse
using Random
using AD_Excitation
using CUDA

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--W"
            help = "helix width"
            arg_type = Int
            required = true
        "--chi"
            help = "vumps virtual bond dimension"
            arg_type = Int
            required = true
        "--J2"
            help = "J2"
            arg_type = Float64
            required = true
        "--folder"
            help = "folder for output"
            arg_type = String
            default = "./data/"
        "--iters"
            help = "number of vumps iterations"
            arg_type = Int
            default = 100
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    Random.seed!(100)
     W = parsed_args["W"]
     χ = parsed_args["chi"]
    J2 = parsed_args["J2"]
    folder = parsed_args["folder"]
    iters = parsed_args["iters"]
    model = J1J2(W, J2)
    vumps(model; infolder = folder, outfolder = folder,
                 Nj = 1, χ=χ, iters = iters, show_every = 1, tol = 1e-8, atype = CuArray, if4site = true)
end

main()