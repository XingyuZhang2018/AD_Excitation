using ArgParse
using Random
using AD_Excitation
using AD_Excitation: init_canonical_mps
using CUDA

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--W"
            help = "helix width"
            arg_type = Int
            required = true
        "--Jx"
            help = "coupling Jx"
            arg_type = Float64
            required = true
        "--Jy"
            help = "coupling Jy"
            arg_type = Float64
            required = true
        "--Jz"
            help = "coupling Jz"
            arg_type = Float64
            required = true
        "--kx"
            arg_type = Float64
            required = true
        "--ky"
            arg_type = Float64
            required = true
        "--N"
            help = "howmany state"
            arg_type = Int
            required = true
        "--chi"
            help = "vumps virtual bond dimension"
            arg_type = Int
            required = true
        "--folder"
            help = "folder for output"
            arg_type = String
            default = "./data/"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    Random.seed!(100)
    W = parsed_args["W"]
    Jx = parsed_args["Jx"]
    Jy = parsed_args["Jy"]
    Jz = parsed_args["Jz"]
    N = parsed_args["N"]
    kx = parsed_args["kx"]
    ky = parsed_args["ky"]
    χ = parsed_args["chi"]
    folder = parsed_args["folder"]
    model = Heisenberg(0.5,W,Jx,Jy,Jz)
    Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, (kx*2*pi/W,ky*2*pi/W), N; 
                                                         infolder = folder, outfolder = folder,
                                                         χ=χ, atype = CuArray)
end

main()