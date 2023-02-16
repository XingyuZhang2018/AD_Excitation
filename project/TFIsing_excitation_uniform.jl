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
        "--lambda"
            help = "TFIsing transverse field"
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
        "--D"
            help = "ipeps virtual bond dimension"
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
    λ = parsed_args["lambda"]
    N = parsed_args["N"]
    kx = parsed_args["kx"]
    ky = parsed_args["ky"]
    D = parsed_args["D"]
    χ = parsed_args["chi"]
    folder = parsed_args["folder"]
    model = TFIsing(0.5,W,λ)
    AL, C, AR = init_canonical_mps(;infolder = "$folder/$model/", 
                                    atype = CuArray, 
                                    Ni=1,Nj=1,       
                                    D = D, 
                                    χ = χ)
    A = AL[:,:,:,1,1]
    Δ, Y, info = @time excitation_spectrum_MPO((kx*2*pi/W,ky*2*pi/W), A, model, N)
    logfile = open("$folder/$model/D$(D)_χ$(χ)_W$(W)_kx$(kx)_ky$(ky).log", "w")
    write(logfile, "$(Δ)")
    close(logfile)
end

main()