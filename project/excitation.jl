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

function main()
    parsed_args = parse_commandline()
    config_file = parsed_args["config"]
    @time excitation_spectrum_canonical_MPO(config_file)
end

main()