module AD_Excitation

    using Parameters 
    using Zygote

    export find_groundstate, VUMPS, ADMPS, IDMRG1
    export checkpoint
    #default settings
    module Defaults
        const eltype = ComplexF64
        const maxiter = 100
        const tol = 1e-8
        const verbose = true
        const atype = Array    
        const infolder = "../data/"
        const outfolder = "../data/"
    end
    
    include("patch.jl")
    include("hamiltonianmodels.jl")

    include("iMPS/environment/environment.jl")
    include("iMPS/environment/quasi_environment.jl")

    include("iMPS/MPO.jl")
    include("iMPS/MPO_2x2.jl")
    include("iMPS/MPO_honeycomb.jl")

    abstract type Algorithm end

    include("iMPS/uniform/variationaliMPS.jl")
    include("iMPS/uniform/excitationiMPS.jl")
    include("iMPS/uniform/excitationiMPO.jl")
    include("iMPS/canonical/vumps.jl")
    include("iMPS/canonical/idmrg.jl")
    include("iMPS/canonical/excitationiMPO.jl")
    include("iMPS/canonical/observable.jl")
    include("iMPS/autodiff.jl")

end
