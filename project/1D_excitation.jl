using AD_Excitation
using Test

@testset "1D XXZ S=1/2 excitation with $atype" for atype in [Array]
    Random.seed!(1000)
    D,χ = 2,8
    gap1 = []
    for Δ in 4.0:0.2:4.0
        model = XXZ(Δ)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        k = pi
        H = hamiltonian(model)
        F, H_mn, N_mn = excitation_spectrum(k, A, H)
        push!(gap1,real(F.values[1]))
    end
    @show gap1
end

@testset "1D Heisenberg S=1/2 excitation with $atype" for atype in [Array]
    Random.seed!(100)
    D,χ = 2,4
    s1 = []
    for k in 0:pi/12:pi
        model = Heisenberg(0.5)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        H = hamiltonian(model)
        F, H_mn, N_mn = excitation_spectrum(k, A, H)
        @show k,real(F.values)
        push!(s1,real(F.values))
    end
    for i in 1:length(s1)
        print("$(s1[i]),")
    end
    for i in 1:length(s1)
        print("$(s1[i][1]),")
    end
end

@testset "1D Heisenberg S=1 excitation with $atype" for atype in [Array]
    Random.seed!(100)
    D,χ = 3,8
    s1 = []
    for k in 0:pi/12:pi
        model = Heisenberg(1.0)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        H = hamiltonian(model)
        F, H_mn, N_mn = excitation_spectrum(k, A, H)
        @show k,real(F.values)
        push!(s1,real(F.values))
    end
    # for i in 1:length(s1)
    #     print("$(s1[i]),")
    # end
    for i in 1:length(s1)
        print("$(s1[i][1]),")
    end
end