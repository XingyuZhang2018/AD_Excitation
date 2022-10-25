using AD_Excitation
using Test

@testset "1D XXZ S=1/2 excitation with $atype" for atype in [Array]
    Random.seed!(1000)
    D,χ = 2,16
    gap1 = []
    for Δ in 1.0:0.2:2.0
        model = XXZ(Δ)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        k = pi
        H = hamiltonian(model)
        Δ, = excitation_spectrum(k, A, H)
        push!(gap1,real(Δ[1]))
    end
    @show gap1
end

@testset "1D XXZ S=1/2 excitation with $atype" for atype in [Array]
    Random.seed!(100)
    D,χ = 2,16
    s1 = []
    for k in 0:pi/12:pi
        @show k
        model = XXZ(1.0)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        H = hamiltonian(model)
        Δ, = excitation_spectrum(k, A, H, 30)
        push!(s1,real(Δ))
    end
    # for i in 1:length(s1)
    #     print("{")
    #     for j in s1[i]
    #         print("$j,")
    #     end
    #     print("},")
    # end
    for i in 1:length(s1)
        print("$(s1[i][1]),")
    end
end

@testset "1D Heisenberg S=1/2 excitation with $atype" for atype in [Array]
    Random.seed!(100)
    D,χ = 2,16
    s1 = []
    for k in pi:pi/12:pi
        model = Heisenberg(0.5)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        H = hamiltonian(model)
        Δ, = excitation_spectrum(k, A, H, 30)
        push!(s1,real(Δ))
    end
    for i in 1:length(s1)
        print("{")
        for j in s1[i]
            print("$j,")
        end
        print("},")
    end
    # for i in 1:length(s1)
    #     print("$(s1[i][1]),")
    # end
end

@testset "1D Heisenberg S=1 excitation with $atype" for atype in [Array]
    Random.seed!(100)
    D,χ = 3,32
    s1 = []
    for k in 0:pi/24:pi
        model = Heisenberg(1.0)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        H = hamiltonian(model)
        Δ, = excitation_spectrum(k, A, H, 30)
        push!(s1,real(Δ))
    end
    for i in 1:length(s1)
        print("{")
        for j in s1[i]
            print("$j,")
        end
        print("},")
    end
    for i in 1:length(s1)
        print("$(s1[i][1]),")
    end
end