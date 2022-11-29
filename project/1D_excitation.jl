using AD_Excitation
using Test
using Random

@testset "1D XXZ S=1/2 excitation with $atype" for atype in [Array]
    Random.seed!(1000)
    D,χ = 2,16
    gap1 = []
    k = 0
    for Δ in 1.0:0.2:2.0
        model = XXZ(Δ)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        Δ, = excitation_spectrum(k, A, model)
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
        Δ, = excitation_spectrum(k, A, model, 30)
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
    for k in 0:pi/12:0
        model = Heisenberg(0.5,1,1.0,-1.0,-1.0)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        Δ, = @time excitation_spectrum(k, A, Heisenberg(0.5,1,1.0,1.0,1.0), 10)
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
    D,χ = 3,16
    s1 = []
    for k in pi:pi/24:pi
        model = Heisenberg(1.0)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        Δ, = @time excitation_spectrum(k, A, model, 2)
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

@testset "1D TFIsing S=1/2 excitation with $atype" for atype in [Array]
    Random.seed!(100)
    D,χ = 2,4
    s1 = []
    for k in 0:pi/12:pi
        model = TFIsing(1/2,1.0)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")
        Δ, = @time excitation_spectrum(k, A, model, 1)
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

@testset "1D TFIsing S=1/2 excitation with $atype" for atype in [Array]
    Random.seed!(100)
    χ = 64
    s1 = []
    for k in 0:pi/12:0
        model = Heisenberg(0.5,1,1.0,-1.0,-1.0)
        Δ, = @time excitation_spectrum_MPO(k, model, 1; gs_from = "u", χ = χ)
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

@testset "excitation_spectrum_canonical_MPO" begin
    model = Heisenberg(0.5,1,1.0,-1.0,-1.0)
    
    k = pi
    s1 = []
    for χ in 2 .^ (6:6)
        @show χ
        Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1;
                                                             χ=χ)
        push!(s1,real(Δ))
    end
    for i in 1:length(s1)
        print("$(s1[i][1]),")
    end
end