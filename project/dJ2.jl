using AD_Excitation
using AD_Excitation: dJ2
using CUDA

value = []
for J2 = 0.45:0.01:0.55
    model = J1J2(6,J2)
    v = @time dJ2(model; 
                  Nj = 1, χ = 512, 
                  infolder = "./data",
                  atype = CuArray,
                  ifmerge = false, 
                  if4site = true
                  )
    push!(value, v)
end

@show value