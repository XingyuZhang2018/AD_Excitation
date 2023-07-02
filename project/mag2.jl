using AD_Excitation
using CUDA

for W in 6:6, J2 in 0.5:0.01:0.5, χ in [512]
    mag2_tol = @time mag2(J1J2(W, J2),[pi,pi]; 
                          χ = χ,
                          infolder = "../data/",
                          atype = CuArray,
                          ifmerge=false, 
                          if4site=true
                          )
    @show(sum(mag2_tol)/4)
end