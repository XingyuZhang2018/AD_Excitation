using AD_Excitation
using CUDA

for W in 6:6, J2 in 0.0:0.01:0.0, χ in [128]
    mag2_tol = @time mag2(J1J2(W, J2),(pi,pi),W^2; 
                          χ = χ,
                          infolder = "../data/",
                          atype = CuArray,
                          ifmerge=false, 
                          if4site=true
                          )
end