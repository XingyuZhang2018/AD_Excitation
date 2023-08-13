using AD_Excitation
using CUDA

for W in 6:6, J2 in 0.4:0.01:0.7, χ in [512]
    @time dimer_order(J1J2(W, J2); 
                              χ = χ,
                              infolder = "../data/",
                              atype = CuArray,
                              ifmerge=false, 
                              if4site=true
                              ) 
end