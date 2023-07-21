using AD_Excitation
using CUDA

for W in 5:5, J2 in 0.0:0.1:0.0, χ in [512]
    @time dimer_order(J1J2(W, J2); 
                              χ = χ,
                              infolder = "../data/",
                              atype = CuArray,
                              ifmerge=false, 
                              if4site=true
                              ) 
end