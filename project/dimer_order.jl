using AD_Excitation
using CUDA

for W in 5:5, J2 in 0.5:0.01:0.5, χ in [1024]
    @time dimer_order(J1J2(W, J2); 
                              χ = χ,
                              infolder = "./data/",
                              atype = CuArray,
                              ifmerge=false, 
                              if4site=true
                              ) 
end