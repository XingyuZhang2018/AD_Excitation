using AD_Excitation
using CUDA

for W in 4:4, J2 in 0.52:0.01:0.52, χ in [512]
    SS = @time SS_correlation(J1J2(W, J2),[pi,pi],100; 
                              χ = χ,
                              infolder = "../data/",
                              atype = CuArray,
                              ifmerge=false, 
                              if4site=true
                              )
    @show(SS)       
end