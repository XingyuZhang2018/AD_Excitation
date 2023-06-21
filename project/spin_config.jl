using AD_Excitation
using CUDA

for W in 5:5, J2 in 0.4:0.01:0.7, χ in [256]
    spin_config(J1J2(W, J2); 
                χ = χ,
                infolder = "./data/",
                atype = CuArray,
                ifmerge=false, 
                if4site=true
                )
end