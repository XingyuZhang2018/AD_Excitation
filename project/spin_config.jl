using AD_Excitation
using CUDA

# for W in 5:5, J2 in 0.4:0.01:0.7, χ in [256]
    spin_config(TFIsing(0.5, 12, 3.04438); 
                χ = 1024,
                infolder = "./data/",
                atype = CuArray,
                ifmerge=false, 
                if4site=false
                )
# end