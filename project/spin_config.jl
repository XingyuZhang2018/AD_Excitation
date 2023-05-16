using AD_Excitation
using CUDA

spin_config(J1J2(4, 0.0); 
            χ = 16,
            infolder = "../data/",
            atype = Array,
            ifmerge=false, 
            if4site=true
            )