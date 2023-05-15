using AD_Excitation
using CUDA

find_groundstate(J1J2(7, 0.1), VUMPS(maxiter=1000);
                 Ni = 1, Nj = 1,
                 χ = 256,
                 atype = CuArray,
                 infolder = "./data/",
                 outfolder = "./data/",
                 verbose = true,
                 ifADinit = false,
                 if4site = true
                 );
# find_groundstate(J1J2(7, 0.3), ADMPS(ifcheckpoint=true);
#                     Ni = 1, Nj = 1,
#                     χ = 256,
#                     atype = CuArray,
#                     infolder = "./data/",
#                     outfolder = "./data/",
#                     verbose = true,
#                     ifMPO = true,
#                     if_vumps_init = true,
#                     if4site = true
#                     );