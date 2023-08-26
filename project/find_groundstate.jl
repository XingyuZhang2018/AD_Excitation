using AD_Excitation
using CUDA

# find_groundstate(J1J2(6,0.42), IDMRG1(maxiter=1000);
#                  Ni = 1, Nj = 1,
#                  χ = 64,
#                  atype = CuArray,
#                  infolder = "./data/",
#                  outfolder = "./data/",
#                  verbose = true,
#                  ifADinit = false,
#                  if4site = true
#                  );
find_groundstate(J1xJ1yJ2(0.5,4,1.0,1.0,0.5), VUMPS(maxiter=1000,show_every=1);
                 Ni = 1, Nj = 1,
                 χ = 16,
                 atype = Array,
                 infolder = "../data/",
                 outfolder = "../data/",
                 verbose = true,
                 ifADinit = false,
                 if2site = false,
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
