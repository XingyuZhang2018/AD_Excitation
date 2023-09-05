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
for J2 in 0.0:0.1:0.5
find_groundstate(J1J2(1, J2), VUMPS(maxiter=1000,show_every=1);
                 Ni = 1, Nj = 1,
                 χ = 32,
                 atype = Array,
                 infolder = "../data/",
                 outfolder = "../data/",
                 verbose = true,
                 ifADinit = false,
                 if2site = true,
                 if4site = false
                 );
end
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
