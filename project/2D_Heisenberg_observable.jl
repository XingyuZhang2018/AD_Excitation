using AD_Excitation

 W = 4
 D = 16
 χ = 16
Nj = 1
x,y = 2,0
k = (x*2*pi/W,y*2*pi/W)
model = Heisenberg(0.5,W,1.0,1.0,1.0)
 infolder = "../data/"
outfolder = "../data/"
atype = Array
X1 = load_canonical_excitaion(infolder, model, Nj, D, χ, k)[1]
pattern_weight(model, k, 10; 
                Nj=Nj, χ=χ, atype=CuArray, 
                infolder=infolder, outfolder=outfolder, 
                ifmerge=false, if4site=true)