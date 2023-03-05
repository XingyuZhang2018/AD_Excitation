#!/bin/bash

# global variables
W=12
chi=1024

# create sbatch jobfile
for kx in $(seq 0 1 $[$W/2])
do 
    for ky in $(seq ${kx} 1 $[$W/2])
    do
    cp Heisenberg_bash W${W}_chi${chi}_kx${kx}_ky${ky} && sed -i "8s/--kx 0 --ky 0/--kx ${kx} --ky ${ky}/1" W${W}_chi${chi}_kx${kx}_ky${ky} && sed -i "8s/--chi 32/--chi ${chi}/1" W${W}_chi${chi}_kx${kx}_ky${ky} && sed -i "8s/--W 12/--W ${W}/1" W${W}_chi${chi}_kx${kx}_ky${ky}
    done
done

# run jobfile
for kx in $(seq 0 1 $[$W/2])
do 
    for ky in $(seq ${kx} 1 $[$W/2])
    do
    sbatch W${W}_chi${chi}_kx${kx}_ky${ky} && rm W${W}_chi${chi}_kx${kx}_ky${ky}
    done
done