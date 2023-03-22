#!/bin/bash

# global variables
W=6
chi=128

# create sbatch jobfile
for kx in $(seq $[-$W/2] 1 $[$W/2])
do 
    for ky in $(seq $[-$W/2] 1 $[$W/2])
    do
    cp Heisenberg_bash W${W}_chi${chi}_kx${kx}_ky${ky} && sed -i "8s/--kx 0 --ky 0/--kx ${kx} --ky ${ky}/1" W${W}_chi${chi}_kx${kx}_ky${ky} && sed -i "8s/--chi 32/--chi ${chi}/1" W${W}_chi${chi}_kx${kx}_ky${ky} && sed -i "8s/--W 12/--W ${W}/1" W${W}_chi${chi}_kx${kx}_ky${ky}
    done
done

# run jobfile
for kx in $(seq $[-$W/2] 1 $[$W/2])
do 
    for ky in $(seq $[-$W/2] 1 $[$W/2])
    do
    sbatch W${W}_chi${chi}_kx${kx}_ky${ky} && rm W${W}_chi${chi}_kx${kx}_ky${ky}
    done
done