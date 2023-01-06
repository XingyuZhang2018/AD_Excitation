#!/bin/bash

# global variables
D=2
chi=64

# create sbatch jobfile
for kx in $(seq -6 1 6)
do 
    for ky in $(seq -6 1 6)
    do
    cp single_job D${D}_chi${chi}_kx${kx}_ky${ky} && sed -i "7s/--kx 0 --ky 0/--kx ${kx} --ky ${ky}/1" D${D}_chi${chi}_kx${kx}_ky${ky} && sed -i "7s/--chi 32/--chi ${chi}/1" D${D}_chi${chi}_kx${kx}_ky${ky}
    done
done

# run jobfile
for kx in $(seq -6 1 6)
do 
    for ky in $(seq -6 1 6)
    do
    sbatch D${D}_chi${chi}_kx${kx}_ky${ky} && rm D${D}_chi${chi}_kx${kx}_ky${ky}
    done
done