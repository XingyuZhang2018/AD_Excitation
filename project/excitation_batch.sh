#!/bin/bash

# global variables
W=6
chi=128

# create sbatch jobfile
for kx in $(seq $[-$W/2] 1 $[$W/2])
do 
    for ky in $(seq $[-$W/2] 1 $[$W/2])
    do
        sed \
        -e "s/\"model\": .*/\"model\": \"Kitaev(${W})\",/g" \
        -e "s/\"χ\": .*/\"χ\": ${chi},/g" \
        -e "s/\"targχ\": .*/\"targχ\": ${chi},/g" \
        -e "s/\"k\": .*/\"k\": [${kx},${ky}]/g" \
        config.json > W${W}_chi${chi}_kx${kx}_ky${ky}.json && \
        sed \
        -e "s/config.json/W${W}_chi${chi}_kx${kx}_ky${ky}.json/g" \
        excitation.sh > W${W}_chi${chi}_kx${kx}_ky${ky}.sh
    done
done

# run jobfile
for kx in $(seq $[-$W/2] 1 $[$W/2])
do 
    for ky in $(seq $[-$W/2] 1 $[$W/2])
    do
        chi=$[2**$[chi_i]] && sbatch W${W}_chi${chi}_kx${kx}_ky${ky}.sh && rm W${W}_chi${chi}_kx${kx}_ky${ky}.sh
    done
done