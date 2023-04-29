#!/bin/bash

# create sbatch jobfile
for W in $(seq 4 1 6)
do 
    for chi_i in $(seq 9 1 9)
    do
        for J2 in $(seq 0.0 0.1 0.5)
        do
            chi=$[2**$[chi_i]] && cp J1J2_excitation.sh J1J2_excitation_J2-${J2}_W${W}_chi${chi} && sed -i "8s/--J2 0.0 --W 4 --chi 64/--J2 ${J2} --W ${W} --chi ${chi}/1" J1J2_excitation_J2-${J2}_W${W}_chi${chi}
        done
    done
done

# run jobfile
for W in $(seq 4 1 6)
do 
    for chi_i in $(seq 9 1 9)
    do
        for J2 in $(seq 0.0 0.1 0.5)
        do
            chi=$[2**$[chi_i]] && sbatch J1J2_excitation_J2-${J2}_W${W}_chi${chi} && rm J1J2_excitation_J2-${J2}_W${W}_chi${chi}
        done
    done
done