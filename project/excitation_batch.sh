#!/bin/bash

# create sbatch jobfile
for W in $(seq 7 1 7)
do 
    for chi_i in $(seq 6 1 7)
    do
        for J2 in $(seq 0.1 0.1 0.5)
        do
            for kx in $(seq $[-0] 1 $[0])
            do 
                for ky in $(seq $[-0] 1 $[0])
                do
                    chi=$((2**chi_i))
                    sed \
                    -e "s|--partition=a800|--partition=v100|g" \
                    -e "s|--model .*|--model \"J1J2($W,$J2)\" \\\|g" \
                    -e "s|--chi .*|--chi $chi \\\|g" \
                    -e "s|--kx .*|--kx $kx \\\|g" \
                    -e "s|--ky .*|--ky $ky \\\|g" \
                    excitation.sh > W${W}_J2-${J2}_chi${chi}_kx${kx}_ky${ky}.sh
                done
            done
        done
    done
done

# run jobfile
for W in $(seq 7 1 7)
do 
    for chi_i in $(seq 6 1 7)
    do
        for J2 in $(seq 0.1 0.1 0.5)
        do
            for kx in $(seq $[-0] 1 $[0])
            do 
                for ky in $(seq $[-0] 1 $[0])
                do
                    chi=$[2**$[chi_i]] && sbatch W${W}_J2-${J2}_chi${chi}_kx${kx}_ky${ky}.sh && rm W${W}_J2-${J2}_chi${chi}_kx${kx}_ky${ky}.sh
                done
            done
        done
    done
done