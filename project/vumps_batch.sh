#!/bin/bash

# create sbatch jobfile
for W in $(seq 7 1 7)
do 
    for chi_i in $(seq 6 1 7)
    do
        for J2 in $(seq 0.0 0.1 0.5)
        do
            chi=$((2**chi_i))
            sed \
            -e "s|--partition=a800|--partition=v100|g" \
            -e "s|--model .*|--model \"J1J2($W,$J2)\" \\\|g" \
            -e "s|--chi .*|--chi $chi \\\|g" \
            vumps.sh > J2-${J2}_W${W}_chi${chi}.sh
        done
    done
done

# run jobfile
for W in $(seq 7 1 7)
do 
    for chi_i in $(seq 6 1 7)
    do
        for J2 in $(seq 0.0 0.1 0.5)
        do
            chi=$[2**$[chi_i]] && sbatch J2-${J2}_W${W}_chi${chi}.sh && rm J2-${J2}_W${W}_chi${chi}.sh
        done
    done
done