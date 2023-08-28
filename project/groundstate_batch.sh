#!/bin/bash

# create sbatch jobfile
for W in $(seq 4 1 4)
do 
    for chi_i in $(seq 7 0.2 7)
    do
        for J2 in $(seq 0.3 0.01 0.5)
        do
            for J1x in $(seq 0.8 0.1 0.8)
            do
                for J1y in $(seq 1.0 0.1 1.0)
                do
                    chi=$(awk -v chi_i="$chi_i" 'BEGIN { printf "%.0f", 2^chi_i }')
                    sed \
                    -e "s|--partition=a800|--partition=a800|g" \
                    -e "s|--alg .*|--alg \"VUMPS\" \\\|g" \
                    -e "s|--model .*|--model \"J1xJ1yJ2(0.5,$W,$J1x,$J1y,$J2)\" \\\|g" \
                    -e "s|--chi .*|--chi $chi \\\|g" \
                    -e "s|--targchi .*|--targchi $chi \\\|g" \
                    -e "s|--if2site .*|--if2site false \\\|g" \
                    -e "s|--if4site .*|--if4site true|g" \
                    groundstate.sh > gs_J1x${J1x}_J1y${J1y}_J2-${J2}_W${W}_chi${chi}.sh
                    sbatch gs_J1x${J1x}_J1y${J1y}_J2-${J2}_W${W}_chi${chi}.sh 
                    rm gs_J1x${J1x}_J1y${J1y}_J2-${J2}_W${W}_chi${chi}.sh
                done
            done
        done
    done
done