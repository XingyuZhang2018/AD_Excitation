#!/bin/bash

# create sbatch jobfile
for W in $(seq 10 1 10)
do 
    for chi_i in $(seq 9 1 9)
    do
        for J2 in $(seq 0.4 0.01 0.7)
        do
            for kx in $(seq $[W/2] 1 $[W/2])
            do 
                for ky in $(seq $[0/2] 1 $[0/2])
                do
                    chi=$((2**chi_i))
                    sed \
                    -e "s|--model .*|--model \"J1J2($W,$J2)\" \\\|g" \
                    -e "s|--chi .*|--chi $chi \\\|g" \
                    -e "s|--if2site .*|--if2site true \\\|g" \
                    -e "s|--if4site .*|--if4site false \\\|g" \
                    -e "s|--kx .*|--kx $kx \\\|g" \
                    -e "s|--ky .*|--ky $ky \\\|g" \
                    -e "s|--n .*|--n 30|g" \
                    excitation_paracloud.sh > ex_W${W}_J2-${J2}_chi${chi}_kx${kx}_ky${ky}.sh
                    sbatch --gpus=1 ex_W${W}_J2-${J2}_chi${chi}_kx${kx}_ky${ky}.sh 
                    rm ex_W${W}_J2-${J2}_chi${chi}_kx${kx}_ky${ky}.sh
                done
            done
        done
    done
done