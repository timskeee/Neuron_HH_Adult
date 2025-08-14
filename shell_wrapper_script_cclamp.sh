#!/bin/bash

mut_count=0
mut_names=()
while IOF= read -r mut_name; do
	((mut_count=mut_count+1))
    echo $mut_name
	mut_names+=("$mut_name")
done < mutant_names.txt
echo $mut_count
# rm -rf $SCRATCH/IC_Fitt*
for name in "${mut_names[@]}"; 
do
	echo $name
	cp -r `pwd`  $SCRATCH/NEURON_GENERAL-2$name
	chmod -R 777 $SCRATCH/NEURON_GENERAL-2$name
	sbatch sbatch_plot.sh $name
	sleep 5
done