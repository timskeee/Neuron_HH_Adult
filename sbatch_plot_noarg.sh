#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J Plotting
#SBATCH --mail-user=bens.roy@gmail.com
#SBATCH --mail-type=NONE
#SBATCH -t 12:00:00
#SBATCH --image=balewski/ubu20-neuron8:v5

#OpenMP settings:
export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#name=$1 #puts name variable as first arg  ##REMOVED

srun  -n 1 -c 64 --output=$SCRATCH/NEURON_GENERAL-2/output_log_files/%A.out --error=$SCRATCH/NEURON_GENERAL-2/output_log_files/%A.err shifter python3 $SCRATCH/NEURON_GENERAL-2/run12HH16HH_compiledSynthMuts_4on1plots.py #$name ##REMOVED