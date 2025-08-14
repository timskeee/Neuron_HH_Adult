#!/bin/bash

# rm -rf $SCRATCH/IC_Fitt*

cp -r `pwd`  $SCRATCH/NEURON_GENERAL-2
chmod -R 777 $SCRATCH/NEURON_GENERAL-2
sbatch sbatch_plot_noarg.sh
	# sleep 10
