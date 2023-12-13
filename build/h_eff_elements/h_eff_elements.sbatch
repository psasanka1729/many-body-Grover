#!/bin/bash
# Job name:
#SBATCH --job-name=h_eff_elements
#
# Partition:
#SBATCH --partition=CMT
#
# Processors:
#SBATCH --ntasks=64
#SBATCH --nodes=4
#
# Wall clock limit:
#SBATCH --time=3-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o h_eff_elements.o%j
#------------------------------------------------------

echo "Starting job h_eff_elements"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=h_eff_elements.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job h_eff_elements complete"
