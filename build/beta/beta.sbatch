#!/bin/bash
# Job name:
#SBATCH --job-name=beta
#
# Partition:
#SBATCH --partition=normal
#
# Processors:
#SBATCH --ntasks=128
#SBATCH --nodes=8
#
# Wall clock limit:
#SBATCH --time=4-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o beta.o%j
#------------------------------------------------------

echo "Starting job beta"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=beta.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job beta complete"
