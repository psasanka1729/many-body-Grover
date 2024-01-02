#!/bin/bash
# Job name:
#SBATCH --job-name=phi_delta
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
#SBATCH -o phi_delta.o%j
#------------------------------------------------------

echo "Starting job phi_delta"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=phi_delta.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job phi_delta complete"
