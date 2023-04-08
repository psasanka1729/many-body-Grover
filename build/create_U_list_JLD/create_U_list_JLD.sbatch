#!/bin/bash
# Job name:
#SBATCH --job-name=create_U_list_JLD
#
# Partition:
#SBATCH --partition=128s
#
# Processors:
#SBATCH --ntasks=16
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=4-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o create_U_list_JLD.o%j
#------------------------------------------------------

echo "Starting job create_U_list_JLD"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=create_U_list_JLD.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job create_U_list_JLD complete"
