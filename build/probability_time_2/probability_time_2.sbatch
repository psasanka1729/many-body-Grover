#!/bin/bash
# Job name:
#SBATCH --job-name=probability_time_2
#
# Partition:
#SBATCH --partition=CMT
#
# Processors:
#SBATCH --ntasks=64
#SBATCH --nodes=4
#
# Wall clock limit:
#SBATCH --time=7-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o probability_time_2.o%j
#------------------------------------------------------

echo "Starting job probability_time_2"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=probability_time_2.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job probability_time_2 complete"
