#!/bin/bash
# Job name:
#SBATCH --job-name=KLd_10_run_1
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
#SBATCH -o KLd_10_run_1.o%j
#------------------------------------------------------

echo "Starting job KLd_10_run_1"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=KLd_10_run_1.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job KLd_10_run_1 complete"
