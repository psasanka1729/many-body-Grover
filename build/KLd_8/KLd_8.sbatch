#!/bin/bash
# Job name:
#SBATCH --job-name=KLd_8
#
# Partition:
#SBATCH --partition=CMT-Kepler
#
# Processors:
#SBATCH --ntasks=16
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=7-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o KLd_8.o%j
#------------------------------------------------------

echo "Starting job KLd_8"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=KLd_8.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job KLd_8 complete"
