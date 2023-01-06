#!/bin/bash
# Job name:
#SBATCH --job-name=GUE_tenpy
#
# Partition:
#SBATCH --partition=normal
#
# Processors:
#SBATCH --ntasks=256
#SBATCH --nodes=16
#
# Wall clock limit:
#SBATCH --time=4-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o GUE_tenpy.o%j
#------------------------------------------------------

echo "Starting job GUE_tenpy"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=GUE_tenpy.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job GUE_tenpy complete"
