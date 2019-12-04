#!/bin/bash
wd=$(pwd)

reqd_cmd="#!/bin/bash
#PBS -l nodes=1:ppn=9:xk
#PBS -l walltime=00:05:00
#PBS -N image_ranking_
#PBS -e \$PBS_JOBID.err
#PBS -o \$PBS_JOBID.out
# -m and -M set up mail messages at begin,end,abort:
# -m bea
# -M nms9@illinois.edu

#. /opt/modules/default/init/bash
module load python/2.0.0
#module load cudatoolkit
cd ${wd}
aprun -n 1 -N 1 python main_1.py"

while echo $1 | grep -q ^-; do
    reqd_cmd="${reqd_cmd} -$1 $2"
    shift
    shift
done

echo "${reqd_cmd}" > run_task.pbs

qsub run_task.pbs
rm run_task.pbs

