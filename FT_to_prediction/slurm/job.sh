#!/bin/bash
set -x

mpirun hostname
source ./model_conf
if [[ ${slurm_train_files_dir:-""} != "" ]];then
    sh /home/HGCP_Program/software-install/afs_mount/bin/afs_mount.sh NLP_KM_Data NLP_km_2018 `pwd`/data ${slurm_train_files_dir}
    mkdir -p log
    ls `pwd`/data > ./log/afs_mount.log
    rm -rf ./data/logs
fi

mpirun sh ./slurm/setup.sh

iplist=`cat nodelist-${SLURM_JOB_ID} | xargs  | sed 's/ /,/g'`

RAND_SEED=$RANDOM
mpirun --bind-to none -x iplist=${iplist} sh train.sh ${RAND_SEED}
