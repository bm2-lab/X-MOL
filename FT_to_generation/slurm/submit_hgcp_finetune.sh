#!bin/bash
if [[ $# != 1 ]]; then
    echo "input: job_conf"
    exit -1
fi

mkdir -p ./package
cd ./package
set -x
python_package=python.tgz
if [[ ! -f ${python_package} ]];then
    wget -r -nH --cut-dirs=4 ftp://cp01-nlp-build2.cp01.baidu.com:/home/disk8/liyukun01/paddle_fluid/python.tgz ./
fi

nccl2_package=nccl_2.3.5.tgz
if [[ ! -f ${nccl2_package} ]];then
    wget -r -nH --cut-dirs=4 ftp://cp01-nlp-build2.cp01.baidu.com:/home/disk8/liyukun01/paddle_fluid/nccl_2.3.5.tgz ./
fi

set +x
cd -

set -x
job_conf=$1
conf=`basename $job_conf`
job_package_path=tmp/job_`date +%s_%N`_${conf}

mkdir -p ${job_package_path}

#package job path
cp $job_conf ./$job_package_path/model_conf
cp -r *.py ./$job_package_path/
cp -r ./finetune/ ./model/ ./reader/ ./utils/ ./package/ ./script/ ./slurm ./$job_package_path/

set +x
cd $job_package_path
ln -s model_conf ./$conf
ln -s ./slurm/train_finetuning.sh ./train.sh
ln -s ./slurm/job.sh ./job.sh
ln -s ./finetune_launch.py ./lanch.py

platform="paddlecloud"

if [[ ! -f $platform || $platform == "paddlecloud" ]];then
    ln -s ./slurm/submit_paddlecloud_finetuning.sh ./submit.sh
else
    ln -s ./slurm/submit_finetuning.sh ./submit.sh
fi

#ln -s ./slurm/submit_finetuning.sh ./submit.sh

set -x
sh submit.sh ./model_conf
set +x
