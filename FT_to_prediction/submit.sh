#!bin/bash

if [[ $# != 1 ]]; then
    echo "input: job_conf"
    exit -1
fi

job_conf=$1
source $job_conf

task_name_len=${#task_name}
if [[ $task_name_len -gt 40 ]];then
    printf '[Warning] task_name "%s" cannot be longer than 40 (now %d). \n' ${task_name} ${task_name_len}
    printf '[Warning] The paddlecloud platform limits the maximum length of the name of submitted task.\n'
    task_name=${task_name:0:40}
fi

if [ $HGCP_CLIENR_BIN ]; then
    HGCP_CLIENR_BIN=$HGCP_CLIENR_BIN
else
    HGCP_CLIENR_BIN=~/.hgcp/software-install/HGCP_client/bin
fi

#package data.tar.gz
upload_file=data.tar.gz
if [[ -f ${upload_file} ]];then
    rm ${upload_file}
fi
tar -zcf ${upload_file} ./ --exclude "./logs" --exclude "./submit.sh" --exclude "./${upload_file}"
#upload
HADOOP_BIN=${HGCP_CLIENR_BIN}/../tools/hadoop-v2/hadoop/bin/hadoop
HADOOP_RUN="${HADOOP_BIN} fs -D hadoop.job.ugi=${ugi} -D fs.default.name=${hdfs_path}"
${HADOOP_RUN} -mkdir ${hdfs_output}/tmp/

${HADOOP_RUN} -test -e ${hdfs_output}/tmp/${upload_file}
if [[ $? == 0 ]];then
    ${HADOOP_RUN} -rm ${hdfs_output}/tmp/${upload_file}
fi

${HADOOP_RUN} -put ${upload_file} ${hdfs_output}/tmp/

#group aksk
sc_group_ak="6f0f5ee59a4e5fbd9738bfc2d96d78cc"
sc_group_sk="7bb2fc78604c518b8218022903a46a66"

group_id=`cat ./slurm/.sc_group.kv | awk -v queue=${queue:-"yq01-p40-4-8"} \
    'BEGIN{FS=","; ret = "";}{if ($1 == queue) {ret = $2}}END{if (ret != "") {print ret}}'`
if [[ ${group_id} == "" ]];then
    printf "[ERROR] subumit to the queue %s is not supported.\nplease try to fix it.\n" ${queue}
    exit -1
fi

task_name=(${task_name//,/-})
paddlecloud job --server paddlecloud.baidu-int.com \
        --port 80 \
        --user-ak ${sc_group_ak} \
        --user-sk ${sc_group_sk} \
        train --job-name ${task_name} \
        --job-version slurm-custom \
        --use-system-aksk \
        --user-name ${submitter} \
        --group-id ${group_id} \
        --fs-name ${hdfs_path} \
        --fs-ugi ${ugi} \
        --output-path ${hdfs_output} \
        --job-tgz-pkg-name ${upload_file} \
        --job-script ./job.sh \
        --job-framework paddle \
        --slurm-nodes ${nodes:-1} \
        --slurm-gpu-pnode 8 \
        --slurm-task-pnode 1 \
        --wall-time ${walltime:-"00:00:00"}
