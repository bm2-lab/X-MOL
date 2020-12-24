#!bin/bash
set -eu

source ./slurm/env.sh
source ./slurm/utils.sh

source ./model_conf

#init
core_num=100
data_dir=data
tmp_dir=tmp
file_list=filelist

HADOOP="hadoop fs -D hadoop.job.ugi=${ugi} -D fs.default.name=${hdfs_path}"

if [[ ${slurm_train_files_dir:-""} == "" ]];then
    mkdir -p $data_dir
    mkdir -p $tmp_dir
    #download train data
    $HADOOP -get ${train_files_dir} ./$data_dir/
    train_tar=`basename ${train_files_dir}`
    cd ./$data_dir; tar -xvf ${train_tar}; rm -rf ${train_tar}; cd -

    #$HADOOP -ls ${train_files_dir} | awk '{print $NF}' > ./${file_list}
    #lint=`wc -l ${file_list} | awk '{print $1}'`
    #pieces=`awk -v line=$line -v core_num=$core_num 'BEGIN{
    #            first=int(line/(core_num)); 
    #            remind=line%first; 
    #            print (first + 2 * int((remind+core_num)/core_num))
    #        }'`
    #split -d -a 2 -l $pieces $file_list $tmp_dir/part.
    #
    #download_one_file() {
    #    local file=$1
    #    for one_file in `cat $file`
    #    do
    #        $HADOOP -get $one_file $data_dir
    #    done
    #}
    #
    #i=0
    #while [ $i -lt $core_num ]
    #do
    #    part_=`echo $i | awk '{printf("part.""%02d", $0)}'`
    #    download_one_file $tmp_dir/${part_}&
    #    i=`expr $i + 1`
    #done
fi

#download init model
if [[ ${hdfs_init_model:-""} != "" && ${init_model:-""} != "" ]];then
    model_tar=`basename ${hdfs_init_model}`
    $HADOOP -get ${hdfs_init_model} ./
    tar xvf ${model_tar}
fi

#python & paddle & nccl2
python_package=./package/python.tgz
nccl2_package=./package/nccl_2.3.5.tgz
rm -rf python python.tgz
tar -zxvf ./$python_package
tar -zxvf ./$nccl2_package
#setup output/log
mkdir -p output
mkdir -p log
