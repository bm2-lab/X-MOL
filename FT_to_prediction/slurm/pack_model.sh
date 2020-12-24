#!/bin/bash
###############################
#FileName:backupLogTimer.sh
#Function:��ʱ������־�ļ�
#Version:0.1
#Authon:xueya
#Date:2014.06.26
###############################

if [[ $# != 1 ]];then
    echo "output"
    exit -1
fi

#��ȡ��ǰ·��
path=`pwd`/$1
echo "current1 path :${path}"
#ѭ��ִ��
while true
do
   #�鿴�ļ����µ��ļ�
   fileList=`ls ${path} 2>/dev/null`
   #�������ļ����µ��ļ�
   for pFile in $fileList
   do
       current_path=${path}/${pFile}
       #�ж��Ƿ������ļ���
       if [[ -d "${current_path}" ]];then
          #��ȡ��ǰʱ��
          currentTime=`date +%Y%m%d%H%M%S`
          #����ѹ���ļ�����
          tarFileName="${current_path}_${currentTime}.tar"
          #ѹ���ļ�
          echo "pack files to $tarFileName"
          cd $path
          tar -cvf ${tarFileName} ${pFile} --remove-files
          cd -
       fi
   done
   #�ȴ�1Сʱ
   sleep 10m
done
