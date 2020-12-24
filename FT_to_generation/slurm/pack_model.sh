#!/bin/bash
###############################
#FileName:backupLogTimer.sh
#Function:定时备份日志文件
#Version:0.1
#Authon:xueya
#Date:2014.06.26
###############################

if [[ $# != 1 ]];then
    echo "output"
    exit -1
fi

#获取当前路径
path=`pwd`/$1
echo "current1 path :${path}"
#循环执行
while true
do
   #查看文件夹下的文件
   fileList=`ls ${path} 2>/dev/null`
   #遍历此文件夹下的文件
   for pFile in $fileList
   do
       current_path=${path}/${pFile}
       #判断是否属于文件夹
       if [[ -d "${current_path}" ]];then
          #获取当前时间
          currentTime=`date +%Y%m%d%H%M%S`
          #定义压缩文件名称
          tarFileName="${current_path}_${currentTime}.tar"
          #压缩文件
          echo "pack files to $tarFileName"
          cd $path
          tar -cvf ${tarFileName} ${pFile} --remove-files
          cd -
       fi
   done
   #等待1小时
   sleep 10m
done
