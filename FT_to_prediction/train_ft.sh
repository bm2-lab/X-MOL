set -eu

#bash -x ./env.sh
source ./slurm/env.sh
source ./slurm/utils.sh

#source ./model_conf
source ./conf_pre/ft_conf.sh

# check
check_iplist

mkdir -p config
cp ${vocab_path} ./config/vocab.txt
tokenizer=${tokenizer-'FullTokenizer'}
if [[ ${tokenizer} == "SentencepieceTokenizer" || \
    ${tokenizer} == "WordsegTokenizer" ]]; then
    cp ${vocab_path}.model ./config/vocab.txt.model
fi

export TOKENIZER=$tokenizer

cp ${CONFIG_PATH} ./config/ernie_config.json

tasks=(${finetuning_task//,/ })
tasks_lr=(${LR_RATE//,/ })
#tasks_batch=(${BATCH_SIZE//,/ })
the_task_num=0
for the_task in ${tasks[@]}
do
    task_lr=${tasks_lr[the_task_num]}
    #task_batch=${tasks_batch[the_task_num]}
    #sh ./script/run_${the_task}.sh $the_task_num $task_lr $task_batch &
    sh ./script/run_${the_task}.sh $the_task_num $task_lr &
    sleep 10
    ((the_task_num+=1))
done
