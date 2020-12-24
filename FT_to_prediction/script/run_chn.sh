set -eux

source ./slurm/env.sh
source ./slurm/utils.sh

source ./model_conf

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1

# check
check_iplist

the_task_num=$1
if [[ $# -ge 2 ]];then
    LR_RATE=$2
fi
if [[ $# -ge 3 ]];then
    BATCH_SIZE=$3
fi

log_prefix=$the_task_num"_chn_"

distributed_args="--log_prefix ${log_prefix} \
                --node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP} \
                --nproc_per_node 1 \
                --selected_gpus ${the_task_num:-0}"
python -u ./lanch.py ${distributed_args} \
    ./run_classifier.py --use_cuda true \
                   --is_distributed true \
                   --use_fast_executor ${e_executor:-"true"} \
                   --tokenizer ${TOKENIZER:-"FullTokenizer"} \
                   --use_fp16 ${use_fp16:-"false"} \
                   --use_dynamic_loss_scaling ${use_fp16} \
                   --init_loss_scaling ${loss_scaling:-128} \
                   --verbose true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 24 \
                   --stream_job ${STREAM_JOB:-""} \
                   --init_pretraining_params ${MODEL_PATH:-""} \
                   --train_set ${TASK_DATA_PATH}/chnsenticorp/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/chnsenticorp/dev.tsv \
                   --test_set ${TASK_DATA_PATH}/chnsenticorp/test.tsv \
                   --vocab_path config/vocab.txt \
                   --checkpoints ./checkpoints \
                   --save_steps 1000 \
                   --weight_decay 0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 100 \
                   --epoch 10 \
                   --max_seq_len 256 \
                   --ernie_config_path config/ernie_config.json \
                   --learning_rate ${LR_RATE-"5e-5"} \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 10 \
                   --num_labels 2 \
                   --random_seed 1 > log/${log_prefix}lanch.log 2>&1
