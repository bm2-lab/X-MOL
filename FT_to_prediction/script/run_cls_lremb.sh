set -eux

source ./slurm/env.sh
source ./slurm/utils.sh

source ./conf_pre/ft_conf.sh

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1

# check
check_iplist

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP}"
python -u ./finetune_launch.py ${distributed_args} \
    ./run_classifier_lremb.py --use_cuda true \
                   --is_distributed true \
                   --use_fast_executor ${e_executor:-"true"} \
                   --tokenizer ${TOKENIZER:-"FullTokenizer"} \
                   --use_fp16 ${use_fp16:-"false"} \
                   --use_dynamic_loss_scaling ${use_fp16} \
                   --init_loss_scaling ${loss_scaling:-128} \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --verbose true \
                   --batch_size 16 \
                   --in_tokens false \
                   --stream_job ${STREAM_JOB:-""} \
                   --init_pretraining_params ${MODEL_PATH:-""} \
                   --init_checkpoint ${CKPT_PATH:-""} \
                   --train_set ${TASK_DATA_PATH}/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/dev.tsv \
                   --test_set ${TASK_DATA_PATH}/test.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ./checkpoints \
                   --save_steps ${SAVE_STEPS} \
                   --weight_decay 0.01 \
                   --warmup_proportion ${WARMUP_PROPORTION:-"0.0"} \
                   --validation_steps ${VALID_STEPS} \
                   --epoch ${EPOCH} \
                   --max_seq_len ${MAX_LEN} \
                   --learning_rate ${LR_RATE:-"1e-4"} \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels ${num_labels} \
                   --random_seed 1 > log/lanch.log 2>&1
