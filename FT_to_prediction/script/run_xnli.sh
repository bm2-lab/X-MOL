set -eux

source ./slurm/env.sh
source ./slurm/utils.sh

source ./model_conf

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1

# check
check_iplist

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP}"
python -u ./lanch.py ${distributed_args} \
    ./run_classifier.py --use_cuda true \
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
                   --batch_size 8192 \
                   --in_tokens true \
                   --stream_job ${STREAM_JOB:-""} \
                   --init_pretraining_params ${MODEL_PATH:-""} \
                   --train_set ${TASK_DATA_PATH}/xnli/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/xnli/dev.tsv \
                   --test_set ${TASK_DATA_PATH}/xnli/test.tsv \
                   --vocab_path config/vocab.txt \
                   --label_map ${TASK_DATA_PATH}/xnli/label_map.json \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ./checkpoints \
                   --save_steps 1000 \
                   --weight_decay 0.01 \
                   --warmup_proportion ${WARMUP_PROPORTION:-"0.0"} \
                   --validation_steps 25 \
                   --epoch 3 \
                   --max_seq_len 128 \
                   --learning_rate ${LR_RATE:-"1e-4"} \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 3 \
                   --random_seed 1 > log/lanch.log 2>&1
