set -eux
#set +ex

source ./slurm/env.sh
source ./slurm/utils.sh

source ./model_conf

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1

# check
check_iplist

mkdir -p ./checkpoints
mkdir -p ./tmpdir
export TMPDIR=`pwd`/tmpdir

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP} \
                --init_new_sent_embedding ${init_new_sent_embedding:-"true"}"
python -u ./lanch.py ${distributed_args} \
    ./run_seq2seq.py --use_cuda true \
                   --do_train $do_train \
                   --do_val $do_val \
                   --do_test $do_test \
                   --do_pred ${do_pred:-"false"} \
                   --train_set ${TASK_DATA_PATH}/${train_set} \
                   --dev_set ${TASK_DATA_PATH}/${dev_set} \
                   --test_set ${TASK_DATA_PATH}/${test_set} \
                   --pred_set ${TASK_DATA_PATH}/${pred_set:-""} \
                   --epoch $epoch \
                   --tokenized_input $tokenized_input \
                   --task_type "trans" \
                   --role_type_size ${role_type_size:-0} \
                   --turn_type_size ${turn_type_size:-0} \
                   --max_src_len $max_src_len \
                   --max_tgt_len $max_tgt_len \
                   --max_dec_len $max_dec_len \
                   --two_stream ${two_stream:-"False"} \
                   --random_noise ${random_noise:-"False"} \
                   --mask_prob $mask_prob \
                   --continuous_position ${continuous_position:-"false"} \
                   --tgt_type_id $tgt_type_id \
                   --pos_emb_size ${pos_emb_size:-512} \
                   --batch_size $batch_size \
                   --learning_rate $learning_rate \
                   --lr_scheduler $lr_scheduler \
                   --warmup_proportion $warmup_proportion \
                   --weight_decay $weight_decay \
                   --weight_sharing false \
                   --label_smooth $label_smooth \
                   --do_dec $do_dec \
                   --beam_size $beam_size  \
                   --length_penalty ${length_penalty:-"0"} \
                   --eos_idx $eos_idx \
                   --mask_idx $mask_idx \
                   --init_pretraining_params ${MODEL_PATH:-""} \
                   --src_vocab_path ${src_vocab_path:-""} \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ./checkpoints \
                   --save_and_valid_by_epoch $save_and_valid_by_epoch \
                   --eval_script ${eval_script:-""} \
                   --random_seed ${random_seed:-"-1"} > log/lanch.log 2>&1
