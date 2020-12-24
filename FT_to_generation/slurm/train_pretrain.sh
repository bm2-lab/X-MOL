set -eu

#bash -x ./env.sh
DD_RAND_SEED=$1

source ./slurm/env.sh
source ./slurm/utils.sh

source ./model_conf

export PATH="$PWD/python/bin/:$PATH"
export PYTHONPATH="$PWD/python/"

export FLAGS_eager_delete_tensor_gb=2.0
#export FLAGS_fraction_of_gpu_memory_to_use=1.0
export FLAGS_sync_nccl_allreduce=1

e_executor=$(echo ${use_experimental_executor-'True'} | tr '[A-Z]' '[a-z]')

use_fuse=$(echo ${use_fuse-'False'} | tr '[A-Z]' '[a-z]')
if [[ ${use_fuse} == "true" ]]; then
    #MB
    export FLAGS_fuse_parameter_memory_size=64
fi

#pack output
nohup sh ./slurm/pack_model.sh ./output > log/pack_model.log 2>&1 &

# check
check_iplist

echo "Rand seed"${DD_RAND_SEED}

distributed_args="--node_ips ${PADDLE_TRAINERS} --node_id ${PADDLE_TRAINER_ID} --current_node_ip ${POD_IP}"
python -u ./lanch.py ${distributed_args} \
    ./train.py --use_cuda "True" \
                --is_distributed ${is_distributed-"True"} \
                --weight_sharing "True" \
                --use_fast_executor ${e_executor-"True"} \
                --use_fuse ${use_fuse-"False"} \
                --nccl_comm_num ${nccl_comm_num:-"1"} \
                --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"False"} \
                --is_unidirectional ${is_unidirectional-"False"} \
                --in_tokens ${in_tokens-"True"} \
                --batch_size ${BATCH_SIZE} \
                --vocab_path ${vocab_path} \
                --train_filelist ${train_filelist} \
                --valid_filelist ${valid_filelist} \
                --epoch ${EPOCH:-100} \
                --validation_steps ${validation_steps:-1000} \
                --random_seed ${DD_RAND_SEED} \
                --hack_old_data ${hack_old_data-"False"} \
                --generate_neg_sample ${generate_neg_sample-"True"} \
                --lr_scheduler ${lr_scheduler} \
                --num_train_steps ${num_train_steps} \
                --checkpoints ./output \
                --use_fp16 ${use_fp16:-"False"} \
                --use_dynamic_loss_scaling ${use_fp16} \
                --init_loss_scaling ${loss_scaling:-128} \
                --save_steps ${SAVE_STEPS} \
                --init_checkpoint ${init_model:-""} \
                --ernie_config_path ${CONFIG_PATH} \
                --learning_rate ${LR_RATE} \
                --warmup_steps ${WARMUP_STEPS:-0} \
                --weight_decay ${WEIGHT_DECAY:-0} \
                --max_seq_len ${MAX_LEN} \
                --skip_steps 10 >> log/lanch.log 2>&1
