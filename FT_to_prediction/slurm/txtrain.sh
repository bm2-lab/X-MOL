set -eu

#bash -x ./env.sh
source ./shell/env.sh
source ./shell/utils.sh

conf=$1
source ./$1

#export PATH="/home/bert/tianxin04/share/python/bin:$PATH"
export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/home/bert/tianxin04/share//nccl_2.3.5/lib/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/bert/:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PATH="/home/bert/tianxin04/bert/tmp/python_with_TruncatedNormal/bin/:${PATH}"

export FLAGS_fraction_of_gpu_memory_to_use=1.0

#pack output
nohup sh ./shell/pack_model.sh ./output > log/pack_model.log 2>&1 &

# check
check_iplist

if [[ ${nodes:-1} -gt 1 ]];then 
    python -u ./train.py --use_cuda \
                    --is_distributed \
                    --use_fast_executor \
                    --weight_sharing \
                    --batch_size ${BATCH_SIZE} \
                    --generate_neg_sample \
                    --data_dir ${traindata_dir} \
                    --validation_set_dir ${testdata_dir} \
                    --checkpoints ./output \
                    --save_steps ${SAVE_STEPS} \
                    --init_model ${init_model:-""} \
                    --learning_rate ${LR_RATE} \
                    --weight_decay ${WEIGHT_DECAY:-0} \
                    --max_seq_len ${MAX_LEN} \
                    --vocab_size ${VOCAB_SIZE} \
                    --num_head ${NUM_HEAD} \
                    --d_model ${D_MODEL} \
                    --num_layers ${NUM_LAYER} \
                    --skip_steps 10 > log/job.log 2>&1
else
                    #--generate_neg_sample \
    python -u ./train.py --use_cuda \
                    --use_fast_executor \
                    --weight_sharing \
                    --batch_size ${BATCH_SIZE} \
                    --data_dir ${traindata_dir} \
                    --validation_set_dir ${testdata_dir} \
                    --checkpoints ./output \
                    --save_steps ${SAVE_STEPS} \
                    --init_model ${init_model:-""} \
                    --learning_rate ${LR_RATE} \
                    --weight_decay ${WEIGHT_DECAY:-0} \
                    --max_seq_len ${MAX_LEN} \
                    --vocab_size ${VOCAB_SIZE} \
                    --num_head ${NUM_HEAD} \
                    --d_model ${D_MODEL} \
                    --num_layers ${NUM_LAYER} \
                    --skip_steps 10 > log/job.log 2>&1
fi 
