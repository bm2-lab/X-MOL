
task="cls"
gipus=8

task_name="MOLft_${task}_normal_ngram_50e"
submitter="xuedongyu"
fs_name="afs://xingtian.afs.baidu.com:9902"
fs_ugi="NLP_KM_Data,NLP_km_2018"
output_path="./ouptput/fine_tune/${task_name}"

mpi_on_k8s=1
mount_afs="true"

slurm_train_files_dir="afs://xingtian.afs.baidu.com:9902/user/NLP_KM_Data/zhanghan/kg_nerl_ids"
testdata_dir="./package/baidu_no_search_v2_valid/"

vocab_path="./package/mol/molecule_dict"
CONFIG_PATH="./package/mol/ernie_config.json"

stream_job=""
finetuning_task=$task
finetuning_data="./package/task_data/reactA/0s"

nodes=1

#init_model=""
init_model="./data/model/step_300000"
other_data=""

#model_based
use_sentencepiece="False"
lr_scheduler="noam_decay"
num_train_steps=1000000
use_fp16="false"
loss_scaling=128
generate_neg_sample="False"

IS_DISTRIBUTED="false"
EPOCH=452
VALID_STEPS=100
SAVE_STEPS=10000
WARMUP_STEPS=0
BATCH_SIZE=16
NUM_HEAD=12
LR_RATE=5e-6
WEIGHT_DECAY=0.01
D_MODEL=768
NUM_LAYER=12
MAX_LEN=512
export TOKENIZER="MolTokenizer"
num_labels=86

#export
export MODEL_PATH=$init_model
export TASK_DATA_PATH=$finetuning_data
export STREAM_JOB=$stream_job
export USE_SENTENCEPIECE=$use_sentencepiece
export USE_FP16=$use_fp16


