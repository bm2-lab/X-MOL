task="test"

task_name="ernie_${task}_local"
submitter="xuedongyu"
hdfs_path="afs://xingtian.afs.baidu.com:9902"
ugi="NLP_KM_Data,NLP_km_2018"
hdfs_output="/user/NLP_KM_Data/xuedong/ernie_finetune/local_test"

slurm_train_files_dir="afs://xingtian.afs.baidu.com:9902/user/NLP_KM_Data/zhanghan/kg_nerl_ids"
testdata_dir="./package/baidu_no_search_v2_valid/"

train_filelist="./package/ernie_molecule_char_v1/train.filelist"
valid_filelist="./package/ernie_molecule_char_v1/valid.filelist"

vocab_path="./package/ernie_molecule_char_v1/molecule_dict"
CONFIG_PATH="./package/ernie_molecule_char_v1/ernie_config.json"

stream_job=""
finetuning_task=$task
finetuning_data="./package/task_data/pcba/"

queue="yq01-p40-3-8"
#queue="yq01-p40-4-8"
#queue="nlp-temp"
#queue="yq01-p40-box-nlp-1-8"

#queue="yq01-p40-box-1-8"
#queue="yq01-v100-box-1-8"
#queue="yq01-v100-box-2-8"

nodes=1

#hdfs_init_model="/user/NLP_KM_Data/liyukun01/ernie-x/ernie_base_molecule_char_v1/5854d09e-f524-55a6-b708-61c786dc18b9/job-0bb5ddf946a0e47c/job-1678a321fc07000/output/rank-10.255.67.20/step_30000_20191128214008.tar"
init_model=""
other_data=""

#model_based
use_sentencepiece="False"
lr_scheduler="noam_decay"
num_train_steps=1000000
use_fp16="False"
loss_scaling=128
generate_neg_sample="False"

SAVE_STEPS=10000
WARMUP_STEPS=0
#BATCH_SIZE=18432
#BATCH_SIZE=12288
BATCH_SIZE=8192
NUM_HEAD=12
LR_RATE=5e-5
WEIGHT_DECAY=0.01
D_MODEL=768
NUM_LAYER=12
MAX_LEN=512
VOCAB_SIZE=112
#VOCAB_SIZE=17999
#VOCAB_SIZE=12055
export TOKENIZER="MolTokenizer"

#export
export MODEL_PATH=$init_model
export TASK_DATA_PATH=$finetuning_data
export STREAM_JOB=$stream_job
export USE_SENTENCEPIECE=$use_sentencepiece
export USE_FP16=$use_fp16
