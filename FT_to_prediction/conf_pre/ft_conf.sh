task="cls"
gipus=8

task_name="MOLft_${task}"
submitter=""
fs_name=""
fs_ugi=""
output_path="./ouptput/fine_tune/${task_name}"

mpi_on_k8s=1
mount_afs="true"

slurm_train_files_dir=""
testdata_dir=""

### attention, this term need to be modified
vocab_path="./package/mol/molecule_dict"

### attention, this term need to be modified
CONFIG_PATH="./package/mol/ernie_config.json"


stream_job=""
finetuning_task=$task

### attention, this term need to be modified
finetuning_data="path_to_training_data_folder"

nodes=1

### attention, this term need to be modified
init_model="./data/model/step_400000"
#init_model=""

#CKPT_PATH="./checkpoints/step_54000"
other_data=""

#model_based
use_sentencepiece="False"
lr_scheduler="noam_decay"
num_train_steps=1000000
use_fp16="false"
loss_scaling=128
generate_neg_sample="False"

IS_DISTRIBUTED="false"

### attention, this term need to be modified
EPOCH=40

### attention, this term need to be modified
VALID_STEPS=1800

SAVE_STEPS=5400
WARMUP_STEPS=0

### attention, this term need to be modified
BATCH_SIZE=16

NUM_HEAD=12

### attention, this term need to be modified
LR_RATE=5e-5

WEIGHT_DECAY=0.01
D_MODEL=768
NUM_LAYER=12
MAX_LEN=512
export TOKENIZER="MolTokenizer"

### attention, this term need to be modified
### in regression tasks, the value of num_labels would not influence the task
num_labels=2

#export
export MODEL_PATH=$init_model
export TASK_DATA_PATH=$finetuning_data
export STREAM_JOB=$stream_job
export USE_SENTENCEPIECE=$use_sentencepiece
export USE_FP16=$use_fp16


