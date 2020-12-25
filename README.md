# X-MOL
**large-scale pre-training for molecular understanding and diverse molecular analysis** <br>
<br>
Work-Flow:<br>
`.......................................|Molecular property prediction...` <br>
`..........tremendous data|.............|Drug-drug inteartion prediction.` <br>
`..large-scale transformer|----X-MOL----|Chemical reaction prediction....` <br>
`.powerful computing power|.............|Molecule generation.............` <br>
`.......................................|Molecule optimization...........` <br>
<br>
environment: <br>
The fine-tuning of X-MOL to prediction tasks and generation tasks are two irrelevant and independent part, the environment (including python and nccl should be downloaded and decompressed into both the two folder) <br>
<br>
    pre_trained X-MOL : https://1drv.ms/u/s!BIa_gVKaCDngi2S994lMsp-Y3TWK?e=l5hbxi <br>
    environment python : https://1drv.ms/u/s!Aoa_gVKaCDngi2OSr1svGMLLb2Xw?e=wwXaqP <br>
    environment nccl : https://1drv.ms/u/s!Aoa_gVKaCDngi2J7pOh7WdKR-pMa?e=GVlYbd <br>
<br>
requirements: <br>
1. python3 <br>
2. RDKit <br>

## Fine-tuning to prediction tasks
modify the **configuration file**: <br>
    `conf_pre/ft_conf.sh` <br>
the terms that need to be modified are **high-lighted**, like: <br>
    `### attention, this term need to be modified` <br>
    `vocab_path="./package/molecule_dict_zinc250k"` <br>
    `### attention, this term need to be modified` <br>
    `CONFIG_PATH="./package/ernie_zinc250k_config.json"` <br>
<br>
fine-tuning to **classification/regression**: <br>
modify the `main()` in `run_classifier.py` <br>
    1. for classification : `task_type = 'cls'` <br>
    2. for regression : `task_type = 'reg'` <br>
<br>
fine-tuning to **single-input/multiple-input**: <br>
modify the `main()` in `run_classifier.py` <br>
    1. for single-inpt : `multi_input = False` <br>
    2. for multiple-input : `multi_input = True` <br>
    modified the `main()` in `finetune_launch.py`: <br>
    `extend_sent = True` <br>
<br>
for **molecule property prediction task**: <br>
    1. **repeat training**: <br>
        modified `finetune_launch.py`, the code in `if __name__ == "__main__":`: <br>
        `while fine_tune_rep < the_numeber_of_repeating_times:` <br>
    2. **random/scaffold split**: <br>
        1. modified `finetune_launch.py`, the code in `if __name__ == "__main__":`: <br>
            keep the `subprocess.call("python3 pt_scaffold_split.py", shell=True)` <br>
        2. modified `pt_scaffold_split.py`, the code in `if __name__ == "__main__":`: <br>
            `sep_file_ex('path_to_training_data_folder', split_func='scaffold', amp=False, ampn=(0,0,0))` <br>
<br>
if the **vocab list** needs to be extended:<br>
modified the `main()` in `finetune_launch.py`: <br>
    `extend_vocab = False` <br>
<br>
run: <br>
    `sh train_ft.sh` <br>
    `sh train_lrtemb.sh` (knowlegde embedding) <br>

## Fine-tuning to generation tasks
modify the configuration file: <br>
    `ft_conf` <br>
the terms that need to be modified are **high-lighted**, like: <br>
    `### attention, this term need to be modified` <br>
    `vocab_path="./package/molecule_dict_zinc250k"` <br>
    `### attention, this term need to be modified` <br>
    `CONFIG_PATH="./package/ernie_zinc250k_config.json"` <br>
<br>
if the vocab list needs to be extended: <br>
modified the `main()` in `finetune_launch_local.py`: <br>
    `extend_vocab = True` <br>
    `extend_fc = True` <br>
<br>
run: <br>
    `sh train_ft.sh` (DL&GD generation tasks) <br>
    `sh train_opt.sh` (optimization tasks) <br>

## Change the number of GPUs used in the training process
for **both the two type tasks**: <br>
modify `finetune_launch.py` (`finetune_launch_local.py` in generation tasks) <br>
valid value of the two arguments in the argparse term `multip_g` <br>
    1. `nproc_per_node` <br>
    2. `selected_gpus` <br>

## Extend the vocab list
**the rules in the extension of vocabulary list:** <br>
    1. the extension must based on the `X-MOL_dict`, as well as the vocabularg list used in pre_training. <br>
    2. the extended vocab must be placed behind the original vocab (the index is start from 122). <br>
    3. do not forget open the `extend_vocab` in the `finetune_launch.py/finetune_launch_local.py`. <br>
    4. once the vocabulary list is extended, the pre-trained model will be changed, please make sure you have a good backup of X-MOL. <br>

