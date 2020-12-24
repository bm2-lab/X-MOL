# X-MOL
large-scale pre-training for molecular understanding and diverse molecular analysis

Work-Flow:<br>
--------------------------|Molecular property prediction <br>
data--------|-------------|Drug-drug inteartion prediction <br>
transformer-|——— X-MOL ———|Chemical reaction prediction <br>
computing---|-------------|Molecule generation <br>
--------------------------|Molecule optimization <br>
<br>
The fine-tuning of X-MOL to prediction tasks and generation tasks are two irrelevant and independent part, the environment (including python and nccl should be downloaded and decompressed into both the two folder) <br>
<br>
pre_trained X-MOL : https://1drv.ms/u/s!BIa_gVKaCDngi2S994lMsp-Y3TWK?e=l5hbxi <br>
environment python : https://1drv.ms/u/s!Aoa_gVKaCDngi2OSr1svGMLLb2Xw?e=wwXaqP <br>
environment nccl : https://1drv.ms/u/s!Aoa_gVKaCDngi2J7pOh7WdKR-pMa?e=GVlYbd <br>
<br>
## fine-tuning to prediction tasks
run: <br>
`sh train_ft.sh` <br>
`sh train_lrtemb.sh` (knowlegde embedding) <br>
modify configuration: <br>
conf_pre/ft_conf.sh <br>
the terms that need to be modified are high-lighted, like: <br>
`### attention, this term need to be modified` <br>
`vocab_path="./package/molecule_dict_zinc250k"` <br>
`### attention, this term need to be modified` <br>
`CONFIG_PATH="./package/ernie_zinc250k_config.json"` <br>
<br>
## fine-tuning to generation tasks
run: <br>
`sh train_ft.sh` <br>
modify configuration: <br>
ft_conf <br>
the terms that need to be modified are high-lighted, like: <br>
`### attention, this term need to be modified` <br>
`vocab_path="./package/molecule_dict_zinc250k"` <br>
`### attention, this term need to be modified` <br>
`CONFIG_PATH="./package/ernie_zinc250k_config.json"` <br>
if the vocab list needs to be extended, modified the `main()` in `finetune_launch_local.py`: <br>
`extend_vocab = False` <br>
`extend_fc = False` <br>


## change the number of GPUs used in the training process
for both the two type tasks: <br>
finetune_launch.py (finetune_launch_local.py in generation tasks) <br>
modify the valid value in the argparse <br>
1. nproc_per_node
2. selected_gpus