# X-MOL
large-scale pre-training for molecular understanding and diverse molecular analysis

Work-Flow:<br>
--------------------------|Molecular property prediction <br>
data--------|-------------|Drug-drug inteartion prediction <br>
transformer-|——— X-MOL ———|Chemical reaction prediction <br>
computing---|-------------|Molecule generation <br>
--------------------------|Molecule optimization <br>
<br>
<br>
The fine-tuning of X-MOL to prediction tasks and generation tasks are two irrelevant and independent part, the environment (including python and nccl should be downloaded and decompressed into both the two folder) <br>
pre_trained X-MOL : https://1drv.ms/u/s!BIa_gVKaCDngi2S994lMsp-Y3TWK?e=l5hbxi <br>
environment python : https://1drv.ms/u/s!Aoa_gVKaCDngi2OSr1svGMLLb2Xw?e=wwXaqP <br>
environment nccl : https://1drv.ms/u/s!Aoa_gVKaCDngi2J7pOh7WdKR-pMa?e=GVlYbd <br>
<br>
## fine-tuning to prediction tasks
`sh train_ft.sh` <br>
`sh train_lrtemb.sh` (knowlegde embedding) <br>
<br>
## fine-tuning to generation tasks
`sh train_ft.sh` <br>
