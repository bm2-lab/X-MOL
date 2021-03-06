# X-MOL : large-scale pre-training for molecular understanding and diverse molecular analysis

<br>

## Introduction of X-MOL

In silico modelling and analysis of small molecules substantially accelerates the process of drug development. Representing and understanding molecules is the fundamental step for various in silico molecular analysis tasks. Traditionally, these molecular analysis tasks have been investigated individually and separately. In this study, we presented X-MOL, which applies large-scale pre-training technology on 1.1 billion molecules for molecular understanding and representation, and then, carefully designed fine-tuning was performed to accommodate diverse downstream molecular analysis tasks, including molecular property prediction, chemical reaction analysis, drug-drug interaction prediction, de novo generation of molecules and molecule optimization. As a result, X-MOL was proven to achieve state-of-the-art results on all these molecular analysis tasks with good model interpretation ability. Collectively, taking advantage of super large-scale pre-training data and super-computing power, our study practically demonstrated the utility of the idea of "mass makes miracles" in molecular representation learning and downstream in silico molecular analysis, indicating the great potential of using large-scale unlabelled data with carefully designed pre-training and fine-tuning strategies to unify existing molecular analysis tasks and substantially enhance the performance of each task. <br>
In our study, X-MOL adopts a well-designed pre-training strategy to learn and understand the SMILES representation efficiently. Specifically, X-MOL designs a generative model during pre-training. In this way, the model is trained to generate a valid and equivalent SMILES representation from an input SMILES representation of the same molecule. This generative training strategy ultimately results in a pre-trained model with a good understanding of the SMILES representation, and it can generate the correct SMILES of the given molecule quite well. As a result, X-MOL builds a super large-scale pre-training model based on the Transformer, which is composed of 12 encoder-decocder layers, 768-dimensional hidden units and 12 attention heads. <br>
Specifically, our generative pre-training strategy is implemented by an encoder-decoder architecture, but it is different from traditional encoder-decoder architectures such as those used in neural machine translation (NMT), as the encoder and decoder in X-MOL share the same layers. In X-MOL, the input random SMILES and output random SMILES are sent into the model simultaneously, and the output random SMILES is totally masked. In addition, only a unidirectional attention operation can be performed within the output random SMILES, which means that each character in the output random SMILES can pay attention only to itself and the previously generated characters. In this way, the shared-layer encoder-decoder architecture in X-MOL is able to unifiy the semantic comprehension of encoder and decoder, also the shared-layer architecture could reduce the number of parameters significantly compared with traditional encoder-decoder architectures. <br>

## Work-flow of X-MOL
`.......Pre-training.................................Fine-tuning...........` <br>
<br>
`.........................................|Molecular property prediction...` <br>
`..........Tremendous data|...............|Drug-drug inteartion prediction.` <br>
`..Large-scale transformer|---> X-MOL --->|Chemical reaction prediction....` <br>
`.Powerful computing power|...............|Molecule generation.............` <br>
`.........................................|Molecule optimization...........` <br>
<br>
![image](https://github.com/bm2-lab/X-MOL/blob/main/images/fig-overview.png) <br>

## Environment
**We provide the pre-trained X-MOL and the script of fine-tuning X-MOL as well as the environment** <br>
Environment: <br>
The fine-tuning of X-MOL to prediction tasks and generation tasks are two irrelevant and independent part, the environment (including python and nccl) should be downloaded and decompressed into both the two folders <br>
<br>
**The provided environment :** <br>
    - Pre_trained X-MOL : https://1drv.ms/u/s!BIa_gVKaCDngi2S994lMsp-Y3TWK?e=l5hbxi <br>
    - Environment-python : https://1drv.ms/u/s!Aoa_gVKaCDngi2U1ip8w2HxjIt4-?e=koGl4c <br>
    - Environment-nccl : https://1drv.ms/u/s!Aoa_gVKaCDngi2J7pOh7WdKR-pMa?e=GVlYbd <br>
**Requirements :** <br> 
    - Python3.7 (although the environment of model traininng, python2, is provided above, the process of preprocessing data and model evaluation is based on a python3 environment) <br>
    - RDKit (2019.09.1.0) <br>

## Fine-tuning to prediction tasks
1. Modify the **configuration file** : <br>
   `conf_pre/ft_conf.sh` <br>
   The terms that need to be modified are **high-lighted**, like : <br>
   `### attention, this term need to be modified` <br>
   `vocab_path="./package/molecule_dict_zinc250k"` <br>
   `### attention, this term need to be modified` <br>
   `CONFIG_PATH="./package/ernie_zinc250k_config.json"` <br>
   <br>
2. Fine-tuning to **classification/regression** : <br>
   Modify the `main()` in `run_classifier.py` <br>
   1. For classification : `task_type = 'cls'` <br>
   2. For regression : `task_type = 'reg'` <br>
      <br>
3. Fine-tuning to **single-input/multiple-input** : <br>
   Modify the `main()` in `run_classifier.py` <br>
   1. For single-input : `multi_input = False` <br>
   2. For multiple-input : `multi_input = True` <br>
      Modify the `main()` in `finetune_launch.py`: <br>
      `extend_sent = True` <br>
      Modify the `"type_vocab_size"` in model config <br>
      <br>
4. For **molecule property prediction task** : <br>
   1. **Repeat training**: <br>
      Modify `finetune_launch.py`, the code in `if __name__ == "__main__":` : <br>
      `while fine_tune_rep < the_numeber_of_repeating_times:` <br>
   2. **Random/scaffold split**: <br>
      - Modify `finetune_launch.py`, the code in `if __name__ == "__main__":` : <br>
         Keep the `subprocess.call("python3 pt_scaffold_split.py", shell=True)` <br>
      - Modify `pt_scaffold_split.py`, the code in `if __name__ == "__main__":` : <br>
         `sep_file_ex('path_to_training_data_folder', split_func='scaffold', amp=False, ampn=(0,0,0))` <br>
         <br>
5. If the **vocab list** needs to be extended :<br>
   Modify the `main()` in `finetune_launch.py` : <br>
    `extend_vocab = False` <br>
   <br>
6. **Run** : <br>
   `sh train_ft.sh` <br>
   `sh train_lrtemb.sh` (knowlegde embedding) <br>

## Fine-tuning to generation tasks
1. Modify the **configuration file** : <br>
   `ft_conf` <br>
   The terms that need to be modified are **high-lighted**, like : <br>
   `### attention, this term need to be modified` <br>
   `vocab_path="./package/molecule_dict_zinc250k"` <br>
   `### attention, this term need to be modified` <br>
   `CONFIG_PATH="./package/ernie_zinc250k_config.json"` <br>
   <br>
2. If the **vocab list** needs to be extended : <br>
   Modify the `main()` in `finetune_launch_local.py`: <br>
    `extend_vocab = True` <br>
    `extend_fc = True` <br>
   <br>
3. **Run** : <br>
   `sh train_ft.sh` (DL&GD generation tasks) <br>
   `sh train_opt.sh` (optimization tasks) <br>

## Change the number of GPUs used in the training process
For **both the two type tasks** : <br>
Modify `finetune_launch.py` (`finetune_launch_local.py` in generation tasks) <br>
Valid value of the two arguments in the argparse term `multip_g` <br>
    1. `nproc_per_node` : GPU numbers <br>
    2. `selected_gpus` : GPU ids<br>

## Extend the vocab list
**The rules in the extension of vocabulary list** : <br>
    1. The extension must based on the `X-MOL_dict`, as well as the vocabularg list used in pre_training. <br>
    2. The extended vocab must be placed behind the original vocabs (the index of new vocabs is start from 122). <br>
    3. Do not forget to turn on the `extend_vocab` in the `finetune_launch.py/finetune_launch_local.py`. <br>
    4. Do not forget to modify the `"vocab_size"` in model config <br>
    5. Once the vocabulary list is extended, the pre-trained model will be changed, please make sure you have a good backup of X-MOL. <br>

## Fine-tuning output
**Path of saving the log file and saved model** : <br>
    1. Log files are saved in `./log/`, a launching log and n running log will be saved (n = the number of GPUs). <br>
    2. The saved model (parameter `SAVE_STEPS=1000` in ft_conf.sh incidates that the model will be stored every 1000 steps during the training process) will be stored in `./checkpoints/`. <br>

## Warm start and cold start
**Warm start** : <br>
Fine-tuning the model on the basis of X-MOL <br>
Set parameter `init_model` in `ft_conf/ft_conf.sh` as `init_model="path/to/decompressed/X-MOL"` <br>
**Cold start** : <br>
Training the model from scratch <br>
Set parameter `init_model` in `ft_conf/ft_conf.sh` as `init_model=""` <br>

## Contact
1810538@tongji.edu.cn or qiliu@tongji.edu.cn
