from multiprocessing import Process
import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from random import shuffle
import numpy as np
import time

def read_dev_output(file_name):
    # read generated SMILES in seq2seq dev output file
    with open(file_name,'r') as f:
        smis = []
        for l in f:
            smi = ''.join(l.split('\t')[0].split(' '))
            smis.append(smi)
    return smis

def get_novelty(gen_smis, ref_smis, return_novelty=False, ref_can=False):
    """
    Get novelty generated MOLs which are not exist in training dataset
    para gen_smis: generated SMILES, in list format
    para ref_smis: training SMILES, in list format
    para return_novelty: if return novelty MOLs, in canonical SMILES format, default False
    """
    c_gen_smis = []
    for s in gen_smis:
        try:
            cs = Chem.MolToSmiles(Chem.MolFromSmiles(s))
        except:
            pass
        else:
            c_gen_smis.append(cs)
    if ref_can:
        c_ref_smis = ref_smis
    else:
        c_ref_smis = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in ref_smis]
        c_ref_smis = list(set(c_ref_smis))
    c_gen_smis = list(set(c_gen_smis))
    nove_smis = [i for i in c_gen_smis if i not in c_ref_smis]

    if return_novelty:
        return nove_smis
    else:
        return len(nove_smis)/len(gen_smis)

def get_novelty_smi(gen_smis, ref_smis, return_novelty=False,):
    """
    Get novelty generated SMILES which are not exist in training dataset
    para gen_smis: generated SMILES, in list format
    para ref_smis: training SMILES, in list format
    para return_novelty: if return novelty MOLs, in canonical SMILES format, default False
    """
    
    nov_smis = [i for i in gen_smis if i not in ref_smis]

    if return_novelty:
        return nov_smis
    else:
        return len(nov_smis)/len(gen_smis)

def get_valid(gen_smis, return_valid=False):
    """
    Get valid SMILES in generated samples
    para gen_smis: generated SMILES, in list format
    para return_valid: if return unique SMILESs, else return the fraction, default False
    """
    valid_smi = []
    for smi in gen_smis:
        try:
            m = Chem.MolFromSmiles(smi)
        except:
            pass
        else:
            if m != None:
                valid_smi.append(smi)
    if return_valid:
        return valid_smi
    else:
        return len(valid_smi)/len(gen_smis)

def get_unique(gen_smis, random_sample_n=-1, valid=True, return_unique=False):
    """
    Get unique generated samples
    para gen_smis: generated SMILES, in list format
    para random_sample_n: the number of sampled SMILESs from gen_smis for uniqueness calculation, 
                          -1 means using the whole gen_smis, default -1
    para valid:  if the unique SMILES should be valid SMILES
    para return_unique: if return unique SMILESs, default False
    """
    base = get_valid(gen_smis, return_valid=True) if valid else gen_smis
    total_smi_n = len(base)
    if random_sample_n>total_smi_n or random_sample_n == -1:
        sample_n = total_smi_n
    else:
        sample_n = random_sample_n
    base_index = list(range(total_smi_n))
    shuffle(base_index)
    sample_smis = [base[base_index[i]] for i in range(sample_n)]
    unique_smis = list(set(sample_smis))

    if return_unique:
        return unique_smis
    else:
        if sample_n == 0:
            return 0
        else:
            return len(unique_smis)/sample_n

def eva_dl(file_list, ref, ids):
    """
    The Distribution-Learning evaluation of the generated SMILES 
    para file_list: the files store the generated SMILES, in list format
    para ref: the number of sampled SMILESs from gen_smis for uniqueness calculation, 
                          -1 means using the whole gen_smis, default -1
    para ids: job id in multi-process, default None, and would return the metircs in Dataframe, otherwise will write to a csv file
    """
    rec_file = open('eva_rec.log','a')
    ref_smis = ref
    vs = []
    us = []
    ns = []
    for idx, file in enumerate(file_list):
        smis = read_dev_output(file)
        v_smis = get_valid(smis, return_valid=True)
        n_smis = get_novelty_smi(v_smis, ref_smis, return_novelty=True)
        vs.append(len(v_smis)/len(smis))
        us.append(get_unique(smis))
        ns.append(len(n_smis)/len(v_smis))
        rec_file.write('DL-evaluation for {0} done\n'.format(file))
    rec_file.close()
    dl_metrics = pd.DataFrame({'valid_score':vs, 'unique_score':us, 'novelty_score':ns})
    if ids == None:
        return dl_metrics
    else:
        dl_metrics.to_csv('subprocess_{0}.csv'.format(ids), index=False)

def eva_gd(file_list, target, ids):
    """
    The Goal-Directed evaluation of the generated SMILES 
    para file_list: the files store the generated SMILES, in list format
    para target: the pre-defined goal for generated SMILES, in list format
    para ids: job id in multi-process, default None, and would return the metircs in Dataframe, otherwise will write to a csv file
    """
    rec_file = open('eva_rec.log','a')
    ave_diff = []
    ave_p = []
    top_1 = []
    top_2 = []
    top_3 = []
    for idx, file in enumerate(file_list):
        smis = read_dev_output(file)

        if len(smis) != len(target):
            cut_ = min(len(smis), len(target))
            smis = smis[:cut_]
            target_e = target[:cut_]
        else:
            target_e = target[:]

        properties = [0,0,0]
        diff = []
        for sidx, smi in enumerate(smis):
            try:
                mol = Chem.MolFromSmiles(smi)
                q = Descriptors.qed(mol)
            except:
                pass
            else:
                diff.append(abs(q-target_e[sidx]))
                properties.append(q)
        properties = sorted(properties)[::-1]
        top_1.append(properties[0])
        top_2.append(properties[1])
        top_3.append(properties[2])
        ave_p.append(np.mean(properties))
        ave_diff.append(np.mean(diff))
        rec_file.write('GD-evaluation for {0} done\n'.format(file))

    rec_file.close()
    gd_metrics = pd.DataFrame({'ave_diff':ave_diff, 'ave_property':ave_p, 'top_1':top_1, 'top_2':top_2, 'top_3':top_3})
    if ids == None:
        return gd_metrics
    else:
        gd_metrics.to_csv('subprocess_{0}.csv'.format(ids), index=False)
            
def multi_process(eva_func, file_dir, file_n, ref, n_jobs, to_file=False):
    """
    Evaluate the generated SMILES in multi-processing
    para eva_func: evaluation function
    para file_dir: the dir to where store the generated molecules
    para file_n: number of store file
    para ref: reference, training SMILES in Distribution-Learning evaluation and target value in Goal-Directed evaluation
    para n_jobs: number of processings in multi-processing
    para to_file: the output file name, default False, means return the metircs to the python console
    """
    # prepare tasks for each subprocesses
    n_jobs = max(n_jobs, 1)
    if file_dir.endswith('/'):
        file_list = ['{0}dev_epoch{1}'.format(file_dir, i) for i in range(file_n)]
    else:
        file_list = ['{0}/dev_epoch{1}'.format(file_dir, i) for i in range(file_n)]
    filen_per_job = round(file_n/n_jobs)
    file_lists = [file_list[i*filen_per_job:(i+1)*filen_per_job] for i in range(n_jobs-1)]
    file_lists.append(file_list[(n_jobs-1)*filen_per_job:])
    # define subprocesses and call the subprocesses
    sub_process = []
    for sp in range(n_jobs):
        sub_process.append(Process(target=eva_func, args=(file_lists[sp], ref, sp)))
    for sp in sub_process:
        sp.start()
    for sp in sub_process:
        sp.join()
    # merge files and remove temporary files
    for spf in range(n_jobs):
        sbcsv = pd.read_csv('subprocess_{0}.csv'.format(spf))
        if spf == 0:
            merged_ = sbcsv
        else:
            merged_ = merged_.append(sbcsv)
        subprocess.call('rm subprocess_{0}.csv'.format(spf), shell=True)
    
    merged_.index = list(range(len(merged_)))
    if to_file:
        merged_.to_csv(to_file)
    else:
        return merged_

if __name__ == '__main__':
    ref_file = './package/task_data/zinc250k_gd/train.tsv'
    n_jobs = 12
    #file_dir = './output_store/qm9_dl/step40w/'
    file_dir = './output_store/zinc250k_gd_200'
    file_n = 240
    to_file = 'zinc250k_gd_gd_40w.csv'
    with open(ref_file,'r') as f:
        ref_smis = []
        h = 1
        for l in f:
            if h :
                h = 0
            else:
                ref_smis.append(l.strip().split('\t')[1])
    tgt = [0.948 for _ in range(10000)]
    start_time = time.time()
    #k = multi_process(eva_dl, file_dir, file_n, ref_smis, n_jobs, to_file)
    k = multi_process(eva_gd, file_dir, file_n, tgt, n_jobs, to_file)
    #file_lists = ['./output_store/zinc250k_gd/dev_epoch{0}'.format(i) for i in range(file_n)]
    #k = eva_gd(file_lists, tgt, None)
    end_time = time.time()
