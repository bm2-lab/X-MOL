from multiprocessing import Process
import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
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

def eva_opt(file_list, ref_train, ref_dev, ids=None):
    """
    The Distribution-Learning evaluation of the generated SMILES 
    para file_list: the files store the generated SMILES, in list format
    para ref_train: training reference, SMILES
    para ref_dev: dev reference, in RDKit.MOL class
    para ids: job id in multi-process, default None, and would return the metircs in Dataframe, otherwise will write to a csv file
    """
    rec_file = open('eva_rec.log','a')
    vs = []
    ns = []
    ss = []
    ds = []
    ni = []
    for idx, file in enumerate(file_list):
        smis = read_dev_output(file)
        valid_e = []
        novelty_e = 0
        sim_e = []
        optv_e = []
        ni_e = 0
        for sidx, smi in enumerate(smis):
            try:
                mol = Chem.MolFromSmiles(smi)
                q = Descriptors.qed(mol)
            except:
                pass
            else:
                valid_e.append(smi)
                if smi not in ref_train:
                    novelty_e += 1
                org_q = Descriptors.qed(ref_dev[sidx])
                opt_q = q - org_q
                if opt_q > 0:
                    ni_e += 1
                optv_e.append(opt_q)
                sim = DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(ref_dev[sidx]), Chem.RDKFingerprint(mol))
                sim_e.append(sim)
        vs.append(len(valid_e)/len(smis))
        ns.append(novelty_e/len(valid_e))
        ss.append(np.mean(sim_e))
        ds.append(np.mean(optv_e))
        ni.append(ni_e/len(valid_e))
        rec_file.write('file : {0} done\n'.format(file))
    rec_file.close()
    opt_metrics = pd.DataFrame({'valid_score':vs, 'novelty_score':ns, 'similarity_score':ss, 'ave_opt':ds, 'frac_inc':ni})
    if ids == None:
        return opt_metrics
    else:
        opt_metrics.to_csv('subprocess_{0}.csv'.format(ids), index=False)

def multi_process(file_dir, file_n, ref_dir, n_jobs, to_file=False):
    """
    Evaluate the optimized SMILES in multi-processing
    para eva_func: evaluation function
    para file_dir: the dir to where store the generated molecules
    para file_n: number of store file
    para ref_dir: reference dir, contain train.tsv & dec.tsv
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

    if ref_dir.endswith('/'):
        dev_file = ref_dir+'dev.tsv'
        train_file = ref_dir+'train.tsv'
    else:
        dev_file = ref_dir+'/dev.tsv'
        train_file = ref_dir+'/train.tsv'
    with open(dev_file, 'r') as df:
        h = 1
        ref_dev = []
        for l in df:
            if h:
                h = 0
            else:
                ref_smi = l.split('\t')[0]
                ref_dev.append(Chem.MolFromSmiles(ref_smi))
    with open(train_file, 'r') as tf:
        h = 1
        ref_train = [[],[]]
        for l in tf:
            if h:
                h = 0
            else:
                smi_0, smi_1 = l.strip().split('\t')
                ref_train[0].append(smi_0)
                ref_train[1].append(smi_1)
        ref_train = ref_train[1]

    # define subprocesses and call the subprocesses
    sub_process = []
    for sp in range(n_jobs):
        sub_process.append(Process(target=eva_opt, args=(file_lists[sp], ref_train, ref_dev, sp)))
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
    ref_dir = './package/task_data/zinc250k_opt/'
    n_jobs = 10
    file_dir = './output_store/opt_scale0701_cold/'
    #file_dir = './checkpoints/'
    file_n = 100
    to_file = './opt_scale0701_cold.csv'
    #to_file = None
    start_time = time.time()
    k = multi_process(file_dir, file_n, ref_dir, n_jobs, to_file)
    end_time = time.time()

