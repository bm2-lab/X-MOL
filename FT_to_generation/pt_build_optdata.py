from multiprocessing import Process
import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from random import shuffle
import numpy as np
import time

def find_similarity(smis, ref, best_n, ids):
    ref_fp = [Chem.RDKFingerprint(Chem.MolFromSmiles(s)) for s in ref]
    smis_fp = [Chem.RDKFingerprint(Chem.MolFromSmiles(s)) for s in smis]
    #fp_sim = {}
    mol_sim = {}
    for idx, smi in enumerate(smis_fp):
        #fp_sim[idx] = []
        mol_sim[smis[idx]] = []
        sims = []
        for ridx, rsmi in enumerate(ref_fp):
            sim = DataStructs.TanimotoSimilarity(smi, rsmi)
            #sim = DataStructs.DiceSimilarity(smi, rsmi)
            sims.append(sim)
        for _ in range(best_n):
            best_idx = np.argmax(sims)
            #fp_sim[idx].append(best_idx)
            mol_sim[smis[idx]].append(ref[best_idx])
            sims.pop(best_idx)
    with open('subprocess_{0}'.format(ids), 'w') as f:
        for smi in mol_sim:
            f.write('{0}:{1}\n'.format(smi, ','.join(mol_sim[smi])))
    

def multi_process(ref, n_jobs, best_n=10, to_file=False):
    """
    Building training data for optimization task
    para ref: reference SMILES
    para n_jobs: number of processings in multi-processing
    para to_file: the output file name, default False, means return the metircs to the python console
    """
    # prepare tasks for each subprocesses
    n_jobs = max(n_jobs, 1)
    data_num = len(ref)
    smisn_per_job = round(data_num/n_jobs)
    ref_lists = [ref[i*smisn_per_job:(i+1)*smisn_per_job] for i in range(n_jobs-1)]
    ref_lists.append(ref[(n_jobs-1)*smisn_per_job:])
    # define subprocesses and call the subprocesses
    sub_process = []
    for sp in range(n_jobs):
        sub_process.append(Process(target=find_similarity, args=(ref_lists[sp], ref, best_n, sp)))
    for sp in sub_process:
        sp.start()
    for sp in sub_process:
        sp.join()
    # merge files and remove temporary files
    merged_ = []
    for spf in range(n_jobs):
        with open('subprocess_{0}'.format(spf)) as f:
            for l in f:
                merged_.append(l)
        subprocess.call('rm subprocess_{0}'.format(spf), shell=True)
    
    if to_file:
        with open(to_file, 'w') as f:
            for rec in merged_:
                f.write(rec)
    else:
        return merged_

if __name__ == '__main__':
    ref_file = './all_data'
    to_file = 'similarity_lib'
    n_jobs = 12
    best_n = 10

    with open(ref_file,'r') as f:
        ref_smis = []
        for l in f:
            ref_smis.append(l.strip().split('\t')[1])
    k = multi_process(ref_smis, n_jobs, best_n, to_file)

