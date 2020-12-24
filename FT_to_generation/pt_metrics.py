from rdkit import Chem
from rdkit.Chem import MACCSkeys
from random import shuffle
import numpy as np
import pandas as pd
#import torch
import scipy
#from pt_metrics_utils.utils import fingerprints, average_agg_tanimoto, get_mol, mol_passes_filters
from pt_metrics_utils.utils import fingerprints, get_mol, mol_passes_filters
from rdkit.Chem import Descriptors

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

def get_unique(gen_smis, random_sample_n=-1, valid=False, return_unique=False):
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
        return len(unique_smis)/sample_n

def get_unique_mol(gen_smis, random_sample_n=-1, return_unique=False):
    """
    Get unique generated MOL
    para gen_smis: generated SMILES, in list format
    para random_sample_n: the number of sampled SMILESs from gen_smis for uniqueness calculation
                          -1 means using the whole gen_smis, default -1
    para return_unique: if return unique MOLs, in canonical SMILES format, default False
    """
    total_smi_n = len(gen_smis)
    base_index = list(range(total_smi_n))
    shuffle(base_index)
    if random_sample_n>total_smi_n or random_sample_n == -1:
        sample_n = total_smi_n
    else:
        sample_n = random_sample_n
    sampled_smi = [gen_smis[base_index[i]] for i in range(sample_n)]
    canonical_smi = []
    for smi in sampled_smi:
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            pass
        else:
            if mol != None:
                c_smi = Chem.MolToSmiles(mol)
                canonical_smi.append(c_smi)
    canonical_smi = list(set(canonical_smi))

    if return_unique:
        return canonical_smi
    else:
        return len(canonical_smi)/sample_n

"""
def get_internal_diversity(gen_smis, fp_type='morgan', p=1):
    # function from MOSES
    
    #  Computes internal diversity as:
    #  1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    
    gen_fps = fingerprints(gen_smis, fp_type=fp_type)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps, agg='mean', p=p)).mean()
"""

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

def get_passes_filters(gen_smis, return_passed=False):
    # function from MOSES
    # term allowed atoms is masked
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passed_smis = []
    for smi in gen_smis:
        passes = mol_passes_filters(smi)
        if passes:
            passed_smis.append(smi)
    if return_passed:
        return passed_smis
    else:
        return len(passed_smis)/len(gen_smis)

def read_dev_output(file_name):
    # read generated SMILES in seq2seq dev output file
    with open(file_name,'r') as f:
        smis = []
        for l in f:
            smi = ''.join(l.split('\t')[0].split(' '))
            smis.append(smi)
    return smis

def eva(file_n, file_dir, ref):
    rec_file = open('eva_rec.log','w')
    if file_dir.endswith('/') == False:
        file_dir += '/'
    with open(ref,'r') as f:
        d = []
        for l in f:
            d.append(l.strip())
    d = d[1:]
    train_file = ref.rstrip('dev.tsv') + 'train.tsv'
    with open(train_file,'r') as f:
        t_smis = []
        for l in f:
            t_smis.append(l.strip().split('\t')[-1])
    t_smis = t_smis[1:]
    t_smis = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in t_smis]
    t_smis = list(set(t_smis))
    #print(d[0])
    if len(d[0].split('\t')) == 1:
        flag = 'DL'
        ref_smis = [i.split('\t')[-1] for i in d]
    else:
        flag = 'GD'
        ref_smis = [i.split('\t')[-1] for i in d]
        ref_values = [i.split('\t')[0] for i in d]
        ref_values = [float(i.rstrip(']').lstrip('[')) for i in ref_values]
    vs = []
    us = []
    ums = []
    ns = []
    value_store = {}
    for i in range(file_n):
        smis = read_dev_output(file_dir+'dev_epoch{0}'.format(i))
        v_smis = get_valid(smis, return_valid=True)
        um_smis = get_unique(v_smis, return_unique=True)
        print('*'*20)
        print('epoch : {0}'.format(i))
        print('*'*20)
        n_smis = get_novelty(v_smis, t_smis, return_novelty=True, ref_can=True)
        base_n = len(smis)
        vs.append(len(v_smis)/base_n)
        us.append(get_unique(smis))
        ums.append(len(um_smis)/base_n)
        ns.append(len(n_smis)/base_n)
        if flag == 'GD':
            vsc = []
            for s in smis:
                try:
                    mol = Chem.MolFromSmiles(s)
                    v = Descriptors.qed(mol)
                except:
                    vsc.append('inv')
                else:
                    vsc.append(int(10*v)*0.1)
            value_store[i] = vsc
        rec_file.write('dev_epoch{0} done\n'.format(i))
    rec_file.close()
    dl_b = pd.DataFrame({'valid_score':vs, 'unique_score':us, 'unique_molecule_score':ums, 'novelty_socre':ns})
    if flag == 'GD':
        value_diff = {i*0.1:[] for i in range(10)}
        value_diff['inv'] = []
        for e in value_store:
            for dv in value_diff:
                value_diff[dv].append(0)
            for idx,s in enumerate(value_store[e]):
                if s == 'inv':
                    value_diff['inv'][-1] += 1
                else:
                    diffv = abs(s-ref_values[idx])
                    diffv = round(10*diffv) *0.1
                    value_diff[diffv][-1] += 1
        sc = []
        for e in range(file_n):
            n = 0
            for k in value_diff.keys():
                if k == 'inv':
                    alpha = 0.0
                else:
                    alpha = 1-k
                n += alpha*value_diff[k][e]
            sc.append(n)
        value_diff['score'] = sc
        gd_b = pd.DataFrame(value_diff)
        return dl_b, gd_b

    else:
        return dl_b
                 


def eva_f(file_n, file_dir, ref, novelty=True, sc_func='qed', ref_v=None):
    rec_file = open('eva_rec.log','w')
    if type(sc_func) == 'str':
        sc_func = sc_func.lower()
        score_func = {'qed':Descriptors.qed, 'logp':Descriptors.MolLogP, 'mw':Descriptors.MolWt, 'tpsa':Descriptors.TPSA}[sc_func]
    else:
        score_func = sc_func
    with open(ref,'r') as f:
        d = []
        for l in f:
            d.append(l.strip())
    d = d[1:]
    train_file = ref.rstrip('dev.tsv') + 'train.tsv'
    with open(train_file,'r') as f:
        t_smis = []
        for l in f:
            t_smis.append(l.strip().split('\t')[-1])
    t_smis = t_smis[1:]
    #print(d[0])
    flag = 'GD'
    ref_smis = [i.split('\t')[-1] for i in d]
    if ref_v == None:
        ref_values = [float(i.split('\t')[0]) for i in d]
    elif isinstance(ref_v,list) and len(ref_v) == len(d):
        ref_values = ref_v
    else:
        ref_values = [ref_v] * len(d)
    vs = []
    us = []
    ums = []
    ns = []
    value_diff = []
    for i in range(file_n):
        smis = read_dev_output(file_dir+'dev_epoch{0}'.format(i))
        v_smis = get_valid(smis, return_valid=True)
        um_smis = get_unique(v_smis, return_unique=True)
        print('*'*20)
        print('epoch : {0}'.format(i))
        print('*'*20)
        if novelty:
            n_smis = get_novelty(v_smis, t_smis, return_novelty=True)
            ns.append(len(n_smis)/base_n)
        base_n = len(smis)
        vs.append(len(v_smis)/base_n)
        us.append(get_unique(smis))
        ums.append(len(um_smis)/base_n)
        vsc = []
        for idx, s in enumerate(smis):
            try:
                mol = Chem.MolFromSmiles(s)
                v = Descriptors.MolWt(mol)
            except:
                pass
            else:
                vsc.append(abs(v-ref_values[idx]))
        if len(vsc) != 0:
            value_diff.append(sum(vsc)/len(vsc))
        else:
            value_diff.append('nan')
        rec_file.write('dev_epoch{0} done\n'.format(i))
    rec_file.close()
    if novelty:
        dl_b = pd.DataFrame({'valid_score':vs, 'unique_score':us, 'unique_molecule_score':ums, 'novelty_socre':ns, 'avg_value_difference':value_diff})
    else:
        dl_b = pd.DataFrame({'valid_score':vs, 'unique_score':us, 'unique_molecule_score':ums, 'avg_value_difference':value_diff})
    return dl_b
