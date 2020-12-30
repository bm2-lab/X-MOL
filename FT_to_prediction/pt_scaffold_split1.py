from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import random
import subprocess
import os
import sys

def scaffold_split(smis, prop, randomize=0):
    """
    scaffold split, different subset contains different scaffolds\n
    split the given SMILESs according to the given proportion\n
    *param smis: input SMILESs\n
    *param prop: the proportion of each subset\n
    *param randomize: the probability of doing randomize while distributing data,\n
                      default None, No randomize;\n
                      0 for default probability, the max probability in prop\n
    return : the splitted index of each SMILES
    """
    if sum(prop) != 1:
        t = sum(prop)
        prop = [i/t for i in prop]

    if randomize == 0:
        randomize = max(prop)

    dataset_size = len(smis)
    subsets_size = [round(dataset_size*i) for i in prop[:-1]]
    subsets_size.append(dataset_size-sum(subsets_size))
    remain_space = subsets_size[:]
    splited_idx = [[] for _ in prop]
    scaffold_lib = {}
    for idx, s in enumerate(smis):
        scf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))
        if scf in scaffold_lib:
            scaffold_lib[scf].append(idx)
        else:
            scaffold_lib[scf] = [idx]
    scaffold_count = {i:len(scaffold_lib[i]) for i in scaffold_lib}
    scaffold_count_rev = {}
    for scf in scaffold_count:
        if scaffold_count[scf] in scaffold_count_rev:
            scaffold_count_rev[scaffold_count[scf]].append(scf)
        else:
            scaffold_count_rev[scaffold_count[scf]] = [scf]
    scaffold_num = list(scaffold_count_rev.keys())
    scaffold_num = sorted(scaffold_num, reverse=True)
    for scf_n in scaffold_num:
        for scf in scaffold_count_rev[scf_n]:
            if randomize:
                marker = random.random()
                if marker <= randomize:
                    add_loc = np.argmax(remain_space)
                else:
                    add_loc = random.choice([i for i in range(len(prop)) if remain_space[i]>scf_n-2])
            else:
                add_loc = random.choice([i for i in range(len(prop)) if remain_space[i]>scf_n-2])
            ss = scaffold_lib[scf]
            splited_idx[add_loc] += ss
            remain_space[add_loc] -= scf_n
        
    return splited_idx


def random_split(smis, prop):
    """
    random split\n
    split the given SMILESs according to the given proportion\n
    *param smis: input SMILESs\n
    *param prop: the proportion of each subset\n
    return : the splitted index of each SMILES
    """

    if sum(prop) != 1:
        t = sum(prop)
        prop = [i/t for i in prop]

    dataset_size = len(smis)
    subsets_size = [round(dataset_size*i) for i in prop[:-1]]
    subsets_size.append(dataset_size-sum(subsets_size))
    idx = list(range(dataset_size))
    k = random.choice((1,2,3))
    for t in range(k):
        random.shuffle(idx)
    marker = 0
    splited_idx = []
    for s in subsets_size:
        splited_idx.append(idx[marker : marker+s])
        marker += s
    return splited_idx


def amp_smis(smis, labels, amp=-1, rem_can=False):
    """
    SMILES amplification
    #param rem_can: if remain canonical SMILES
    """
    smis_c = [[smis[i],labels[i]] for i in range(len(smis))]
    if amp == 0:
        return smis_c
    smis_a = []
    mols = [Chem.MolFromSmiles(i) for i in smis]
    for idx, mol in enumerate(mols):
        rep = 0
        if amp == -1:
            goal_num = 2 + int(len(smis[idx])/2)
        else:
            goal_num = amp
        rsmis = []
        while (len(rsmis) < goal_num) and (rep <100):
            rs = Chem.MolToSmiles(mol, doRandom=True)
            if rs not in rsmis:
                rsmis.append(rs)
            rep += 1
        for rs in rsmis:
            smis_a.append([rs, labels[idx]])
    if rem_can:
        smis_a = smis_a + smis_c
    random.shuffle(smis_a)
    return smis_a




def sep_file(dir, prop=(0.8,0.1,0.1), sfn=('train','dev','test'), split_func='random', amp=False, ampn=(-1,1,1)):
    """
    split data in file, and write the splited into subset file\n
    *param dir: the dir of file of data to be splited\n
    *param prop: the proportion of each subset\n
    *param sfn: subset files' name
    *param split_func: split data in random way or scaffold way
    """
    assert len(prop) == len(sfn)
    assert split_func in ('random', 'scaffold')
    if dir.endswith('/'):
        file = dir+'all_data'
        sfn = [dir+i for i in sfn]
    else:
        file = dir+'/all_data'
        sfn = [dir+'/'+i for i in sfn]
    with open(file, 'r') as f:
        d = []
        for l in f:
            d.append(l.strip().split(','))
    for n, fn in enumerate(sfn):
        if not fn.endswith('.tsv'):
            sfn[n] = sfn[n] + '.tsv'
    smis = [i[0] for i in d]
    labels = [i[1] for i in d]
    if split_func == 'random':
        splited_idx = random_split(smis, prop)
    else:
        splited_idx = scaffold_split(smis, prop)
    for fidx, fn in enumerate(sfn):
        didx = splited_idx[fidx]
        with open(fn, 'w') as sf:
            sf.write('text_a\tlabel\n')
            if not amp:
                for idx in didx:
                    sf.write('{0}\t{1}\n'.format(smis[idx], labels[idx]))
            else:
                sp_smis = [smis[idx] for idx in didx]
                sp_labels = [labels[idx] for idx in didx]
                smis_a = amp_smis(sp_smis, sp_labels, ampn[fidx])
                for rec in smis_a:
                    sf.write('{0}\t{1}\n'.format(rec[0],rec[1]))

def sep_file_ex(dir, prop=(0.8,0.1,0.1), sfn=('train','dev','test'), split_func='random', amp=False, ampn=(-1,0,0), save_file=False):
    """
    split data in file, and write the splited into subset file\n
    *param dir: the dir of file of data to be splited\n
    *param prop: the proportion of each subset\n
    *param sfn: subset files' name
    *param split_func: split data in random way or scaffold way
    *param amp: if do amplification to the subset
    *param ampn: the multiplying power of amplification
                 0 for not amplification
                 n for n times amplification
                 -1 for 2+int(len(SMILES)/2) times amplification 
    *param save_file: if save the splitted file
    """

    k = os.listdir(dir)
    k = [i for i in k if '_' in i]
    tag = int(len(k)/3)
    random.seed(tag)

    assert len(prop) == len(sfn)
    assert split_func in ('random', 'scaffold')
    if dir.endswith('/'):
        file = dir+'all_data'
        sfn = [dir+i for i in sfn]
    else:
        file = dir+'/all_data'
        sfn = [dir+'/'+i for i in sfn]
    with open(file, 'r') as f:
        d = []
        for l in f:
            d.append(l.strip().split(','))
    for n, fn in enumerate(sfn):
        if not fn.endswith('.tsv'):
            sfn[n] = sfn[n] + '.tsv'
    smis = [i[0] for i in d]
    labels = [i[1] for i in d]
    if split_func == 'random':
        splited_idx = random_split(smis, prop)
    else:
        splited_idx = scaffold_split(smis, prop)
    for fidx, fn in enumerate(sfn):
        didx = splited_idx[fidx]
        with open(fn, 'w') as sf:
            sf.write('text_a\tlabel\n')
            if not amp:
                for idx in didx:
                    sf.write('{0}\t{1}\n'.format(smis[idx], labels[idx]))
            else:
                sp_smis = [smis[idx] for idx in didx]
                sp_labels = [labels[idx] for idx in didx]
                smis_a = amp_smis(sp_smis, sp_labels, ampn[fidx], rem_can=True)
                for rec in smis_a:
                    sf.write('{0}\t{1}\n'.format(rec[0],rec[1]))
    if save_file:
        for fn in sfn:
            tfn = fn[:-4] + '_{0}.tsv'.format(tag)
            subprocess.call('cp {0} {1}'.format(fn, tfn), shell=True)


if __name__ == '__main__':
    # ampn: -1: n_len/2, 0: no amp, n: n
    path=sys.argv[1]
    if path == 'delaney':
        amp_r = 4
    elif path == 'sampl':
        amp_r = -1
    else:
        amp_r = 0
    if path not in ('hiv', 'bace_c', 'bbbp'):
        sep_file_ex('package/task_data/{0}'.format(path), split_func='random', amp=False, ampn=(amp_r,0,0), save_file=True)
    else:
        sep_file_ex('package/task_data/bbbp', split_func='scaffold', amp=False, ampn=(amp_r,0,0), save_file=True)
