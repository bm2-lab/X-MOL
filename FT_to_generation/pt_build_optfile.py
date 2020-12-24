from rdkit import Chem
from rdkit import DataStructs
from random import shuffle
import numpy as np
import time
from rdkit.Chem import Descriptors
from tqdm import tqdm
from multiprocessing import Process
import os
import subprocess

def get_(similarity_lib, scale, to_file=False, n_dev=10000, show=True, ids=None):
    if type(similarity_lib) == str:
        with open(similarity_lib,'r') as f:
            libm = []
            libs = []
            for l in f:
                m, sm = l.strip().split(':')
                libm.append(m)
                libs.append(sm.split(','))
    else:
        libm = [i[0] for i in similarity_lib]
        libs = [i[1] for i in similarity_lib]
    libmq = []
    print('cal ref QEDs')
    with tqdm(total=len(libm)) as ref_pbar:
        for i in libm:
            libmq.append(Descriptors.qed(Chem.MolFromSmiles(i)))
            if show:
                ref_pbar.update()
    libsq = []
    libss = []
    print('cal candidate QEDs')
    with tqdm(total=len(libs)) as cdd_pbar:
        for lidx,i in enumerate(libs):
            temp_ = []
            k = 0
            tp = libm[lidx]
            while len(temp_)<scale and k<len(i):
                if i[k] != tp:
                    temp_.append(i[k])
                k += 1
            libss.append(temp_)
            libsq.append([Descriptors.qed(Chem.MolFromSmiles(j)) for j in temp_])
            if show:
                cdd_pbar.update()
    opt = []
    optv = []
    print('build pair')
    with tqdm(total=len(libm)) as bd_pbar:
        for midx in range(len(libm)):
            diff = [abs(libmq[midx]-libsq[midx][cidx]) for cidx in range(len(libsq[midx]))]
            sel = np.argmax(diff)
            optv.append(max(diff))
            if libmq[midx]<libsq[midx][sel]:
                opt.append([libm[midx], libss[midx][sel]])
            else:
                opt.append([libss[midx][sel], libm[midx]])
            if show:
                bd_pbar.update()
    print('remove repeats')
    opt = ['&'.join(i) for i in opt]
    opt = list(set(opt))
    opt = [i.split('&') for i in opt]
    
    if to_file:
        with open(to_file,'w') as f:
            for r in opt:
                f.write(','.join([str(i) for i in r])+'\n')
    simv = []
    print('cal pair similarity')
    with tqdm(total=len(libm)) as sv_pbar:
        for r in opt:
            simv.append(DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(Chem.MolFromSmiles(r[0])),Chem.RDKFingerprint(Chem.MolFromSmiles(r[1]))))
            if show:
                sv_pbar.update()
    optv = np.mean(optv)
    simv = np.mean(simv)
    print('split data')
    idx= list(range(len(opt)))
    shuffle(idx)
    train_idx = idx[:-10000]
    dev_idx = idx[-10000:]
    train_opt = [opt[i] for i in train_idx]
    dev_opt = [opt[i] for i in dev_idx]

    if ids == None:
        return train_opt, dev_opt, optv, simv
    else:
        with open('train_subprocess_{0}.tsv'.format(ids), 'w') as f:
            for r in train_opt:
                f.write('{0}\t{1}\n'.format(r[0], r[1]))
        with open('dev_subprocess_{0}.tsv'.format(ids), 'w') as f:
            for r in dev_opt:
                f.write('{0}\t{1}\n'.format(r[0], r[1]))
        with open('rec_subprocess_{0}'.format(ids),'w') as f:
            f.write('{0}\n'.format(optv))
            f.write('{0}\n'.format(simv))


def get_s(similarity_lib, scale, to_file=False, n_dev=10000, show=True, ids=None):
    if type(similarity_lib) == str:
        with open(similarity_lib,'r') as f:
            libm = []
            libs = []
            for l in f:
                m, sm = l.strip().split(':')
                libm.append(m)
                libs.append(sm.split(','))
    else:
        libm = [i[0] for i in similarity_lib]
        libs = [i[1] for i in similarity_lib]
    libmq = []
    libmfp = []
    print('cal ref QEDs')
    with tqdm(total=len(libm)) as ref_pbar:
        for i in libm:
            rmol = Chem.MolFromSmiles(i)
            libmfp.append(Chem.RDKFingerprint(rmol))
            libmq.append(Descriptors.qed(rmol))
            if show:
                ref_pbar.update()
    opt = []
    optv = []
    simv = []
    print('build pair')
    with tqdm(total=len(libm)) as bd_pbar:
        for midx in range(len(libm)):
            rfp = libmfp[midx]
            rq = libmq[midx]
            max_d = 0
            csmi = 'C1CCCCC1'
            sim_v = 0
            for cdd in libs[midx]:
                cmol = Chem.MolFromSmiles(cdd)
                cfp = Chem.RDKFingerprint(cmol)
                sim = DataStructs.TanimotoSimilarity(rfp, cfp)
                if sim<scale[1] and sim>scale[0]:
                    cq = Descriptors.qed(cmol)
                    diff = cq - rq
                    if diff > max_d:
                        csmi = cdd
                        max_d = diff
                        sim_v = sim
            if max_d > 0:
                opt.append([libm[midx], csmi])
                optv.append(max_d)
                simv.append(sim_v)
            if show:
                bd_pbar.update()
    if to_file:
        with open(to_file,'w') as f:
            for r in opt:
                f.write(','.join([str(i) for i in r])+'\n')

    print('split data')
    idx= list(range(len(opt)))
    shuffle(idx)
    if len(opt)<n_dev:
        n = len(str(len(opt)))-1
        kn = '1'+'0'*n
        kn = int(int(kn)/10)
    else:
        kn = n_dev
    train_idx = idx[:-kn]
    dev_idx = idx[-kn:]
    train_opt = [opt[i] for i in train_idx]
    dev_opt = [opt[i] for i in dev_idx]

    if ids == None:
        return train_opt, dev_opt, optv, simv
    else:
        with open('train_subprocess_{0}.tsv'.format(ids), 'w') as f:
            for r in train_opt:
                f.write('{0}\t{1}\n'.format(r[0], r[1]))
        with open('dev_subprocess_{0}.tsv'.format(ids), 'w') as f:
            for r in dev_opt:
                f.write('{0}\t{1}\n'.format(r[0], r[1]))
        optv = np.array(optv)
        simv = np.array(simv)
        np.save('simv_subprocess_{0}.npy'.format(ids), simv)
        np.save('optv_subprocess_{0}.npy'.format(ids), optv)


def multi(similarity_lib, n_jobs, scale, n_dev=10000):
    lib = []
    with open(similarity_lib,'r') as f:
        for l in f:
            m, sm = l.strip().split(':')
            lib.append([m, sm.split(',')])
    n_jobs = max(n_jobs, 1)
    recn_per_job = round(len(lib)/n_jobs)
    rec_lists = [lib[i*recn_per_job:(i+1)*recn_per_job] for i in range(n_jobs-1)]
    rec_lists.append(lib[(n_jobs-1)*recn_per_job:])
    n_dev = int(n_dev/n_jobs)
    
    sub_process = []
    for sp in range(n_jobs):
        sub_process.append(Process(target=get_s, args=(rec_lists[sp], scale, False, n_dev, False, sp)))
    for sp in sub_process:
        sp.start()
    for sp in sub_process:
        sp.join()
    # merge files and remove temporary files
    train_opt = []
    dev_opt = []
    simv = []
    optv = []

    for spf in range(n_jobs):
        with open('train_subprocess_{0}.tsv'.format(spf)) as f:
            train_sp = f.readlines()
            train_sp = [i.strip().split('\t') for i in train_sp]
        train_opt += train_sp
        with open('dev_subprocess_{0}.tsv'.format(spf)) as f:
            dev_sp = f.readlines()
            dev_sp = [i.strip().split('\t') for i in dev_sp]
        dev_opt += dev_sp
        simv_sp = np.load('simv_subprocess_{0}.npy'.format(spf))
        simv += list(simv_sp)
        optv_sp = np.load('optv_subprocess_{0}.npy'.format(spf))
        optv += list(optv_sp)
        subprocess.call('rm train_subprocess_{0}.tsv'.format(spf), shell=True)
        subprocess.call('rm dev_subprocess_{0}.tsv'.format(spf), shell=True)
        subprocess.call('rm simv_subprocess_{0}.npy'.format(spf), shell=True)
        subprocess.call('rm optv_subprocess_{0}.npy'.format(spf), shell=True)

    return train_opt, dev_opt, optv, simv
