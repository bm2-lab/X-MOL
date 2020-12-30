import sys
import pandas as pd
import numpy as np

def add_info(df):
    k = df.columns
    ks = [len(i) for i in k]
    ks = max(ks)
    focus = k[2]
    focus_value = df[focus]
    best_focus_value = max(focus_value)
    best_row = df[df[focus] == best_focus_value]
    n = len(df[k[0]])
    idx = list(range(1,n+1)) + ['*'] + ['best'] +['ave'] + ['std']
    best = {i:list(best_row[i])[0] for i in k}
    ave = {i:np.mean(df[i]) for i in k}
    std = df.std()
    std = {i:std[i] for i in k}
    sep = {i:'-'*(ks-1) for i in k}
    df = df.append([sep])
    df = df.append([best])
    df = df.append([ave])
    df = df.append([std])
    df.index = idx
    return df

print('type focus_0: [val/test/dev/step]')
f0 = sys.stdin.readline().strip()
print('file dir')
file_dir = sys.stdin.readline().strip()
file_dir += '/job.log.0'

with open(file_dir, 'r') as f:
    logs = []
    for l in f:
        logs.append(l.strip())

mem = []
if f0 == 'val':
    f0s = 'dev'
else:
    f0s = f0[:]
for l in logs:
    if l.startswith('batch_size:'):
        mem.append({'{0}'.format(f0):[], 'test':[]})
    elif l.startswith('[{0} evaluation] '.format(f0s)):
        mem[-1]['{0}'.format(f0)].append(float(l.split('] ')[1].split(', ')[0].split(': ')[1]))
    elif l.startswith('[test evaluation] '):
        mem[-1]['test'].append(float(l.split('] ')[1].split(', ')[0].split(': ')[1]))

lib = {'loc':[], 'best_test_auc':[], 'best_{0}_auc'.format(f0):[]}
for rep in mem:
    #for rec in range(len(rep['test'])):
    full_test_times = len(rep['test'])
    best_loc = np.argmax(rep['{0}'.format(f0)])
    best_foucs = rep['{0}'.format(f0)][best_loc]
    best_test = rep['test'][best_loc]
    #lib['loc'].append('{0}/{1}'.format(best_loc, len(rep['test'])))
    lib['loc'].append(best_loc)
    lib['best_test_auc'].append(best_test)
    lib['best_{0}_auc'.format(f0)].append(best_foucs)

lib = pd.DataFrame(lib)
lib = add_info(lib)
k = list(lib['loc'])
full_rep_times = len(mem)
for i in range(len(k)): 
    if i < full_rep_times or i in (full_rep_times+1, full_rep_times+2): 
        k[i] = '{0}/{1}'.format(k[i], full_test_times-1) 
lib['loc'] = k
print(lib)
        
        



