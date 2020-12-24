from sklearn.metrics import roc_auc_score, accuracy_score
import os
import numpy as np

def ana(temp_data, step=-1):
    store = []
    for f in temp_data:
        if f[:-4].split('_')[-1] == '0':
            store.append([])
        store[-1].append(f)
    if step != -1:
        store = [i[:step] for i in store]
    return store

def cal_auc(label_file, prob_file, dir_pre=None):
    if dir_pre:
        label_file = dir_pre + label_file
        prob_file = dir_pre + prob_file
    label = np.load(label_file)
    prob = np.load(prob_file)
    auc = roc_auc_score(label, prob[:,1])
    return auc

def cal_acc(label_file, prob_file, dir_pre=None):
    if dir_pre:
        label_file = dir_pre + label_file
        prob_file = dir_pre + prob_file
    label = np.load(label_file)
    prob = np.load(prob_file)[:,1]
    prob[prob<0.5] = 0
    prob[prob>0.5] = 1
    acc = accuracy_score(label, prob)
    return acc

def arg(k, rank, d='max'):
    ks = sorted(k)
    if d == 'max':
        ks = ks[::-1]
        v = ks[rank]
        idx = k.index(v)
    return idx

ckpt = input('target_dir:')
if ckpt == '':
    ckpt = 'checkpoints/'
#ckpt = 'output/bace_c/out_0/'
focus = 'dev'
get = 'test'
rank = 0 # select the step idx which ranking rank in focus term, when rank=0 ,will return the best performance in focus
step = -1 # find the final step idx in the first step steps, if step=-1, will find step idx in step[:-1], as well as the whole steps

k = os.listdir(ckpt)
k = [i for i in k if 'final' not in i]

labels = [i for i in k if i.startswith('label')]
probs = [i for i in k if i.startswith('prob')]

dev_labels = [i for i in labels if 'dev' in i]
dev_probs = [i for i in probs if 'dev' in i]
test_labels = [i for i in labels if 'test' in i]
test_probs = [i for i in probs if 'test' in i]

dev_labels = ana(sorted(dev_labels), step)
dev_probs = ana(sorted(dev_probs), step)
test_labels = ana(sorted(test_labels), step)
test_probs = ana(sorted(test_probs), step)

rep = len(dev_probs)

auc_dev = []
auc_test = []
acc_dev = []
acc_test = []

for ridx in range(rep):
    auc_dev.append([])
    auc_test.append([])
    acc_dev.append([])
    acc_test.append([])
    for didx in range(len(test_probs[ridx])):
        dev_label = dev_labels[ridx][didx]
        dev_prob = dev_probs[ridx][didx]
        test_label = test_labels[ridx][didx]
        test_prob = test_probs[ridx][didx]

        dev_auc = cal_auc(dev_label, dev_prob, dir_pre=ckpt)
        test_auc = cal_auc(test_label, test_prob, dir_pre=ckpt)
        dev_acc = cal_acc(dev_label, dev_prob, dir_pre=ckpt)
        test_acc = cal_acc(test_label, test_prob, dir_pre=ckpt)

        auc_dev[-1].append(dev_auc)
        auc_test[-1].append(test_auc)
        acc_dev[-1].append(dev_acc)
        acc_test[-1].append(test_acc)

best_auc = {}
best_acc = {}
for ridx in range(rep):
    if focus == 'dev':
        best_auc_idx = arg(auc_dev[ridx], rank)
        best_acc_idx = arg(acc_dev[ridx], rank)
    else:
        best_auc_idx = arg(auc_test[ridx], rank)
        best_acc_idx = arg(acc_test[ridx], rank)
    
    if get == 'test':
        best_test_auc = auc_test[ridx][best_auc_idx]
        best_test_acc = acc_test[ridx][best_acc_idx]
    else:
        best_test_auc = auc_dev[ridx][best_auc_idx]
        best_test_acc = acc_dev[ridx][best_acc_idx]

    best_auc[ridx] = [best_test_auc, '{0}/{1}'.format(best_auc_idx, len(auc_test[ridx]))]
    best_acc[ridx] = [best_test_acc, '{0}/{1}'.format(best_acc_idx, len(acc_test[ridx]))]

mean_best_auc = np.mean([best_auc[i][0] for i in best_auc])
mean_best_acc = np.mean([best_acc[i][0] for i in best_acc])
