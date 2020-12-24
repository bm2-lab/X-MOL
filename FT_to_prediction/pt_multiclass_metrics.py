from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import os

def to_one_hot(feature, class_num):
    one_hot = np.zeros((len(feature), class_num))
    for idx, i in enumerate(feature):
        one_hot[idx][feature[idx]] = 1
    return one_hot

with open('../package/task_data/DEEPDDI/dev.tsv','r') as f:
    label = []
    h = 1
    for l in f:
        if h == 1:
            h = 0
        else:
            label.append(int(l.strip().split('\t')[-1]))

#label_oh = to_one_hot(label, 86)


validation_probs = os.listdir()
validation_probs = [i for i in validation_probs if i.endswith('.npy')]
validation_probs = sorted(validation_probs)

multiclass_metrics = {'mean_acc':[], 'pr_macro':[], 'rc_macro':[], 'f1_macro':[], 'pr_micro':[], 'rc_micro':[], 'f1_micro':[]}
for file in validation_probs:
    vp = np.load(file)
    vpi = []
    for sample in vp:
        vpi.append(np.argmax(sample))
    #vpi_oh = to_one_hot(vpi, 86)
    n = 0
    for i in range(len(vpi)):
        if vpi[i] == label[i]:
            n += 1
    acc = n/len(vpi)
    multiclass_metrics['mean_acc'].append(acc)
    multiclass_metrics['pr_macro'].append(precision_score(label, vpi, average='macro'))
    multiclass_metrics['pr_micro'].append(precision_score(label, vpi, average='micro'))
    multiclass_metrics['rc_macro'].append(recall_score(label, vpi, average='macro'))
    multiclass_metrics['rc_micro'].append(recall_score(label, vpi, average='micro'))
    multiclass_metrics['f1_macro'].append(f1_score(label, vpi, average='macro'))
    multiclass_metrics['f1_micro'].append(f1_score(label, vpi, average='micro'))

multiclass_metrics = pd.DataFrame(multiclass_metrics)
multiclass_metrics.to_csv('multiclass_metrics.csv')
