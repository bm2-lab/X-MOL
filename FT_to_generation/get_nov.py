from pt_metrics import get_novelty, get_valid, read_dev_output, eva_f, eva
import pandas as pd
"""
task_list = ('qed_f','qed_n','logp_n','mw_n')
train_file = {'qed_f':'qed_f','qed_n':'qed_n','logp_n':'logp','mw_n':'mw'}
gen_file = {'qed_f':'qed_f_hot_v4-100e','qed_n':'qed_f_hot_v4-normalize','logp_n':'logp_hot_v4-normalize','mw_n':'mw_hot_v4-normalize'}
def get_n(task):
    nov = {}
    with open('package/task_data/'+train_file[task]+'/train.tsv', 'r') as f:
        ref = []
        for l in f:
            ref.append(l.strip().split('\t')[-1])
    ref = ref[1:]
    task_ns = []
    for i in range(100):
        smis = read_dev_output('output_store/{0}/dev_epoch{1}'.format(gen_file[task],i))
        vsmis = get_valid(smis, return_valid=True)
        ns = get_novelty(vsmis, ref, return_novelty=True)
        task_ns.append(len(ns)/len(smis))
    nov[task] = task_ns

    nov = pd.DataFrame(nov)
    nov.to_csv('novelty_{0}.csv'.format(task))
"""

if __name__ == '__main__':
    #k = eva_f(100, 'output_store/zinc250k_dl', 'package/task_data/zinc250k_dl/dev.tsv', novelty=True, sc_func='qed', ref_v=None):
    k = eva(100, 'output_store/zinc250k_dl/', 'package/task_data/zinc250k_dl/dev.tsv')
    k.to_csv('zinc250k_dl.csv')
