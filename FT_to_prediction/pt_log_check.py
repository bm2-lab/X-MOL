import sys
import numpy as np
import pandas as pd

def extract_info_pt(in_file, out_file, args, retrun_dict=False, ma=False, buffer=1):
    if 'step' not in args:
        args = [i for i in args]
        args = ['step'] + args
    with open(in_file, 'r') as f:
        log = []
        for l in f:
            if l.startswith('epoch: ') and 'progress' in l:
                log.append(l.strip())
    info = {a:[] for a in args}
    for a in args:
        if a != 'step':
            info[a] = [float(l.split('{0}: '.format(a))[1].split(',')[0]) for l in log]
        else:
            info[a] = [int(l.split('step: ')[1].split(',')[0]) for l in log]
    if ma:
        argx = []
        for i in args:
            if i == 'step':
                argx.append('step')
            else:
                argx.append(i)
                argx.append(i+'_ma')
                info[i+'_ma'] = moving_avg(info[i], buffer)
        args = argx
    if retrun_dict:
        return info
    else:
        with open(out_file, 'w') as f:
            f.write(','.join(args)+'\n')
            for idx in range(len(log)):
                write_args = [str(info[a][idx]) for a in args]
                f.write(','.join(write_args)+'\n')

def extract_info_ft(in_file, out_file, earg, args, retrun_dict=False, ma=False, buffer=1):

    assert earg in ('train', 'valid', 'dev')
    flag = {'train':'epoch: ', 'valid':'[test evaluation]', 'dev':'[dev evaluation]'}

    if earg == 'train':
        if 'step' not in args:
            args = [i for i in args]
            args = ['step'] + args
    with open(in_file, 'r') as f:
        log = []
        for l in f:
            if l.startswith(flag[earg]) and 'ave' in l:
                log.append(l.strip())
    info = {a:[] for a in args}
    for a in args:
        if a != 'step':
            info[a] = [float(l.split('{0}: '.format(a))[1].split(',')[0]) for l in log]
        else:
            info[a] = [int(l.split('step: ')[1].split(',')[0]) for l in log]
    if ma:
        argx = []
        for i in args:
            if i == 'step':
                argx.append('step')
            else:
                argx.append(i)
                argx.append(i+'_ma')
                info[i+'_ma'] = moving_avg(info[i], buffer)
        args = argx
    if retrun_dict:
        return info
    else:
        with open(out_file, 'w') as f:
            f.write(','.join(args)+'\n')
            for idx in range(len(log)):
                write_args = [str(info[a][idx]) for a in args]
                f.write(','.join(write_args)+'\n')

def moving_avg(data, buffer):
    assert buffer >= 2
    assert buffer < len(data)/2
    buffer_up = buffer // 2
    buffer_down = buffer - 1 - buffer_up
    data_ma = []
    for idx in range(len(data)):
        if idx <= buffer_up:
            avg_data = data[:idx + buffer_down]
        elif (len(data)-idx-1) < buffer_down:
            avg_data = data[idx - buffer_up:]
        else:
            avg_data = data[idx-buffer_up:idx+buffer_down+1]
        avg = np.mean(avg_data)
        data_ma.append(avg)
    return data_ma

def extract_ft_info_final(in_file):
    flag_t = 0
    flag_v = 0
    final_test = []
    final_val = []
    with open(in_file, 'r') as f:
        for l in f:
            if flag_t == 1:
                final_test.append(l.rstrip('\n')[18:])
                flag_t = 0
            if flag_v == 1:
                final_val.append(l.rstrip('\n')[17:])
                flag_v = 0

            if l.startswith('Final test result:'):
                flag_t = 1
            if l.startswith('Final validation result:'):
                flag_v = 1
    test_args = [i.split(': ')[0] for i in final_test[0].split(', ')]
    val_args = [i.split(': ')[0] for i in final_val[0].split(', ')]
    test = {i:[] for i in test_args}
    val = {i:[] for i in val_args}
    for t in final_test:
        k = [i.split(': ')[1] for i in t.split(', ')]
        for idx, a in enumerate(k):
            if ' ' in a:
                test[test_args[idx]].append(float(a.split(' ')[0]))
            else:
                test[test_args[idx]].append(float(a))
    for t in final_val:
        k = [i.split(': ')[1] for i in t.split(', ')]
        for idx, a in enumerate(k):
            if ' ' in a:
                val[val_args[idx]].append(float(a.split(' ')[0]))
            else:
                val[val_args[idx]].append(float(a))

    test = append_info(pd.DataFrame(test))
    val = append_info(pd.DataFrame(val))
    return test, val
    
def append_ave(df):
    k = df.columns
    n = len(df[k[0]])
    idx = list(range(n)) + ['ave']
    ave = {i:np.mean(df[i]) for i in k}
    df = df.append([ave])
    df.index = idx
    return df

def append_info(df):
    k = df.columns
    focus = k[1]
    focus_value = df[focus]
    if focus.split('_')[-1] in ('loss', 'mse', 'mae'):
        best_focus_value = min(focus_value)
    else:
        best_focus_value = max(focus_value)
    best_row = df[df[focus] == best_focus_value]
    n = len(df[k[0]])
    idx = list(range(1,n+1)) + ['*'] + ['best'] +['ave'] + ['std']
    best = {i:list(best_row[i])[0] for i in k}
    ave = {i:np.mean(df[i]) for i in k}
    std = df.std()
    std = {i:std[i] for i in k}
    sep = {i:'--------' for i in k}
    df = df.append([sep])
    df = df.append([best])
    df = df.append([ave])
    df = df.append([std])
    df.index = idx
    return df

def find_extremum_in_dict(dict, ex='big', idx='value'):
    assert idx in ('key', 'value')
    assert ex in ('big','small')

    def rep(a, b, ex):
        if ex == 'big':
            c = max(a,b)
        else:
            c = min(a,b)
        if c == a:
            rep_flag = False
        else:
            rep_flag = True
        
        return c, rep_flag

    if ex == 'big':
        target_v = 0
    else:
        target_v = 999
    tmp = 0
    for k in dict:
        if idx == 'key':
            target_v, rep_flag = rep(target_v, k, ex)
            if rep_flag:
                tmp = k
        else:
            target_v, rep_flag = rep(target_v, dict[k], ex)
            if rep_flag:
                tmp = k

    return tmp, dict[tmp]

def extract_ft_info_best(in_file, focus_0='dev', focus_1='auc', cut_unfinished=True):
    # get the best performance during the whole training process
    # in_file: input log file
    # focus_0: test or validation? which one to be the target when determine the best performance
    # focus_1: auc, acc, loss? which one to be the target when determine the best performance
    # cut_unfinished: if drop the repeat which is not finished when killing the job
    assert focus_0 in ('test', 'val', 'dev', 'step')
    assert focus_1 in ('auc', 'acc', 'loss','r2','mae')

    if focus_1 in ('loss', 'mae'):
        ex = 'small'
    else:
        ex = 'big'
    focus_1_loc = {'auc':2, 'acc':1, 'loss':0, 'r2':1, 'mae':2}[focus_1]
    if focus_0 in ('val', 'step'):
        focus_0 = 'dev'
    if focus_0 == 'step':
        focus_0 = 'dev'
        v = 'key'
    else:
        v = 'value'
    rec_title = '[{0} evaluation] '.format(focus_0)
    

    with open(in_file, 'r') as f:
        d = []
        for l in f:
            d.append(l.strip())

    mem = []
    features = []

    for l in d:
        if l.startswith('batch_size:'):
            mem.append({'focus':{}, 'test':{}})
        elif l.startswith('epoch:') and 'step' in l:
            step = int(l.split(',')[2].split(': ')[-1])
        elif l.startswith(rec_title):
            info = l.split(']')[1][1:].split(', ')[focus_1_loc].split(': ')[-1]
            mem[-1]['focus'][step] = float(info)
        if l.startswith('[test evaluation] '):
            info = l.split(']')[1][1:].rstrip(' s').split(', ')
            if features == []:
                features = [i.split(': ')[0].split(' ')[-1] for i in info[:3]]
            info = [float(i.split(': ')[-1]) for i in info[:3]]
            mem[-1]['test'][step] = info
    
    standard_len = len(mem[0]['test'])
    if cut_unfinished:
        mem = [i for i in mem if len(i['test'])==standard_len]
    else:
        for repeat in range(len(mem)):
            if len(mem[repeat]['test']) == standard_len:
                mem[repeat]['finished'] = int(1)
            else:
                mem[repeat]['finished'] = int(0)
            mem[repeat]['trained_steps'] = max(mem[repeat]['test'].keys())

    df_d = {'best_step':[], 'index@{0}_{1}'.format(focus_0, focus_1):[], 'test_loss':[], 'test_'+features[1]:[], 'test_'+features[2]:[]}
    if not cut_unfinished:
        df_d['finished'] = []
        df_d['trained_steps'] = []
    for repeat in mem:
        best_step, best_focus_0_value = find_extremum_in_dict(repeat['focus'], ex, v)
        if best_step in repeat['test']:
            best_test_loss, best_test_acc, best_test_auc  = repeat['test'][best_step]
            df_d['best_step'].append(best_step)
            df_d['index@{0}_{1}'.format(focus_0, focus_1)].append(best_focus_0_value)
            df_d['test_loss'].append(best_test_loss)
            df_d['test_'+features[1]].append(best_test_acc)
            df_d['test_'+features[2]].append(best_test_auc)
            if not cut_unfinished:
                df_d['finished'].append(repeat['finished'])
                df_d['trained_steps'].append(repeat['trained_steps'])
        else:
            print("the last testing performance should be the best but the job was killed when the last validation is finished but testing isn't")
    
    df = append_info(pd.DataFrame(df_d))
    return df

if __name__ == '__main__':
    type = sys.argv[1]    

    assert type in ('pt','ft','pre','fine','pre_train','fine_tune','pretrain','finetune')

    if type in ('pt','pre','pre_train','pretrain'):
        print('type extraction args, splited by " ", valid: loss ppl-c next_sent_acc')
        if arg == '\n':
            arg = ('loss','ppl-c','next_sent_acc')
        else:
            arg = sys.stdin.readline().strip().split(' ')

        extract_info_pt(in_file='./log/job.log.0',
        out_file='./log/job_rec.txt', 
        args=arg, 
        retrun_dict=False,
        ma=True, 
        buffer=20)

    else:
        print('type focus_0: [val/test/dev/step]')
        f0 = sys.stdin.readline().strip()
        print('type focus_1:[auc/acc/loss/r2/mae]')
        f1 = sys.stdin.readline().strip()
        test = extract_ft_info_best('./log/job.log.0', focus_0=f0, focus_1=f1, cut_unfinished=False)
        print(test)
    
