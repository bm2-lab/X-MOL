import numpy as np
import pandas as pd
import os

class extract_info_multirepeats():
    def __init__(self, file):
        self.file = file
        self.result = 0
        self.extract_result()
        self.extract_metrics()


    def extract_result(self):
        store = {}
        repeats = 0
        cache_dev = []
        cache_test = []
        with open(self.file, 'r') as f:
            for l in f:
                if l.startswith('Num train examples') and (len(cache_dev)>0):
                    #if repeats != -1:
                    store[repeats] = [cache_dev, cache_test]
                    repeats += 1
                    cache_dev = []
                    cache_test = []
                else:
                    if l.startswith('[dev evaluation]'):
                        cache_dev.append(l.strip()[17:])
                    if l.startswith('[test evaluation]'):
                        cache_test.append(l.strip()[18:])
        if len(cache_dev) != 0:
            store[repeats] = [cache_dev, cache_test]
        self.result = store
    
    def extract_metrics(self):
        if not self.result:
            self.extract_result()
        temple = self.result[0][0][0]
        temple = temple[:-2].split(', ')
        temple = [i.split(': ')[0] for i in temple]
        temple = {j:i for i,j in enumerate(temple)}
        self.metrics = temple

    def extract_test(self, met):
        assert met in self.metrics
        test_ = {i:self.result[i][1] for i in self.result}
        self.test_eva = {}
        for rep in test_:
            rep_eva = []
            for rec in test_[rep]:
                rec_s = rec[:-2].split(', ')[self.metrics[met]]
                rec_s = float(rec_s.split(': ')[1])
                rep_eva.append(rec_s)
            self.test_eva[rep] = rep_eva

    def extract_dev(self, met):
        assert met in self.metrics
        dev_ = {i:self.result[i][0] for i in self.result}
        self.dev_eva = {}
        for rep in dev_:
            rep_eva = []
            for rec in dev_[rep]:
                rec_s = rec[:-2].split(', ')[self.metrics[met]]
                rec_s = float(rec_s.split(': ')[1])
                rep_eva.append(rec_s)
            self.dev_eva[rep] = rep_eva    

    def extract_best(self, met, focus='dev'):
        sq = 0
        if met == 'ave rmse':
            met = 'ave loss'
            sq = 1
        assert met in self.metrics
        assert focus in ('dev', 'test')
        self.extract_test(met)
        self.extract_dev(met)
        if met in ('ave loss', 'ave mae', 'ave rmse'):
            if focus == 'dev':
                best_idx = {i:np.argmin(self.dev_eva[i]) for i in self.dev_eva}
            else:
                best_idx = {i:np.argmin(self.test_eva[i]) for i in self.test_eva}
        else:
            if focus == 'dev':
                best_idx = {i:np.argmax(self.dev_eva[i]) for i in self.dev_eva}
            else:
                best_idx = {i:np.argmax(self.test_eva[i]) for i in self.test_eva}
        best = {i:[self.test_eva[i][best_idx[i]], '{0}/{1}'.format(best_idx[i], len(self.test_eva[i]))] for i in best_idx}
        if sq:
            best = {i:[np.sqrt(best[i][0]),best[i][1]] for i in best}
        return best

