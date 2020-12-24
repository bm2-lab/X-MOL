from random import shuffle

ratio = (0.7, 0.3)
cv = 10

def cvs(data, cv):
    n = len(data)
    fold_n = round(n/cv)
    cv_t = []
    for i in range(cv-1):
        sidx = i*fold_n
        eidx = (i+1)*fold_n
        cv_t.append(data[sidx:eidx])
    cv_t.append(data[fold_n*(cv-1):])
    return cv_t

def merge(cv_data, cvn):
    assert cvn < len(cv_data)
    trains = []
    for i in range(len(cv_data)):
        if i != cvn:
            trains += cv_data[i]
    devs = cv_data[cvn]
    return trains, devs

with open('cv_rec.txt','r') as f:
    cvr = f.readlines()
cvr = int(cvr[0])

header = 'aryl_halide\tadditive\tbase\tligand\tlabel\n'

if cvr == 0:
    with open('DataA.tsv.ordered', 'r') as f:
        k = []
        for l in f:
            k.append(l)

    k = k[1:]
    num_ = len(k)

    idx = list(range(num_))
    shuffle(idx)

    ratio = [i/sum(ratio) for i in ratio]
    train_ = [k[i] for i in idx[:int(ratio[0]*num_)]]
    test_ = [k[i] for i in idx[int(ratio[0]*num_):]]

    shuffle(train_)
    with open('train-store','w') as f:
        for r in train_:
            f.write(r)

    with open('test.tsv','w') as f:
        f.write(header)
        for r in test_:
            f.write(r)

    cv_data = cvs(train_, cv)
    traindata, devdata = merge(cv_data, cvr)
    with open('train.tsv','w') as f:
        f.write(header)
        for r in traindata:
            f.write(r)

    with open('dev.tsv','w') as f:
        f.write(header)
        for r in devdata:
            f.write(r)

else:
    with open('train-store','r') as f:
        train_ = []
        for l in f:
            train_.append(l)
    
    cv_data = cvs(train_, cv)
    traindata, devdata = merge(cv_data, cvr)
    with open('train.tsv','w') as f:
        f.write(header)
        for r in traindata:
            f.write(r)

    with open('dev.tsv','w') as f:
        f.write(header)
        for r in devdata:
            f.write(r)

cvr += 1
if cvr == cv:
    cvr = 0

with open('cv_rec.txt','w') as f:
    f.write(str(cvr))

