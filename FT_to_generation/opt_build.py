from pt_build_optfile import *

with open('rec.log','a') as rec:
    for i in range(5):
        #train_opt, dev_opt, optv, simv = get_('similarity_lib',i,False,10000,False,None)
        train_opt, dev_opt, optv, simv = multi('similarity_lib',10,i+1)
        with open('train_{0}.tsv'.format(1+i),'w') as f:
            f.write('src\ttgt\n')
            for r in train_opt:
                f.write('{0}\t{1}\n'.format(r[0], r[1]))
        with open('dev_{0}.tsv'.format(1+i),'w') as f:
            f.write('src\ttgt\n')
            for r in dev_opt:
                f.write('{0}\t{1}\n'.format(r[0], r[1]))
        name = str(i+1)
        rec.write('scale:{0}\n'.format(name))
        optv = np.array(optv)
        simv = np.array(simv)
        np.save('scale{0}_optv.npy'.format(name), optv)
        np.save('scale{1}_simv.npy'.format(name), simv)
