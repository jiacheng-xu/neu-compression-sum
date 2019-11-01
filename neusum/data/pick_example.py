import os
import pickle
from neusum.evaluation.rough_rouge import get_rouge_est_str_2gram
from typing import List
if __name__ == '__main__':
    file = "0.419-0.179-0.379-dmTrue1.03True-10397-cp_0.5"
    path = "/scratch/cluster/jcxu/exComp"
    f = os.path.join(path,file)
    with open(f,'rb') as fd:
        x = pickle.load(fd)
    pred = x['pred']
    ref = x['ref']
    ori = x['ori']
    for p, r, o in zip(pred,ref,ori):
        r = "\n".join(r[0])
        for oo,pp in zip(o,p):

            lead3_val = get_rouge_est_str_2gram(r, oo)
            if lead3_val == 0:
                continue
            model_val = get_rouge_est_str_2gram(r, pp)
            if model_val / lead3_val > 1.2 and lead3_val>0.2:
                print("P: {}\tBaseline: {}".format(model_val, lead3_val))
                print("Ori: {}\nPred: {}".format(o, p))
                print(r)
        # o = "\n".join(o)
        # p = "\n".join(p)
