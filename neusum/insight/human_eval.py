from typing import List
import os, pickle
from neusum.service.basic_service import meta_str_surgery, easy_post_processing


def iterator(bag):
    pass


from random import shuffle
import random


def dropword(inp_str: str):
    inp_list = inp_str.split(" ")
    indc = random.sample(range(0, len(inp_list)), int(len(inp_list) / 10))
    inp_list = [x for idx, x in enumerate(inp_list) if idx not in indc]
    return " ".join(inp_list)


def replace_lrbrrb(inp_str: str):
    inp_str = inp_str.replace("-LRB-", '(')
    inp_str = inp_str.replace("-RRB-", ')')
    return inp_str


import csv


def assign_task(ext_bag, ext_dp_bag, model_bag,see_bag):
    cells = []
    num_of_unit = 4
    for ext, extdp, model,see in zip(ext_bag, ext_dp_bag, model_bag,see_bag):
        _tmp = [None for _ in range(2 * num_of_unit)]
        lis = [ext, extdp, model,see]
        nam = ['ext', 'extdp', 'model','see']
        idx = list(range(num_of_unit))
        shuffle(idx)
        for m in range(num_of_unit):
            _tmp[int(2 * m)] = lis[idx[m]]
            _tmp[int(2 * m + 1)] = nam[idx[m]]
        # _tmp[2] = lis[idx[1]]
        # _tmp[3] = nam[idx[1]]
        # _tmp[4] = lis[idx[2]]
        # _tmp[5] = nam[idx[2]]
        cells.append(_tmp)
    return cells


from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

d = TreebankWordDetokenizer()


def detok(inp_str):
    inp_list = inp_str.split(" ")
    return d.detokenize(inp_list)


def read_abigail_output(path) -> List[str]:
    files = os.listdir(path)
    bag = []
    for fname in files:
        with open(os.path.join(path, fname), 'r') as fd:
            lines = fd.read().splitlines()
            lines = [detok(l) for l in lines]
            bag += lines
    return bag


def rm_head_cnn(inp_str: str):
    if inp_str.find("CNN -RRB-", 0, 50) > 0:
        where = inp_str.find("CNN -RRB-", 0, 50)
        inp_str = inp_str[where + 9:]
    return inp_str


def fix_vowel(inp_str):
    lis = inp_str.split(" ")
    v = "aeio"
    len_of_seq = len(lis)
    for idx in range(len_of_seq - 1):
        word = lis[idx]
        if word == "a" or word == "an":
            nex_w: str = lis[idx + 1]
            if nex_w[0] in v:
                lis[idx] = "an"
            else:
                lis[idx] = "a"
    return " ".join(lis)


if __name__ == '__main__':
    path = "/scratch/cluster/jcxu/exComp"
    file = "0.325-0.120-0.289-cnnTrue1.0-1True-1093-cp_0.6"
    see_output = "/scratch/cluster/jcxu/data/cnndm_compar/pointgencov/cnn"
    ext_bag, model_bag, ext_dp_bag, see_bag = [], [], [], []
    see_bag = read_abigail_output(see_output)
    with open(os.path.join(path, file), 'rb') as fd:
        x = pickle.load(fd)
        pred = x['pred']
        ori = x['ori']
        cnt = 0
        for pre, o in zip(pred, ori):
            shuffle(pre)
            shuffle(o)
            p = [meta_str_surgery(easy_post_processing(replace_lrbrrb(fix_vowel(x)))).lower() for x in pre]
            o = [meta_str_surgery(easy_post_processing(replace_lrbrrb(rm_head_cnn(x)))).lower() for x in o]
            o_drop = [dropword(x) for x in o]

            o = [detok(x) for x in o]
            o_drop = [detok(x) for x in o_drop]
            p = [detok(x) for x in p]
            # print("\n".join(p))
            # print("-" * 5)
            ext_bag += o
            ext_dp_bag += o_drop
            model_bag += p
            cnt += 1
            if cnt > 250:
                # print(ext_bag)
                # print(ext_dp_bag)
                # for visp, viso, viss in zip(model_bag, ext_bag, see_bag):
                #     print(visp)
                #     print(viso)
                #     print(viss)
                #     print("--------")
                # exit()
                shuffle(ext_dp_bag)
                shuffle(model_bag)
                shuffle(ext_bag)
                cells = assign_task(ext_bag, ext_dp_bag, model_bag,see_bag)
                cells_len = len(cells)
                cstep = int(cells_len/2)
                for idx in range(cstep):
                    x = "\t".join(cells[2*idx]+cells[2*idx+1])
                    print(x)
                    if idx >250:
                        exit()
                with open('data_v0.csv', 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ',
                                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
                    for c in cells:
                        print("\t".join(c))
                        spamwriter.writerow(c)

                # print(model_bag)
                exit()
