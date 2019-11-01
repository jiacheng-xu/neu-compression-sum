# merge part result and get CNNDM
import allennlp

from typing import List
import os
import pickle


def data_feeder(path, flist: List[str]):
    # flist = ['test.pkl.cnn.00', 'test.pkl.dm.02', 'test.pkl.dm.01', 'test.pkl.dm.00']
    for f in flist:
        with open(os.path.join(path, f), 'rb') as fd:
            data = pickle.load(fd)
            for instance_fields in data:
                yield instance_fields


def show_ratio(iter):
    cnt = 0
    d = {}
    import math
    for inst in iter:
        ratio = inst['comp_rouge_ratio'].field_list
        for idx, x in enumerate(ratio):
            if idx >= 3:
                break
            x = x.array
            for y in x:
                if y > 0.001:
                    cat = math.floor(y * 20)
                    key = str(cat)
                    if key in d:
                        d[key] += 1
                    else:
                        d[key] = 1
                    cnt += 1
                if cnt > 10000:
                    for k, v in d.items():
                        print("{}\t{}".format(int(k) / 20, v))
                    exit()


from collections import Counter, OrderedDict


def inspect_every_node_type(name, lis: List):
    l = len(lis)
    txts = [x[0] for x in lis]  # [ [a,v,c], [asd,asd,asd]///

    average_txt_len = sum([len(x) for x in txts]) / len(txts)
    c = Counter()
    for t in txts:
        for e in t:
            c[e] += 1
        # print(c)
    populart = c.most_common(10)
    ratio = [x[2] for x in lis]
    pos = [x for x in ratio if x >= 1.00]
    pos_rate = len(pos) / len(ratio)
    return l, ("{}\t{}\t{}\t{}\t{}".format(name, l, average_txt_len, pos_rate, populart))


def inspect(name, lis):
    l = len(lis)
    ratio = [x[2] for x in lis]
    pos = [x for x in ratio if x >= 1.00]
    pos_rate = len(pos) / len(ratio)

    act = [x[1] for x in lis]
    act_pos = [x for x in act if x > 0.99]
    act_rate = len(act_pos) / len(act)

    txts = [x[3] for x in lis]
    average_txt_len = sum([len(x) for x in txts]) / len(txts)
    c = Counter()
    for t in txts:
        for e in t:
            c[e] += 1
        # print(c)
    populart = c.most_common(10)
    return l, ("{}\t{}\t{}\t{}\t{}\t{}".format(name, l, average_txt_len, pos_rate, populart, act_rate))


import json


def compression_real_insight(iter):
    f = "/scratch/cluster/jcxu/exComp/dmTrue1.0-1True3"
    with open(f, 'r') as fd:
        x = fd.read().splitlines()
    x = x[1400:1500]
    d = {}
    cnt = 0
    for s in x:
        y = json.loads(s)
        l = y[11]
        for unit in l:
            node_type = unit['type']
            leng = unit['len']
            active = unit['active']
            word = unit['word']
            ratio = unit['ratio']
            cnt += 1
            if node_type in d:
                d[node_type] = d[node_type] + [[leng, active, ratio, word]]
            else:
                d[node_type] = [[leng, active, ratio, word]]

    bag = {}
    print(cnt)
    for k, v in d.items():
        count, s = inspect(k, v)
        bag["{}".format(count)] = s
        # print(s)
        if count / cnt >= 0.04:
            print(s)


def compression_raw_insight(iter):
    cnt = 0
    d = {}
    for inst in iter:
        cp_meta = inst['comp_meta'].field_list
        doc_list = inst['metadata'].metadata['doc_list']
        start = 0
        for cpm, doc in zip(cp_meta, doc_list):
            # cpm one MetadataField
            # doc is a list of str
            start += 1
            if start >= 5:
                break
            for compressions in cpm:

                node_type, sel_idx, rouge, ratio = compressions
                if node_type == 'BASELINE' or ratio < 0.01:
                    continue
                cnt += 1
                txt = [w for idx, w in enumerate(doc) if idx in sel_idx]
                if node_type in d:
                    d[node_type] = d[node_type] + [[txt, rouge, ratio]]
                else:
                    d[node_type] = [[txt, rouge, ratio]]
        if cnt >= 10000:
            bag = {}
            for k, v in d.items():
                count, s = inspect_every_node_type(k, v)
                bag["{}".format(count)] = s
                if count / cnt >= 0.8:
                    print(s)

            exit()


def oracle_to_trim_oracle(iter, num):
    cnt = 0
    ori_bag, comp_bag = [], []
    for inst in iter:
        try:
            original = inst['_non_compression_sent_oracle']['{}'.format(num)]['best']['R1']
            compression = inst['_sent_oracle']["{}".format(num)]['best']['R1']
            ori_bag.append(original)
            comp_bag.append(compression)
            cnt += 1
            if cnt >= 1000:
                print(sum(ori_bag) / len(ori_bag))
                print(sum(comp_bag) / len(comp_bag))
                exit()
        except TypeError:
            continue


# insight about compression stat from raw data
def insight_compression_raw(data, path: str,
                            flist: List[str]):
    iter = data_feeder(path, flist)
    leadn = 3
    if data == 'nyt':
        leadn = 5

    # oracle_to_trim_oracle(iter, leadn)
    compression_raw_insight(iter)


def read_cp_decision():
    pass


if __name__ == '__main__':
    data = 'cnn'
    if data == 'nyt':
        path = "/scratch/cluster/jcxu/data/2merge-nyt"
    else:
        path = "/scratch/cluster/jcxu/data/2merge-cnndm"

    cnns = ['test.pkl.cnn.00']
    dms = ['test.pkl.dm.02', 'test.pkl.dm.01', 'test.pkl.dm.00']
    nyt = ['test.pkl.nyt.02', 'test.pkl.nyt.01', 'test.pkl.nyt.00', 'test.pkl.nyt.03']

    if data == 'nyt':
        feed = nyt
    elif data == 'cnn':
        feed = cnns
    elif data == 'dm':
        feed = dms
    else:
        raise NotImplementedError
    compression_real_insight(data)
    # insight_compression_raw(data, path, feed)
