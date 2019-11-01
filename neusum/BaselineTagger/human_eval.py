#!/usr/bin/env python
# -*- coding: utf-8 -*-
def sample_id(dir, sample=20):
    files = os.listdir(dir)
    random.shuffle(files)
    files = files[:sample]
    return files


import os, json

from nltk.tokenize.moses import MosesDetokenizer

detokenizer = MosesDetokenizer()

flatten = lambda l: [item for sublist in l for item in sublist]

import random


def check_avg_len(dir):
    files = os.listdir(dir)
    random.shuffle(files)
    files = files[:500]
    l_info = []
    for f in files:
        with open(os.path.join(dir, f), 'r') as fd:
            lines = fd.read().splitlines()
        lines = [x.split(" ") for x in lines]
        lines = flatten(lines)
        l_info.append(len(lines))
    print("len: {}".format(len(l_info)))
    return sum(l_info) / len(l_info)


import string
# import unicodecsv as csv
import csv


def random_drop_str(inp_str, ratio=0.05):
    toks = inp_str.split(" ")
    trimmed_toks = []
    for t in toks:
        if t in string.punctuation:
            trimmed_toks.append(t)
        elif random.random() > ratio:
            trimmed_toks.append(t)
    trimmed_toks_str = " ".join(trimmed_toks)
    return trimmed_toks_str


if __name__ == '__main__':
    root = "/backup3/jcxu/exComp/xu-durrett-output/"
    dir_sel_sents = 'sel-sents'
    dir_our_compression = 'xu-compressions'
    dir_reference = 'ref'
    dir_lead3 = 'lead3'
    dir_offshelf_compress_sel = 'lstm-comp-sel-sents'
    dir_offshelf_compress_sel_more_retain = 'lstm-comp-sel-sents-ret'
    dir_offshelf_compress_sel_more_del = 'lstm-comp-sel-sents-del'

    dir_to_compare = dir_offshelf_compress_sel_more_del
    # dir_offshelf_compress_lead3 = 'lstm-comp-lead3'
    # l_sel = check_avg_len(os.path.join(root, dir_sel_sents))
    # print(l_sel)
    # l_compression = check_avg_len(os.path.join(root, dir_our_compression))
    # print(l_compression)
    # l_standard = check_avg_len(os.path.join(root, dir_to_compare))
    # print(l_standard)
    random.seed(2019)
    files = sample_id(os.path.join(root, dir_to_compare), 500)

    rand_lens, ours_lens, ooff_lens, ext_lens = [], [], [], []
    to_write = []
    ratio = 0.24
    for f in files:
        lines_sel = open(os.path.join(root, dir_sel_sents, f), 'r').read().splitlines()
        lines_ours = open(os.path.join(root, dir_our_compression, f), 'r').read().splitlines()
        lines_offshel = open(os.path.join(root, dir_to_compare, f), 'r').read().splitlines()
        linrs_lead3 = open(os.path.join(root, dir_lead3, f), 'r').read().splitlines()

        lines_rand_dropout = [random_drop_str(x, ratio) for x in lines_sel]
        # lines_rand_dropout = lines_sel

        rand_len = sum([len(x.split(" ")) for x in lines_rand_dropout])
        rand_lens.append(rand_len)

        ours_len = sum([len(x.split(" ")) for x in lines_ours])
        ours_lens.append(ours_len)

        ooff_len = sum([len(x.split(" ")) for x in lines_offshel])
        ooff_lens.append(ooff_len)


        sel_len = sum([len(x.split(" ")) for x in lines_sel])
        ext_lens.append(sel_len)
        for _rand, _ours, _offshel,_lead3 in zip(lines_rand_dropout, lines_ours, lines_offshel,linrs_lead3):

            ours = _ours.split(" ")

            all_words = _ours.split()
            for i in range(len(all_words)):
                if all_words[i].lower() in ["a", "an"]:
                    if all_words[i + 1][0].lower() in "aeiou":
                        all_words[i] = all_words[i][0] + "n"
                    else:
                        all_words[i] = all_words[i][0]
            _ours = " ".join(all_words)
            # print(_lead3)
            # print(detokenizer.detokenize(_lead3.split(" "), return_str=True))
            print(detokenizer.detokenize(_ours.split(" "), return_str=True))

            set_ours = set(_ours.split(" "))
            set_offshelf = set(_offshel.split(" "))
            if len(set_ours.intersection(set_offshelf)) / len(set_ours.union(set_offshelf)) > 0.96:
                # print(_ours)
                # print(_offshel)
                continue
            text_fields = [_rand, _ours, _offshel]
            type_fields = ['rand', 'our', 'off']
            order = list(range(3))
            random.shuffle(order)
            _tmp = []
            for o in order:
                # _tmp.append(detokenizer.detokenize(text_fields[o].split(" "), return_str=True))
                _tmp.append(text_fields[o])
                _tmp.append(type_fields[o])
            to_write.append(_tmp)
    print(sum(rand_lens) / len(rand_lens))
    print(sum(ours_lens) / len(ours_lens))
    print(sum(ooff_lens) / len(ooff_lens))
    print(sum(ext_lens)/ len(ext_lens))
    fields = ['sent_1', 'label_1', 'sent_2', 'label_2', 'sent_3', 'label_3']
    f1 = ['p1_{}'.format(x) for x in fields]
    f2 = ['p2_{}'.format(x) for x in fields]
    f3 = ['p3_{}'.format(x) for x in fields]
    f4 = ['p4_{}'.format(x) for x in fields]
    random.shuffle(to_write)
    with open('turkers_cnn.csv', 'w', newline='', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        # spamwriter.writerow(fields)
        spamwriter.writerow(f1 + f2 + f3 + f4)
        # to_write = to_write[:100]
        cnt = 0
        for idx in range(int(len(to_write) / 4) - 1):
            cnt += 1
            spamwriter.writerow(to_write[idx] + to_write[idx + 1] + to_write[idx + 2] + to_write[idx + 3])
            if cnt >= 10:
                break
        # for l in to_write:
        #     spamwriter.writerow(l)

# extractive 128-57
# off 112 - 46
# ours 104 - 21
# random 185 - 47
