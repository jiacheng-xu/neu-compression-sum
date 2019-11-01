import os
import json

# data_dir = "/home/cc/data/cnn/read_ready-grammarTrue-miniFalse-maxsent20-beam5/dev.txt"
data_dir = "/home/cc/data/cnn/read_ready-grammarTrue-miniFalse-maxsent50-beam8/dev.txt"

with open(data_dir, 'r') as fd:
    lines = fd.read().splitlines()
print("Analysis!")
from collections import Counter, OrderedDict

# Length of sent and num of deletion (absolute num and portion)     IN LEAD3 and IN ALL
# distribution of del units (occurance, length) and (score)their performance vs baseline   IN LEAD3
# distribution of del units and their performance vs baseline   in all
# on average the % of baseline
# avg len of abs
# avg len of lead 2
# avg len of lead 3

avg_word_num_in_a_sent_in_doc = []
avg_len_abs, avg_len_lead2, avg_len_lead3, avg_len_lead1 = [], [], [], []
avg_rank_of_baseline_lead3, avg_rank_of_baseline_all = [], []
num_del_lead3, num_del_all = [], []
del_score_over_baseline_lead3, del_score_over_baseline_all = {}, {}
del_unit_occur_lead3, del_unit_occur_all = Counter(), Counter()

sent_len_and_num_deletion_portion_lead3, sent_len_and_num_deletion_portion_all = {}, {}  # sent: 1-30, all span, span better than baseline

del_compression_length_lead3, del_compression_length_all = [], []


def sent_len_and_num_deletion_portion(zip_inp, dic: dict):
    for sent, doc_list in zip_inp:
        sent_len = len(doc_list)
        if sent_len not in dic:
            dic[sent_len] = []
        dic[sent_len].append(len(sent['del_span']))
    return dic


def sum_len(inp):
    return sum([len(a) for a in inp])


def avg_len(inp):
    return avg([len(a) for a in inp])


def rank_of_baseline(inp_list):
    tmp = []
    for inp in inp_list:
        l = inp['del_span']
        total = len(l)
        for idx, it in enumerate(l):
            if it['node'] == 'BASELINE':
                tmp.append(float(idx / total))
                break
    return tmp


def num_del(inp_lists):
    tmp = []
    for inp in inp_lists:
        l = inp['del_span']
        total = len(l)
        tmp.append(total - 1)
    return tmp


def del_score_over_baseline(inp_lists, dic: dict):
    for inp in inp_lists:
        sp = inp['del_span']
        BASELINE_rouge = [s["rouge"] for s in sp if s["node"] == 'BASELINE']
        assert len(BASELINE_rouge) == 1
        BASELINE_rouge = BASELINE_rouge[0]
        if BASELINE_rouge < 0.01:
            continue
        for s in sp:
            if s['node'] not in dic:
                dic[s['node']] = []
            cur = dic[s['node']]
            cur.append(s['rouge'] / BASELINE_rouge)
            dic[s['node']] = cur
    return dic


def avg(x):
    return round(sum(x) / len(x), 2)


def count_occ(inp_lists, cnter: Counter):
    for inp in inp_lists:
        sp = inp['del_span']
        for idx, it in enumerate(sp):
            cnter[it['node']] += 1
    return cnter


for l in lines:
    d = json.loads(l)
    abs_list = d["abs_list"]
    avg_len_abs.append(sum([len(a) for a in abs_list]))

    doc_list = d["doc_list"][1:]  # omit sos
    doc_list_lead3 = doc_list[0:3]
    avg_len_lead1.append(sum_len(doc_list[0:1]))
    avg_len_lead2.append(sum_len(doc_list[0:2]))
    avg_len_lead3.append(sum_len(doc_list_lead3))
    avg_word_num_in_a_sent_in_doc.append(avg_len(doc_list))

    sent = d['sentences'][1:]
    sent_lead3 = sent[0:3]

    avg_rank_of_baseline_lead3 += rank_of_baseline(sent_lead3)
    avg_rank_of_baseline_all += rank_of_baseline(sent)

    num_del_lead3 += num_del(sent_lead3)
    num_del_all += num_del(sent)

    del_unit_occur_lead3 = count_occ(sent_lead3, del_unit_occur_lead3)
    del_unit_occur_all = count_occ(sent, del_unit_occur_all)

    del_score_over_baseline_lead3 = del_score_over_baseline(sent_lead3, del_score_over_baseline_lead3)
    del_score_over_baseline_all = del_score_over_baseline(sent, del_score_over_baseline_all)

    sent_len_and_num_deletion_portion_lead3 = sent_len_and_num_deletion_portion(zip(sent_lead3, doc_list_lead3),
                                                                                sent_len_and_num_deletion_portion_lead3)
    sent_len_and_num_deletion_portion_all = sent_len_and_num_deletion_portion(zip(sent, doc_list),
                                                                              sent_len_and_num_deletion_portion_all)

print("data_dir: {}".format(data_dir))
print("Average num of words in abstract: {}".format(avg(avg_len_abs)))
print("avg_word_num_in_a_sent_in_doc: {}".format(avg(avg_word_num_in_a_sent_in_doc)))
print("Average num of words in first 1 sent in doc: {}".format(avg(avg_len_lead1)))
print("Average num of words in first 2 sents in doc: {}".format(avg(avg_len_lead2)))
print("Average num of words in first 3 sents in doc: {}".format(avg(avg_len_lead3)))

print("BASELINE=no deletion. avg_rank(percentage)_of_baseline_lead3: {0:.2f}\tavg_rank_of_baseline_all: {1:.2f}".format(
    avg(avg_rank_of_baseline_lead3),
    avg(avg_rank_of_baseline_all)))
print(
    "num of del units in lead3: {0:.2f}\tnum of del units in all: {1:.2f}".format(avg(num_del_lead3), avg(num_del_all)))

del_unit_occur_lead3 = dict(del_unit_occur_lead3)
for k in iter(del_unit_occur_lead3):
    if k == "BASELINE":
        continue
    del_unit_occur_lead3[k] = round(float(del_unit_occur_lead3[k]) / del_unit_occur_lead3["BASELINE"], 2)

del_unit_occur_all = dict(del_unit_occur_all)
for k in iter(del_unit_occur_all):
    if k == "BASELINE":
        continue
    del_unit_occur_all[k] = round(float(del_unit_occur_all[k]) / del_unit_occur_all["BASELINE"], 2)

print("Distribution of del_unit_ Occurrence _lead3: {}".format(del_unit_occur_lead3))
print("Distribution of del_unit_ Occurrence _all: {}".format(del_unit_occur_all))


def avg_dict(inp):
    for k in iter(inp):
        inp[k] = avg(inp[k])
    return inp


for k in iter(del_score_over_baseline_lead3):
    del_score_over_baseline_lead3[k] = avg(del_score_over_baseline_lead3[k])

for k in iter(del_score_over_baseline_all):
    del_score_over_baseline_all[k] = avg(del_score_over_baseline_all[k])

print("del units' score over baseline_lead3: {}\tdel_score_over_baseline_all: {}".format(del_score_over_baseline_lead3,
                                                                                         del_score_over_baseline_all))
sent_len_and_num_deletion_portion_lead3 = avg_dict(sent_len_and_num_deletion_portion_lead3)
sent_len_and_num_deletion_portion_all = avg_dict(sent_len_and_num_deletion_portion_all)

print('-' * 20)
print("In a sent, sent len (num of words) vs. num of possible deletion units"
      )
for key in sorted(sent_len_and_num_deletion_portion_lead3.keys()):
    if int(key) < 5 or int(key) > 30:
        continue
    print("#word num:%s\t%s" % (key, sent_len_and_num_deletion_portion_lead3[key]))

print('-' * 20)
print("sent_len_and_num_deletion_portion_all")
for key in sorted(sent_len_and_num_deletion_portion_all.keys()):
    if int(key) < 5 or int(key) > 30:
        continue
    print("#word num:%s\t%s" % (key, sent_len_and_num_deletion_portion_all[key]))
