# data_path = "/backup2/jcxu/data/dm/dm-gramTrue-miniFalse-maxsent30-beam8"
# data_path = "/backup2/jcxu/data/cnndm/"
import os
import sys
import os
import hashlib
import struct
import subprocess
import collections
import shutil
from typing import List
import multiprocessing
from neusum.evaluation.rouge_with_pythonrouge import RougeStrEvaluation
import pickle

from neusum.service.basic_service import easy_post_processing


def merge_cnn_dm():
    cnn = "/scratch/cluster/jcxu/exComp/0.327,0.122,0.290-cnnTrue1.0-1True3-1093-cp_0.5"
    dm = "/scratch/cluster/jcxu/exComp/0.427,0.192,0.388-dmTrue1.0-1True3-10397-cp_0.7"
    total_pred = []
    total_ref = []
    f = open(cnn, "rb")
    cnn_dict = pickle.load(f)
    f.close()
    fine_cnn_pd = []
    for x in cnn_dict["pred"]:
        fine_x = [easy_post_processing(s) for s in x]
        fine_cnn_pd.append(fine_x)
    total_pred += fine_cnn_pd
    # total_pred += cnn_dict["pred"]
    total_ref += cnn_dict["ref"]

    f = open(dm, "rb")
    dm_dict = pickle.load(f)
    f.close()
    fine_dm_pd = []
    for x in dm_dict["pred"]:
        fine_x = [easy_post_processing(s) for s in x]
        fine_dm_pd.append(fine_x)
    total_pred += fine_dm_pd
    # cnnpd = [easy_post_processing(x) for x in dm_dict["pred"]]
    # total_pred += cnnpd
    # total_pred += dm_dict["pred"]
    # total_pred += dm_dict["pred"]
    total_ref += dm_dict["ref"]
    rouge_metrics = RougeStrEvaluation(name='mine')
    for p, r in zip(total_pred, total_ref):
        rouge_metrics(pred=p, ref=r)
    rouge_metrics.get_metric(True, note='test')


def my_lead3():
    data_path = "/scratch/cluster/jcxu/data/2merge-nyt"
    print(data_path)
    lead = 3
    if 'nyt' in data_path:
        lead = 5
    files = [x for x in os.listdir(data_path) if x.startswith("test.pkl")]
    rouge_metrics_sent = RougeStrEvaluation(name='mine')
    import pickle
    print(lead)
    for idx, file in enumerate(files):
        print(idx)
        f = open(os.path.join(data_path, file), 'rb')
        print("reading {}".format(file))
        data = pickle.load(f)
        for instance_fields in data:
            meta = instance_fields['metadata']
            doc_list = meta['doc_list'][:lead]
            abs_list = meta['abs_list']

            doc_list = [" ".join(x) for x in doc_list]
            abs_list = [" ".join(x) for x in abs_list]
            rouge_metrics_sent(pred=doc_list, ref=[abs_list])
    rouge_metrics_sent.get_metric(True, note='test')


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    out = pool.map(hashhex, url_list)
    pool.close()
    pool.join()
    return out


def get_refresh_metric():
    gold_path = "/backup3/jcxu/data/gold-cnn-dailymail-test-orgcase"
    # 0034b7c223e24477e046cf3ee085dd006be38b27.gold
    model_path = "/backup3/jcxu/data/cnn-dailymail-ensemble-model11-model7"
    # 0034b7c223e24477e046cf3ee085dd006be38b27.model
    full_dataname = "cnn"

    test_urls = '/backup3/jcxu/data/cnn-dailymail/url_lists/{}_wayback_test_urls.txt'.format(full_dataname)

    with open(test_urls, 'r', encoding='utf-8') as fd:
        lines = fd.read().splitlines()
        url_names = get_url_hashes(lines)
        print("len of urls {}".format(len(url_names)))
    print(url_names[0])
    rouge_metrics_sent = RougeStrEvaluation(name='refresh')
    for url in url_names:
        # gold
        try:
            with open(os.path.join(gold_path, url + '.gold'), 'r') as fd:
                abs = fd.read().splitlines()
            with open(os.path.join(model_path, url + '.model'), 'r') as fd:
                pred = fd.read().splitlines()
            rouge_metrics_sent(pred=pred, ref=[abs])
        except IOError:
            print(url)
    full_dataname = "dailymail"

    test_urls = '/backup3/jcxu/data/cnn-dailymail/url_lists/{}_wayback_test_urls.txt'.format(full_dataname)

    with open(test_urls, 'r', encoding='utf-8') as fd:
        lines = fd.read().splitlines()
        url_names = get_url_hashes(lines)
        print("len of urls {}".format(len(url_names)))
    print(url_names[0])
    for url in url_names:
        # gold
        try:
            with open(os.path.join(gold_path, url + '.gold'), 'r') as fd:
                abs = fd.read().splitlines()
            with open(os.path.join(model_path, url + '.model'), 'r') as fd:
                pred = fd.read().splitlines()
            rouge_metrics_sent(pred=pred, ref=[abs])
        except IOError:
            print(url)
    rouge_metrics_sent.get_metric(True)


import numpy as np


def get_xxz_metric():
    article = "/backup3/jcxu/data/xxz-latent/test.article"
    pred_file = "/backup3/jcxu/data/xxz-latent/1.test.out"

    with open(pred_file, 'r') as fd:
        lines = fd.read().splitlines()
        l = int(len(lines) / 2)

    with open(article, 'r') as fd:
        article = fd.read().splitlines()
    for i in range(l):
        dist = lines[2 * i + 1]
        probs = []
        dists = dist.split("\t")[1].split("|")
        for tuple in dists:
            tuple_list = tuple.strip()
            tuple_list = tuple_list.split(" ")
            tuple_list = [float(x) for x in tuple_list]
            probs.append(1 - tuple_list[2])
        art = article[i]
        art_sents = art.split("<S_SEP>")

        bag = []
        idxs = list(np.argsort(probs))
        for location_in_artile, rank in enumerate(idxs):
            if rank <= 2:
                bag.append([art_sents[location_in_artile]])


# get_refresh_metric()
# get_xxz_metric()
def get_qyz():
    path = "/scratch/cluster/jcxu/data/cnndm_compar/qyz-output"
    data = "cnn"
    ref = "qyz_{}_sum.txt"
    pred = "qyz_{}.txt"
    rouge_metrics = RougeStrEvaluation(name='neusum')
    os.path.join(path, pred_path)

    def read_prediction(pred_path):
        with open(pred_path, 'r') as fd:
            lines = fd.read().splitlines()
        lines = [x.split("\t")[1] for x in lines]
        return lines

    ##SENT##
    def read_sum(sum_path):
        with open(sum_path, 'r') as fd:
            lines = fd.read().splitlines()
        lines = [x.replace("##SENT##"," ") for x in lines]
        return lines


if __name__ == '__main__':
    # my_lead3()
    merge_cnn_dm()
