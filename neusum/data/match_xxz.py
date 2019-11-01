from neusum.data.extract_data_from_raw import hashhex, get_url_hashes, get_art_abs, read_text_file


def extract_keyword_xxz():
    pass


import re

from neusum.evaluation.rough_rouge import get_rouge_est_str_2gram
import os


def set_original_files(full_dataname, lowercase: bool = False):
    # work on the original files
    # For CNN DM, filter test set of CNN and DM.
    # return a dict with uid and feature words.
    test_urls = '/scratch/cluster/jcxu/cnn-dailymail/url_lists/{}_wayback_test_urls.txt'.format(full_dataname)
    original_file_dir = "/scratch/cluster/jcxu/data/original-cnndm/{}/stories/".format(full_dataname)
    # article, abstract = get_art_abs(os.path.join(dir_to_read, name))

    d = {}
    with open(test_urls, 'r', encoding='utf-8') as fd:
        lines = fd.read().splitlines()
        url_names = get_url_hashes(lines)

    for url in url_names:
        story_file = os.path.join(original_file_dir, url + '.story')
        article, abstract = get_art_abs(story_file)
        feat = article.split(" ")[:50]
        feat_sent = " ".join(feat)
        out = re.findall(r"[\w']+", feat_sent)
        if lowercase:
            feat = [x.lowercase() for x in out]
        feat = set(feat)
        d['{}-{}'.format(full_dataname, url)] = feat
    return d

def match_qyz(match_dict):
    split_tok = "##SENT##"
    path = "/backup3/jcxu/data/xxz-latent/test.article"
    with open(path, 'r') as fd:
        lines = fd.read().splitlines()
    line_num = len(lines)
    output_list = ["" for _ in range(line_num)]

    feat_lines = [[] for _ in range(line_num)]
    meta_sents = []
    for idx, l in enumerate(lines):
        sents = l.split(split_tok)
        meta_sents.append(sents)
        l = l.replace(split_tok, "")
        toks = l.split(" ")[:52]
        toks = set([x for x in toks if x != ""])

        for key, val in match_dict.items():
            joint = toks.union(val)
            rat = len(joint) / len(toks)
            if rat > 0.85:
                # print("Match. ")
                output_list[idx] = key
                del match_dict[key]
                break
    print("remain")
    print(match_dict)
    print("match")
    print(output_list)
    return output_list, meta_sents

def match_see():
    pass


def match_xxz(match_dict):
    path = "/backup3/jcxu/data/xxz-latent/test.article"
    with open(path, 'r') as fd:
        lines = fd.read().splitlines()
    line_num = len(lines)
    output_list = ["" for _ in range(line_num)]

    feat_lines = [[] for _ in range(line_num)]
    meta_sents = []
    for idx, l in enumerate(lines):
        sents = l.split("<S_SEP>")
        meta_sents.append(sents)
        l = l.replace("<S_SEP>", "")
        toks = l.split(" ")[:52]
        toks = set([x for x in toks if x != ""])

        for key, val in match_dict.items():
            joint = toks.union(val)
            rat = len(joint) / len(toks)
            if rat > 0.85:
                # print("Match. ")
                output_list[idx] = key
                del match_dict[key]
                break
    print("remain")
    print(match_dict)
    print("match")
    print(output_list)
    return output_list, meta_sents


import numpy as np


def read_xxz_prediction():
    path = "/backup3/jcxu/data/xxz-latent/1.test.out"
    with open(path, 'r') as fd:
        lines = fd.read().splitlines()
    l = int(len(lines) / 2)
    meta_results = []
    for idx in range(l):
        preds = []
        content = lines[idx * 2 + 1]
        right = content.split("\t")[1]
        decisions = right.split("|")
        decisions = [x.strip() for x in decisions]
        for dec in decisions:
            _, _, pred = dec.split(" ")
            pred = float(pred)
            preds.append(pred)

        result = []
        if len(preds) < 3:
            result = list(range(len(preds)))
            meta_results.append(result)
            continue

        cnt = 0
        while cnt < 3:
            out = np.argmax(preds)
            result.append(out)
            preds[out] = -1
            cnt += 1
        meta_results.append(result)
    return meta_results


def wt_xxz_output_to_disk(dir, fname_list, full_docs, decision):
    for output, doc, dec in zip(fname_list, full_docs, decision):
        sent_picked = [doc[idx] for idx in dec]
        wt_content = "\n".join(sent_picked)
        with open(os.path.join(dir, output + '.txt'), 'w') as fd:
            fd.write(wt_content)


if __name__ == '__main__':

    # to reverse engineer xxz
    d_cnn = set_original_files("cnn")
    d_dm = set_original_files("dailymail")
    d = {**d_cnn, **d_dm}
    print("Finish examing raw data!")
    """
    outputlist,meta_sents = match_xxz(d)    # output_list: ['cnn-u3hf2ufh23','dailymail-jhdu237rh4',....]
    choice_result = read_xxz_prediction()
    # wt xxz output to disk

    wt_path = "/backup3/jcxu/data/xxz-latent/xxz-output"
    wt_xxz_output_to_disk(wt_path, outputlist,meta_sents,choice_result)

    """