import logging
import os
import torch
import shutil
import numpy as np

np.set_printoptions(threshold=np.inf)
from neusum.evaluation.rouge_with_pythonrouge import RougeStrEvaluation
import multiprocessing

punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


def _para_get_metric(metric: RougeStrEvaluation, key, note):
    current_metrics = metric.get_metric(reset=True, note=note)
    current_best_cp_A = [x for x in current_metrics.keys() if x.endswith("_A")]
    assert len(current_best_cp_A) == 1
    current_best_cp_A = current_best_cp_A[0]
    cp_A_val = current_metrics[current_best_cp_A]
    return current_metrics, cp_A_val, metric, key


def _para_get_metric_reset_false(metric: RougeStrEvaluation, key, note):
    current_metrics = metric.get_metric(reset=False, note=note)
    current_best_cp_A = [x for x in current_metrics.keys() if x.endswith("_A")]
    assert len(current_best_cp_A) == 1
    current_best_cp_A = current_best_cp_A[0]
    cp_A_val = current_metrics[current_best_cp_A]
    return current_metrics, cp_A_val, metric, key


from collections import OrderedDict


def para_get_metric(dict_of_rouge: OrderedDict, reset, note=""):
    new_dict = {}
    dict_of_rouge_func = list(dict_of_rouge.values())  # this is only the copy!!!
    list_rouge_dict_key = list(dict_of_rouge.keys())
    l = len(dict_of_rouge_func)

    cnt = 5
    pool = multiprocessing.Pool(processes=cnt)
    if reset:
        out = pool.starmap(_para_get_metric, zip(dict_of_rouge_func, list_rouge_dict_key, [note] * l))
    else:
        out = pool.starmap(_para_get_metric_reset_false, zip(dict_of_rouge_func, list_rouge_dict_key, [note] * l))
    pool.close()
    pool.join()

    best_cp_A = 0
    for idx, it in enumerate(out):
        met, cp_a, met_write_to_dictofrouge, key = it
        dict_of_rouge[key] = met_write_to_dictofrouge
        best_cp_A = max(best_cp_A, cp_a)
        new_dict = {**new_dict, **met}
    new_dict["cp_A"] = best_cp_A
    return new_dict


def multilabel_margin_loss(inp, tgt):
    loss_func = torch.nn.MultiLabelMarginLoss(reduction='none')
    out = loss_func(inp, tgt)
    return out


def print_tensor(inp):
    ts = inp.cpu().data.numpy()
    print(ts)


def flip_first_two_dim(inp):
    if len(inp.size()) == 2:
        return inp.permute(1, 0).contiguous()
    elif len(inp.size()) == 3:
        return inp.permute(1, 0, 2).contiguous()


def clear_dir(_path):
    if os.path.isdir(_path):
        shutil.rmtree(_path)
    os.mkdir(_path)


def meta_str_surgery(inp_str: str):
    # if found . '' before the last 15 chars, change it to , ''
    # if found , '' in the last 15 chars, change it to . ''
    loc = 10
    inp_str = surgery_on_quotes(inp_str, ". ''", ", ''", beg=0, end=-loc)
    inp_str = surgery_on_quotes(inp_str, ", ''", ". ''", beg=-loc, end=None)
    return inp_str


def surgery_on_quotes(inp_str: str, match_str: str, replace_with: str, beg: int, end: int = None):
    """
    In region beg to end of inp_str, if find match_str, replace it with replace_with
    :param inp_str:
    :param match_str:
    :param replace_with:
    :param beg:
    :param end:
    :return:
    """

    if end == None:
        fd = inp_str.find(match_str, beg)
    else:
        fd = inp_str.find(match_str, beg, end)
    if fd == -1:
        return inp_str
    l = len(match_str)
    left = inp_str[:fd]
    right = inp_str[fd + l:]
    # print(left)
    # print(right)
    new_str = left + replace_with + right
    return new_str


def easy_post_processing(inp: str):
    """
    Given a string from the compression model, 1) remove first ,:;.?!
    2) remove redundant 3) upper case first char.
    :param inp:
    :return:
    """

    inp = inp.strip()

    # remove first punctuation
    while True:
        if len(inp) <= 1:
            return ""
        if inp[0] in ",.!:;?":
            inp = inp[1:]
        else:
            break
    inp_str = inp.replace("` '", "")
    inp_str = inp_str.replace(", ,", "")
    inp_str = inp_str.replace("  ", " ")

    # replace , '' if occurs at the end
    if inp_str.find(", ''", -13) > 0:
        loc = inp_str.find(", ''", -13)
        inp_str = inp_str[:loc] + ". ''" + inp_str[loc + 4:]

    # inp_str = inp_str.replace(", ''", ". ''")

    # Captulize the first char
    if inp_str[0].islower():
        inp_str = inp_str[0].upper() + inp_str[1:]

    inp_str = inp_str.strip()
    if inp_str[-1] not in punc:
        inp_str = inp_str + '.'
    inp_str = inp_str.replace(", .", ".")
    if len(inp_str) < 4:
        return ""
    return inp_str


def prepare_global_logger(stdout_file_name,
                          level=logging.DEBUG):
    ch = logging.StreamHandler()
    ch.setLevel(level + 10)
    logging.getLogger().addHandler(ch)
    if os.path.isfile(stdout_file_name):
        os.remove(stdout_file_name)
    stdout_handler = logging.FileHandler(stdout_file_name)
    stdout_handler.setLevel(level)
    logging.getLogger().addHandler(stdout_handler)
    logging.getLogger().setLevel(level)
    return logging.getLogger()


import random


def prepare_file_name(**kwargs):
    _tmp = []
    rand_id = random.randrange(1000)
    _tmp.append(str(rand_id))
    for k, v in list(kwargs.items()):
        _tmp.append('{}_{}'.format(k, v))
        if len(_tmp) > 10:
            break
    return '-'.join(_tmp)


def convert_list_to_paragraph(inp, split_token='@@SS@@'):
    # ['Gauk-Roger', 'contributed','@@SS@@', 'to', 'this', 'report', '.', '@@SS@@']
    # => ["Gauk-Roger contributed", "to this report ."]
    bag = []
    buff = []
    for x in inp:
        if (x == '') or (x == ' '):
            continue
        if x != split_token:
            buff.append(x)
        else:
            if buff is not []:
                bag.append(' '.join(buff))
                buff = []
    if buff is not []:
        bag.append(' '.join(buff))
    return bag


def log_predict_example(name, pred_label, gold_label, pred_abs, gold_abs):
    logger = logging.getLogger()
    logger.info("\nName: {}\t~~Logit: {}\t~~Label: {}"
                "\n--Pred: {}\n--Ref: {}\n".format(name,
                                                   pred_label, gold_label,
                                                   ' | '.join(pred_abs),
                                                   ' | '.join(gold_abs)))
    if random.random() < 0.02:
        print("\nName: {}\t~~Logit: {}\t~~Label: {}"
              "\n--Pred: {}\n--Ref: {}\n".format(name,
                                                 pred_label, gold_label,
                                                 ' | '.join(pred_abs),
                                                 ' | '.join(gold_abs)))


def log_compression_example(name, pred_sent, pred_rouge, potential_oracle_sent, potential_rouge, abs):
    logger = logging.getLogger()
    assert len(pred_sent) == len(pred_rouge) == len(potential_oracle_sent) == len(potential_rouge)
    s = []
    for a, b, c, d, in zip(pred_sent, pred_rouge, potential_oracle_sent, potential_rouge):
        s.append("=-\t{0}\t{1:.2f}\t{2}\t{3:.2f}".format(a, b, c, d))
    s = "\n".join(s)
    logger.info("=-Name: {}\t=-Pred_Sent\t=-Potential\n{}"
                "\n--Ref: {}\n".format(name,
                                       s,
                                       ' | '.join(abs)))


from typing import List


def log_universal(**kwargs):
    logger = logging.getLogger()
    pr = []
    for key, value in kwargs.items():
        if type(value) == str:
            pr.append("{}: {}".format(key, value))
        elif type(value) == float:
            pr.append("{0}: {1:.2f}".format(key, value))
        elif type(value) == List:
            for ele in value:
                pr.append("{}: {}".format(key, ele))
        else:
            pr.append("{}: {}".format(key, value))
    logger.info("\n".join(pr))


def log_a_string(inp):
    logger = logging.getLogger()
    logger.info(inp)


def isnan(x):
    return x != x


def checkNaN(inp):
    x = (inp != inp)
    indicator = torch.isinf(inp)
    if torch.sum(indicator) > 0:
        print(indicator)
        raise KeyError("Inf")
        exit()
    if torch.sum(x) > 0:
        print(inp)
        raise KeyError("NaN")
        exit()


def read_merge_span(fpath):
    span_txt = []
    span_sent_idx = []

    txt_buff = []
    sent_idx_buff = 0
    current_span_idx = 1
    with open(fpath, 'r') as fd:
        lines = fd.read().splitlines()
    for l in lines:
        if len(l) < 2:
            txt_buff.append('@@SS@@')
        else:
            tabs = l.split('\t')
            sent_idx, word, edu_idx = int(tabs[0]), tabs[2], int(tabs[-1])

            if edu_idx != current_span_idx:
                span_sent_idx.append(sent_idx_buff)
                sent_idx_buff = sent_idx
                span_txt.append(' '.join(txt_buff))
                txt_buff = [word]
                current_span_idx = edu_idx
            else:
                txt_buff.append(word)
    if txt_buff != []:
        span_sent_idx.append(sent_idx_buff)
        span_txt.append(' '.join(txt_buff))
    return span_txt, span_sent_idx


def batch_extraction_from_dict(batch_data, key_name):
    return [d[key_name] for d in batch_data]


def convert_msk_to_span_indices(mask_matrix):
    """
    mask_matrix: batchsz, seq_len       float tensor
    :return: # (batch_size, num_spans=1, 2)
    """
    device_id = mask_matrix.get_device()
    batchsz = mask_matrix.size()[0]
    # print(device_id)
    len_info = torch.sum(mask_matrix, dim=1).data
    span = torch.zeros([batchsz, 1, 2], dtype=torch.long, device=torch.device('cuda:{}'.format(device_id)))
    for idx, l in enumerate(len_info):
        span[idx, 0, 1] = l - 1
    # print(span)
    return span


def read_merge_simple(fpath):
    """Only return the words as a string"""
    with open(fpath, 'r') as fd:
        lines = fd.read().splitlines()
        bag = []
        for l in lines:
            if len(l) < 2:
                bag.append('\n')
            else:
                bag.append(l.split('\t')[2])
    return ' '.join(bag)


def print_size(*args, **kwargs):
    for x in args:
        print(x.size())
    for k, v in enumerate(kwargs):
        print("K :{}".format(v.size()))


import datetime


def log_board_file(fpath, args, metrics: dict):
    rt_list = []
    if args.dbg:
        return
    time = datetime.datetime.now()
    rt_list.append("Time: {}".format(time))

    rt_list.append("Data Name")
    rt_list.append(args.data_name)
    rt_list.append("Compression")
    rt_list.append(args.compression)
    rt_list.append("aggressive_compression")
    rt_list.append(args.aggressive_compression)
    rt_list.append("alpha")
    rt_list.append(args.alpha)
    rt_list.append("fix_edu_num")
    rt_list.append(args.fix_edu_num)
    rt_list.append("schedule")
    rt_list.append(args.schedule)
    rt_list.append("compress_leadn")
    rt_list.append(args.compress_leadn)

    for k in metrics:
        if k.startswith("validation_") or k.startswith("best_test_") or (k == "training_epochs") or (
                k == "training_loss") or (k == "validation_loss"):
            rt_list.append(k)
            rt_list.append(metrics[k])
    rt_list.append("FileName")
    rt_list.append(args.fname)
    # log all validation_
    # log all best_test_
    # log training_epochs training_loss validation_loss
    rt_list = [str(x) for x in rt_list]
    rt = "\t".join(rt_list)
    if not os.path.isfile(fpath):
        with open(fpath, 'w') as fd:
            fd.write("\n")

    with open(fpath, 'a') as file:
        file.write("\n" + rt)
