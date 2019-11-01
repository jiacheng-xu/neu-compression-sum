import json
import multiprocessing

root = '/home/cc/final-cnn/merge'
# root = '/backup2/jcxu/data/cnn-v1/merge'
import itertools
import time
from neusum.evaluation.rough_rouge import get_rouge_est_str_4gram
top_k_combo = 5
P_SENT = 10
NUM_EDU = [2, 3, 4, 5]

from neusum.service.basic_service import read_merge_span, read_merge_simple
from neusum.evaluation.rough_rouge import  get_rouge_est_str_2gram
# First get the rouge of all EDUs individually
# Then pick up top P_SENT
# Combine all the P_SENT into NUM_EDU sents according to the order
# (get the length to compare with EDU)
# Get the top top_k_combo combinations

import numpy
# from neusum.evaluation.rouge_with_pythonrouge import rouge_protocol
from typing import List


def assemble_doc_list_from_idx(doc, idxs):
    _tmp = []
    for i in idxs:
        _tmp.append(doc[i])
    return _tmp


def comp_oracle_combination(_filtered_doc_list,
                            _num_edu,
                            _absas_read_str,
                            abs_as_read_list,
                            map_from_new_to_ori_idx,
                            beam_sz=4):
    pass


def comp_num_seg_out_of_p_sent_beam(_filtered_doc_list,
                                    _num_edu,
                                    _absas_read_str,
                                    abs_as_read_list,
                                    map_from_new_to_ori_idx,
                                    beam_sz=8):
    beam = []
    if len(_filtered_doc_list) <= _num_edu:
        return {"nlabel": _num_edu,
                "data": {},
                "best": None
                }

    combs = list(range(1, len(_filtered_doc_list)))     ## TODO should be 1 because SOS as the first sentence
    # _num_edu seq_len
    cur_beam = {
        "in": [],
        "todo": combs,
        "val": 0
    }
    beam.append(cur_beam)
    for t in range(_num_edu):
        dict_pattern = {}
        # compute top beam_sz for every beam
        global_board = []
        for b in beam:
            already_in_beam = b['in']
            todo = b['todo']

            leaderboard = {}
            for to_add in todo:
                after_add = already_in_beam + [to_add]
                _tmp = assemble_doc_list_from_idx(_filtered_doc_list, after_add)
                _tmp = '\n'.join(_tmp)
                average_f_score = get_rouge_est_str_4gram(_absas_read_str, _tmp)

                leaderboard[to_add] = average_f_score
            sorted_beam = [(k, leaderboard[k]) for k in sorted(leaderboard, key=leaderboard.get, reverse=True)]

            for it in sorted_beam:
                new_in = already_in_beam + [it[0]]
                new_in.sort()
                str_new_in = [str(x) for x in new_in]
                if '_'.join(str_new_in) in dict_pattern:
                    continue
                else:
                    dict_pattern['_'.join(str_new_in)] = True
                new_list = todo.copy()
                new_list.remove(it[0])
                # rank_actual_idx = sort_idx_map[it[0]]
                # new list
                # if rank_actual_idx + 1 == len(_filtered_doc_list):
                #     continue
                # it[0] is the index in combs = index in filter_doc_list
                # sort_idx_map ====> the rank in original document
                # new_list only contains stuff have larger rank than rank_actual_idx
                # for _i, _rank in enumerate(sort_idx_map):
                #     if _rank > rank_actual_idx:
                #         new_list.append(combs[_i])
                # assert len(new_list) != 0
                _beam = {
                    "in": new_in,
                    "todo": new_list,
                    "val": it[1]
                }
                global_board.append(_beam)
        # merge and get the top beam_sz among all

        sorted_global_board = sorted(global_board, key=lambda x: x["val"], reverse=True)

        _cnt = 0
        check_dict = []
        beam_waitlist = []
        for it in sorted_global_board:
            str_in = sorted(it['in'])
            str_in = [str(x) for x in str_in]
            _tmp_key = '_'.join(str_in)
            if _tmp_key in check_dict:
                continue
            else:
                beam_waitlist.append(it)
                check_dict.append(_tmp_key)
            _cnt += 1
            if _cnt >= beam_sz:
                break
        beam = beam_waitlist
    # if len(beam) < 2:
    #     print(len(_filtered_doc_list))
    #     print(_num_edu)
    # Write oracle to a string like: 0.4 0.3 0.4
    _comb_bag = {}
    for it in beam:
        n_comb = it['in']
        n_comb.sort()
        n_comb_original = [map_from_new_to_ori_idx[a] for a in n_comb]
        n_comb_original.sort()  # json label
        n_comb_original = [int(x) for x in n_comb_original]
        # print(n_comb_original)
        _tmp = assemble_doc_list_from_idx(_filtered_doc_list, n_comb)
        # score = rouge_protocol([[_tmp]], [[abs_as_read_list]])
        _tmp = '\n'.join(_tmp)
        f1 = get_rouge_est_str_4gram(_absas_read_str, _tmp)

        # f1 = score['ROUGE-1-F']
        # f2 = score['ROUGE-2-F']
        # fl = score['ROUGE-L-F']
        # f_avg = (f1 + f2 + fl) / 3
        _comb_bag[f1] = {"label": n_comb_original,
                         "R1": f1,
                         # "R2": f2,
                         # "RL": fl,
                         # "R": f_avg,
                         "nlabel": _num_edu}
    # print(len(_comb_bag))
    if len(_comb_bag) == 0:
        return {"nlabel": _num_edu,
                "data": {},
                "best": None
                }
    else:
        best_key = sorted(_comb_bag.keys(), reverse=True)[0]
        rt_dict = {"nlabel": _num_edu,
                   "data": _comb_bag,
                   "best": _comb_bag[best_key]
                   }
        return rt_dict


def sent_oracle(doc_list: List, abs_str, name, path_write_data,
                use_beam=True, approx_rouge_filter=True):
    """

    :param doc_list: List of strings. all \n are represented as @@SS@@
    :param abs_str: a single string will \n
    :param name:
    :return:
    """
    # print(name)
    # print(time.localtime())
    doc_list = [d.strip() for d in doc_list]  # trim
    for d in doc_list:
        assert len(d) > 0
    len_of_doc = len(doc_list)
    doc_list.insert(0, "<SOS>")
    doc_as_readable_list = [d.replace("@@SS@@", "").strip() for d in doc_list]
    abs_as_readable_list = [x for x in abs_str.split("\n") if (x != "") and (x != " ")]  # no SS and \n

    doc_list = [d.replace('\n', '@@SS@@') for d in doc_list]
    doc_list = [" ".join(d.split()) for d in doc_list]
    rt_doc = ' '.join(doc_list)
    # abs_str = abs_str.replace('\n', '@@SS@@')
    rt_abs = abs_str.replace('\n', '@@SS@@')
    # print(len(rt_doc.split(" ")))

    span = []
    jdx = 0
    for d in doc_list:
        num = len([x for x in d.split(' ') if x != ''])
        span.append(str(jdx))
        span.append(str(jdx + num - 1))
        jdx += num
    assert (len(rt_doc.split(" ")) - int(span[-1]) - 1) == 0
    # return None
    span_str = ' '.join(span)
    f_score_list = []
    f_score_full = []
    for i in range(len_of_doc):
        if approx_rouge_filter:
            average_f_score, f1, f2, fl = get_rouge_est_str_2gram(gold='\n'.join(abs_as_readable_list),
                                                                  pred=doc_as_readable_list[i])

        else:
            score = rouge_protocol([[doc_as_readable_list[i]]], [[abs_as_readable_list]])
            f1 = score['ROUGE-1-F']
            f2 = score['ROUGE-2-F']
            fl = score['ROUGE-L-F']

        f_score_full.append([f1, f2, fl])
        f_score_list.append(f1 + f2 + fl)
    top_p_sent_idx = numpy.argsort(f_score_list)[-P_SENT:]

    map_from_new_to_ori_idx = []
    # filter
    filtered_doc_list = []
    for i in range(len(top_p_sent_idx)):
        filtered_doc_list.append(doc_as_readable_list[top_p_sent_idx[i]])
        map_from_new_to_ori_idx.append(top_p_sent_idx[i])

    # filter_doc_list stores filtered doc
    # map_from_new_to_ori_idx contains their original index
    combination_data_dict = {}
    for num_of_edu in NUM_EDU:
        if use_beam:
            combination_data = comp_num_seg_out_of_p_sent_beam(_filtered_doc_list=filtered_doc_list,
                                                               _num_edu=num_of_edu,
                                                               _absas_read_str=abs_str,
                                                               abs_as_read_list=abs_as_readable_list,
                                                               map_from_new_to_ori_idx=map_from_new_to_ori_idx)
        else:
            raise NotImplementedError
            # combination_data = comp_num_seg_out_of_p_sent(_filtered_doc_list=filtered_doc_list, _num_edu=num_of_edu,
            #                                               _absas_read_str=abs_str,
            #                                               abs_as_read_list=abs_as_readable_list,
            #                                               map_from_new_to_ori_idx=map_from_new_to_ori_idx)
        combination_data_dict[num_of_edu] = combination_data
    json_str = json.dumps(combination_data_dict)
    rt = '\t'.join([name, rt_doc, rt_abs, span_str, json_str])

    # with open(os.path.join(path_write_data, name + '.txt'), 'w') as fd:
    #     fd.write(rt)

    return rt


import sys


def truncate_doc_list(_max_edu_num, doc):
    bag = []
    for idx, content in enumerate(doc):
        if idx >= _max_edu_num:
            if "@@SS@@" in content:
                bag.append(content)
                break
        bag.append(content)
    return bag


def convert_edu_doc_to_sent_doc(doc):
    buff = []
    bag = []
    for i, content in enumerate(doc):
        if '@@SS@@' in content:
            buff.append(content)
            buff = ' '.join(buff)
            bag.append(buff)
            buff = []
        else:
            buff.append(content)
    if buff != []:
        buff = ' '.join(buff)
        bag.append(buff)
    return bag


def comp_sent_ora(single_file, max_edu_num, beam, path_doc, path_abs, path_write_data):
    doc_files = os.listdir(path_doc)
    f_docs = [f for f in doc_files if f.endswith('.doc.merge')]

    abs_files = os.listdir(path_abs)
    f_abss = [f for f in abs_files if f.endswith('.abs.merge')]

    assert len(f_docs) == len(f_abss)

    bag_abs = []
    bag_doc = []
    bag_name = []
    # f_docs = f_docs[:100]
    for j, fdoc in enumerate(f_docs):
        # print(j)
        name = fdoc.split('.')[0]
        doc_spans, doc_sent_idx = read_merge_span(os.path.join(path_doc, fdoc))
        doc_spans = convert_edu_doc_to_sent_doc(doc_spans)
        doc_spans = truncate_doc_list(max_edu_num, doc_spans)
        doc_spans = [s for s in doc_spans if len(s) > 0]
        inp_abs_str = read_merge_simple(os.path.join(path_abs, name + '.abs.merge'))

        bag_doc.append(doc_spans)
        bag_abs.append(inp_abs_str)
        bag_name.append(name)

        # f = sent_oracle(doc_spans, inp_abs_str, name, path_write_data)

    bag_path_write_data = [path_write_data] * len(bag_name)
    bag_beam = [beam] * len(bag_name)
    assert len(bag_doc) == len(bag_abs)
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    pairs = pool.starmap(sent_oracle, zip(bag_doc, bag_abs, bag_name, bag_path_write_data, bag_beam))
    pool.close()
    pool.join()
    pairs = [p for p in pairs if p is not None]
    print('Final Stage')
    with open(single_file, 'w') as fd:
        fd.write('\n'.join(pairs))


def comp_edu_ora(single_file, max_edu_num, beam, path_doc, path_abs, path_write_data):
    doc_files = os.listdir(path_doc)
    f_docs = [f for f in doc_files if f.endswith('.doc.merge')]

    abs_files = os.listdir(path_abs)
    f_abss = [f for f in abs_files if f.endswith('.abs.merge')]

    assert len(f_docs) == len(f_abss)

    bag_abs = []
    bag_doc = []
    bag_name = []
    f1_bag, f2_bag, fl_bag = [], [], []
    start_time = time.time()
    # f_docs = f_docs[:100]
    for j, fdoc in enumerate(f_docs):
        # print(j)
        name = fdoc.split('.')[0]
        doc_spans, doc_sent_idx = read_merge_span(os.path.join(path_doc, fdoc))
        doc_spans = truncate_doc_list(max_edu_num, doc_spans)
        doc_spans = [s for s in doc_spans if len(s) > 3]
        inp_abs_str = read_merge_simple(os.path.join(path_abs, name + '.abs.merge'))

        bag_doc.append(doc_spans)
        bag_abs.append(inp_abs_str)
        bag_name.append(name)

        # f = sent_oracle(doc_spans, inp_abs_str, name, path_write_data)

    bag_path_write_data = [path_write_data] * len(bag_name)
    bag_beam = [beam] * len(bag_name)
    assert len(bag_doc) == len(bag_abs)
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    pairs = pool.starmap(sent_oracle, zip(bag_doc, bag_abs, bag_name, bag_path_write_data, bag_beam))
    pool.close()
    pool.join()
    pairs = [p for p in pairs if p is not None]
    print('Final Stage')
    with open(single_file, 'w') as fd:
        fd.write('\n'.join(pairs))


if __name__ == '__main__':
    # path_doc_ = root + '/dev/doc'
    # path_abs_ = root + '/dev/abs'
    # wt_to = root + '/dev.txt'
    # comp_edu_ora(path_doc=path_doc_, path_abs=path_abs_)

    beam_flag = True if sys.argv[1].lower() == 'true' else False
    partition_name = sys.argv[2]
    max_edu_num = int(sys.argv[3])
    sent_level = True if sys.argv[4].lower() == 'true' else False
    print("beam",
          beam_flag)
    start = time.time()
    dir = root + '/{}/'.format(partition_name)
    if beam_flag:
        if sent_level:
            main_write = dir + 'sent-{}-{}-beam.txt'.format(partition_name, max_edu_num)
        else:
            main_write = dir + '{}-{}-beam.txt'.format(partition_name, max_edu_num)
    else:
        if sent_level:
            main_write = dir + 'sent-{}-{}-brute.txt'.format(partition_name, max_edu_num)
        else:
            main_write = dir + '{}-{}-brute.txt'.format(partition_name, max_edu_num)

    path_doc_ = dir + 'doc'
    path_abs_ = dir + 'abs'
    if beam_flag:
        path_write_data = dir + "data_byline_beam"
    else:
        path_write_data = dir + "data_byline_brute"
    import os
    import shutil

    if os.path.isdir(path_write_data):
        shutil.rmtree(path_write_data)
        print("delete previous data")

    os.mkdir(path_write_data)
    # wt_to = root + '/test-nobeam.txt'
    if sent_level:
        comp_sent_ora(main_write, max_edu_num, beam_flag, path_doc=path_doc_, path_abs=path_abs_,
                      path_write_data=path_write_data)
    else:
        comp_edu_ora(main_write, max_edu_num, beam_flag, path_doc=path_doc_, path_abs=path_abs_,
                     path_write_data=path_write_data)
    end = time.time()
    print(end - start)
    # path_doc_ = root + '/train/doc'
    # path_abs_ = root + '/train/abs'
    # wt_to = root + '/train.txt'
    # comp_edu_ora(path_doc=path_doc_, path_abs=path_abs_)
