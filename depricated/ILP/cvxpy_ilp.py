import random

import cvxpy as cp
from typing import List
# proceed ref grams
import nltk
from nltk.tokenize import word_tokenize

from neusum.evaluation.smart_approx_rouge import get_rouge_est_str_2gram_smart
import numpy as np


def ILP_protocol(reference_summary: str, sent_units: List[str],
                 min_word_limit=30, max_word_limit=180, step=5):
    result_bag = {}
    # "nlabel": num of sent picked,
    # "data": {0.2539969834087481: {'label': [1, 13, 16], 'R1': 0.2539969834087481, 'nlabel': 3}, ...},
    # "best": None

    best_score = 0
    early_stop_cnt_down = 0

    # raw_ref_token = word_tokenize(reference_summary)
    raw_ref_token = reference_summary.split(" ")

    raw_txt_tokens = [sent_unit.split(" ") for sent_unit in sent_units]

    # keep track of len of text units
    txt_lens = [cp.Constant(np.int_(len(raw_txt_token))) for raw_txt_token in
                raw_txt_tokens]  # for constraint computation

    raw_txt_token_set = set(sum(raw_txt_tokens, []))
    fined_ref_token_set = raw_txt_token_set.intersection(set(raw_ref_token))

    fined_txt_tokens = []
    for sent_unit in raw_txt_tokens:
        set_sent_unit = set(sent_unit)
        set_sent_unit.intersection_update(fined_ref_token_set)
        fined_txt_tokens.append(list(set_sent_unit))
    # remove oov words in texts w.r.t. ref

    # remove redundancy in ref summary
    ref_tok = list(fined_ref_token_set)
    ref_len = len(ref_tok)

    for max_len in range(min_word_limit, max_word_limit, step):
        # print(max_len)
        all_of_constraints = []
        # Reference tok indicator. r_i. objective = sum(r_i)
        ref_gram_var = [cp.Variable(boolean=True) for _ in range(ref_len)]
        # ref_gram_var = cp.Variable(shape=ref_len, boolean=True)

        # sum( s_i ) - sum(compression) >=  ref_occurance_constraint
        ref_occurance_constraints = [None for _ in range(ref_len)]

        assert len(sent_units) > 1
        # sent_vars = cp.Variable(shape=(len(sent_units)), boolean=True)
        sent_var_list = [cp.Variable(boolean=True) for _ in range(len(sent_units))]
        for sent_idx, txt_tok in enumerate(fined_txt_tokens):
            # txt_tok is a list of str
            check_every_sent(txt_tok, ref_tok, ref_occurance_constraints,
                             sent_var_list[sent_idx])
            # ref_occurance_constraints = check_every_sent(txt_tok, ref_tok, ref_occurance_constraints,
            # sent_vars[sent_idx])
        #
        # print(ref_occurance_constraints)
        for idx, x in enumerate(ref_occurance_constraints):
            all_of_constraints = all_of_constraints + [
                ref_occurance_constraints[idx] <= -ref_gram_var[idx]]

        # length restriction
        length_r = None
        for l, v in zip(txt_lens, sent_var_list):
            if length_r:
                length_r = length_r + v * l
            else:
                length_r = v
        len_restrict = length_r
        # len_restrict = cp.sum([l * v for l, v in zip(txt_lens, sent_var_list)])
        # cp_lens = cp.Constant(txt_lens)
        # len_restrict = cp.sum(cp.multiply(cp_lens, sent_vars))

        constraints_with_length = all_of_constraints + [len_restrict <= cp.Constant(np.int_(max_len))]
        # constraints_with_length = all_of_constraints
        # all_of_constraints.append(len_restrict <= max_len)
        # constraints_with_length.append(1 + cp.min(sent_vars) >= 1)
        # obj = None
        # for ref_v in ref_gram_var:
        #     if obj:
        #         obj = obj + ref_v
        #     else:
        #         obj = ref_v
        # obj_var = obj
        #
        obj_var = sum(ref_gram_var)

        obj = cp.Maximize(cp.sum(obj_var))
        prob = cp.Problem(obj, constraints_with_length)

        prob.solve(solver=cp.GLPK_MI)
        if random.random() < 0.001:
            print(prob)
            print(prob.value)
        # print(prob.value)
        # try:
        # print(sent_var_list)
        pred_index_list = []
        for pred_idx, var in enumerate(sent_var_list):
            if var.value == None:
                print('-' * 100)
                print(prob.status)
                print(sent_var_list)

                print(sent_units)
                exit()
            if var.value > 0.001:
                pred_index_list.append(pred_idx)

        # print(fname_without_suffix)
        pred_index_list.sort()
        num_sent_sel = len(pred_index_list)
        selected_sents_str = [sent for idx, sent in enumerate(sent_units) if idx in pred_index_list]
        selected_sents_str = "\n".join(selected_sents_str)
        score = get_rouge_est_str_2gram_smart(gold=reference_summary, pred=selected_sents_str)
        # print(score)

        if str(num_sent_sel) not in result_bag:
            result_bag[str(num_sent_sel)] = {
                "nlabel": num_sent_sel,
                "data": {}
            }
        result_bag[str(num_sent_sel)]["data"][str(score)] = {'label': pred_index_list, 'R1': score,
                                                             'nlabel': num_sent_sel,
                                                             # 'sent': selected_sents_str
                                                             }
        early_stop_cnt_down += 1
        if score > best_score:
            early_stop_cnt_down = 0
            best_score = score
        if early_stop_cnt_down >= 5:
            break
        del ref_gram_var
        del len_restrict
        del sent_var_list
        del ref_occurance_constraints
        del all_of_constraints, prob, obj
    # print(result_bag)
    return result_bag


def compression_of_one_sentence():
    pass


def ILP_protocol_w_compression(reference_summary: str, sent_units: List[str], compression: List[dict],
                               min_word_limit=30, max_word_limit=40, step=3):
    print("Compression")
    constraint_list = []
    ref_toks = reference_summary.split(" ")
    ref_toks = [x.lower() for x in ref_toks]
    ref_toks_set = list(set(ref_toks))
    uniq_tok_num = len(ref_toks_set)
    y_tok = cp.Variable(shape=(uniq_tok_num), boolean=True)

    len_doc = len(sent_units)
    sent_var = cp.Variable(shape=len_doc, boolean=True)
    len_of_each_sentence = cp.Constant([len(x) for x in sent_units])
    length_constraint_sents = sent_var * len_of_each_sentence
    constraint_list.append(length_constraint_sents <= max_word_limit)
    obj = cp.Maximize(cp.sum(ref_toks))
    prob = cp.Problem(obj, constraints=constraint_list)
    print(prob)
    # prob.solve(solver=cp.GLPK_MI)
    prob.solve()
    print(prob.status)
    print(obj.value)
    print(y_tok.value)
    print(sent_var.value)
    exit()


def check_every_sent(sent: List[str], ref_tok: List[str], ref_occurance_constraints, x_i):
    # for every sent, check the ref_gram and find those overlapping.
    # x_i = cp.Variable(boolean=True)  # the indicator of the inclusion of the current text unit
    # const_bag = []
    # sent_gram = word_tokenize(sent)
    # l = len(sent_gram)
    # sent_gram = list(set(sent_gram))
    for gram in sent:
        # if gram not in ref_gram:
        #     continue
        idx = ref_tok.index(gram)
        if ref_occurance_constraints[idx]:
            ref_occurance_constraints[idx] = ref_occurance_constraints[idx] - x_i
        else:
            ref_occurance_constraints[idx] = - x_i

        # constrain = cp.constraints.nonpos.NonPos(x_i - ref_var[idx])
        # const_bag.append(constrain)
    # return ref_occurance_constraints


import numpy as np
#
# if __name__ == '__main__':
#     ILP_protocol(reference_summary="a b c d e f g h i",
#                  sent_units=["a b c x y z", "d a d i a d a", "a b", "a a a a e f", "d e f a a a a a a a a"],
#                  max_word_limit=8)
#     exit()
#     from ILP.data_example import *
#
#     ref_grams = word_tokenize(summary)
#     ref_grams = list(set(ref_grams))
#     ref_len = len(ref_grams)
#
#     ref_gram_var = cp.Variable(shape=ref_len, boolean=True)
#     ref_constrain_var = [cp.Constant(0) for _ in range(ref_len)]
#     all_of_constraints = []
#     xs, ls = [], []
#     for sent in doc:
#         ref_constrain_var, x_i, len_of_sent = check_every_sent(sent, ref_grams, ref_constrain_var)
#         # x_i: :cp.Variable
#         ls.append(cp.Constant(len_of_sent))
#         xs.append(x_i)
#         # all_of_constraints += constraint_vars
#     # len_unit = cp.Constant(np.asarray(ls))
#     for idx, x in enumerate(ref_constrain_var):
#         all_of_constraints.append(x >= ref_gram_var[idx])
#
#     len_unit = ls
#     s = 0
#     for idx in range(len(xs)):
#         s += xs[idx] * len_unit[idx]
#     all_of_constraints += [s <= 10]
#     obj = cp.Maximize(cp.sum(ref_gram_var))
#     prob = cp.Problem(obj, all_of_constraints)
#     prob.solve()
#     print(prob)
#     print(prob.solution)
#     # print(prob.solver_stats)
#     print(ref_gram_var.value)
#     # print(all_of_constraints)
#     exit()
# import itertools
