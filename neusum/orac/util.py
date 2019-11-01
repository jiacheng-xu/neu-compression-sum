from typing import List
import copy
import re
import random
# from neusum.evaluation.rough_rouge import get_rouge_est_str_2gram
from neusum.evaluation.smart_approx_rouge import get_rouge_est_str_2gram_smart
import numpy

top_k_combo = 5
P_SENT = 15
NUM_EDU = [1, 2, 3, 4, 5]
import json
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

word_token_indexers = {"tokens": SingleIdTokenIndexer()}

CONTENT_LIB = ['.', 'the', ',', 'to', 'of', 'a', 'and', 'in', "'s", 'was', 'for', 'that', '`', 'on',
               'is', 'The', "'", 'with', 'said', ':', 'his', 'he', 'at', 'as', 'it', 'I', 'from',
               'have', 'has', 'by', '``', "''", 'be', 'her', 'are', 'who', 'an', 'had', 'not', 'been',
               'were', 'they', 'their', 'after', 'she', 'but', 'this', 'will', '--', 'which', "n't",
               'It', 'when', 'up', 'out', 'one', 'about']


def read_one(file_path):
    instances = []
    with open(file_path, 'r') as fd:
        for line in fd:
            data_dict = json.loads(line)
            doc_str = data_dict['doc']
            allen_token_word_in_doc = TextField([Token(word) for word in doc_str.split()], word_token_indexers)
            instances.append(Instance({"text": allen_token_word_in_doc}))
    return instances


def modify_sentences(sent_idx, text_bits, rm):
    before = text_bits[sent_idx]
    after = before - rm
    text_bits[sent_idx] = after
    return text_bits


def flip_exsisting_combination(topk_combinations, sentences, abs_str, flip=0.2):
    drop_num = 0
    best_combi = topk_combinations[0]
    trim_done = []
    for i in range(len(best_combi['done'])):
        if random.random() < flip:
            drop_num += 1
        else:
            trim_done.append(best_combi['done'][i])

    texts = []
    compressions = []
    for unit in sentences:
        texts.append(unit[0].text)
        compressions.append(unit[1])
    sent_num = len(sentences)
    text_bits = [set(range(len(x))) for x in texts]
    list_of_compressions = []
    for sent_idx, comp in enumerate(compressions):
        for one_comp_option in comp:
            if one_comp_option['node'] != "BASELINE":
                list_of_compressions.append([sent_idx, one_comp_option])
    total_len = len(list_of_compressions)

    replacement = random.sample(list_of_compressions, drop_num)

    for trim in trim_done:
        text_bits = modify_sentences(trim[0], text_bits, rm=trim[1]['selected_idx'])
    for rep in replacement:
        text_bits = modify_sentences(rep[0], text_bits, rm=rep[1]['selected_idx'])
    new_rouge = get_rouge_est_str_2gram(gold=abs_str, pred=
    assemble_text_and_bit(texts, text_bits))
    return new_rouge


def assemble_text(inp: List[List[str]]):
    bag = []
    for i in inp:
        bag.append(" ".join(i))
    return "\n".join(bag)


def assemble_doc_list_from_idx(doc, idxs):
    _tmp = []
    for i in idxs:
        _tmp.append(doc[i])
    return _tmp


def assemble_text_and_bit(text, bit):
    tmp = []
    for tt, bb in zip(text, bit):
        sent = [tt[b] for b in bb]
        tmp.append(" ".join(sent))
    return "\n".join(tmp)


class TreeNode():
    def __init__(self, tag: str, text, children: List, depth: int):
        self.text = text
        self.tag = tag
        self.children = children
        self.depth = depth
        self.start_idx = -1
        self.end_idx = -1

    def __repr__(self):
        return "Text: {}\tTag:{}\tDepth:{}".format(" ".join(self.text), self.tag, self.depth)


def read_single_parse_tree(inp_str) -> TreeNode:
    """
    Given a string from stanfordnlp, convert to a tree.
    :param inp_str: (ROOT\n  (FRAG\n    (NP (NN NEW))\n ....
    :return:
    """
    inp_str = inp_str.replace("\n", "")
    # inp_str = re.split('\(|\)|\s', inp_str)
    inp_str = re.sub(' +', ' ', inp_str)
    # inp = inp.split(" ")
    # inp_str = inp_str.replace("-LRB- (", "-LRB- [")
    # inp_str = inp_str.replace("-RRB- )", "-RRB- ]")
    out = parse_subtree(inp_str, depth=0)
    out = add_idx_of_tree(out, start_idx=0, end_idx=len(out.text))
    return out


def extract_tokens_allen(allen_lines):
    pass


def extract_tokens(snlp_dict):
    """

    :param snlp_dict:
    :return: buff: List[List[str]]
            string: a C B\nD e f\n...
    """
    sentences = snlp_dict['sentences']
    buff = []
    buff_str = []
    for s in sentences:
        tokens = s["tokens"]
        tokens_list = [x["word"] for x in tokens]
        buff.append(tokens_list)
        buff_str.append(" ".join(tokens_list))
    return buff, "\n".join(buff_str)


def read_file(fpath) -> str:
    with open(fpath, 'r') as fd:
        output = fd.read()
    return output


def extract_parse(snlp_dict) -> List[str]:
    sentences = snlp_dict['sentences']
    buff_parse = [s['parse'] for s in sentences]
    return buff_parse


def add_idx_of_tree(tree: TreeNode, start_idx: int, end_idx: int):
    tree.start_idx = start_idx
    tree.end_idx = end_idx
    if not tree.children:
        return tree
    cur_start = start_idx
    new_children = []
    for child in tree.children:
        span_len = len(child.text)
        new_children.append(add_idx_of_tree(child, start_idx=cur_start, end_idx=cur_start + span_len))
        cur_start += span_len
    tree.children = new_children
    return tree


def return_childrens(inp: str) -> List:
    dep = 0
    rt_list = []
    buff = []
    for idx, c in enumerate(inp):
        if c == '(':
            dep += 1
        elif c == ')':
            dep -= 1
        if buff == [] and c == " ":
            pass
        else:
            buff.append(c)
        if dep == 0 and buff != []:
            rt_list.append("".join(buff))
            buff = []
    return rt_list


def parse_subtree(inp_str: str, depth: int):
    # print(inp_str)
    index_lb = inp_str.find("(")
    index_rb = inp_str.rfind(")")
    idex_split_of_tag_children = inp_str.find(" ")
    tag = inp_str[index_lb + 1:idex_split_of_tag_children]
    child = inp_str[idex_split_of_tag_children + 1:index_rb]
    if "(" in child:
        children_list = return_childrens(child)  # (NP   =>(x x) (x x) (x x)<=)
        children_node = [parse_subtree(x, depth + 1) for x in children_list]
        txt_buff = []
        for n in children_node:
            txt_buff += n.text
        text = txt_buff
    else:
        # reach leaf node
        text = [child.strip()]
        children_node = None
    return TreeNode(tag=tag, text=text, children=children_node, depth=depth)


def folding_rouge(abs_list, txt_str: str):
    # experimental way
    n_fold = int(len(abs_list) / len(txt_str.split(" ")))
    num_word_one_fold = int(len(abs_list) / n_fold)
    rouge_bag = []
    for idx in range(n_fold):
        if idx == n_fold - 1:
            tmp_abs = abs_list[idx * num_word_one_fold:]
        else:
            tmp_abs = abs_list[idx * num_word_one_fold:(idx + 1) * num_word_one_fold]
        r = get_rouge_est_str_2gram_smart(gold=" ".join(tmp_abs), pred=txt_str)
        rouge_bag.append(r)
    new_rouge = sum(rouge_bag) / len(rouge_bag)
    return new_rouge
    # print("{}\t{}\t{}".format(new_rouge >= _rouge, new_rouge, _rouge))


def find_deletable_span_rule_based(tree: TreeNode, root_len: int, sibling=None,
                                   parent=None
                                   ) -> List[TreeNode]:
    """
    Extraction:
    1. S (!= VP and len(S) > 8 words)
        case: [S [NP county officials in Las Vegas] [VP appointed two women [S [VP to fill vacancies in the state Assembly]]] ]
        When S=VP, it's not an individual sentence.

    Compression:
    1. PP
    2. SBAR
    3. ADVP
    4. ADJP
    5. S (== VP and left != [WHNP,IN]) because WHNP + S = SBAR
        eg. in playing soccer
    6. NP-TMP
    7. -LRB-  ... ... -RRB-
    """
    # Rule is a list with TAG names
    # return a list of dict{"node", "selected_idx"} which are deletable
    tag = tree.tag
    deletable_bag = []

    # Extraction
    if tree.tag == 'S':
        if tree.children[0] != None:

            if (tree.children[0].tag != "VP") and (len(tree.text) > 5) and (tree.end_idx < root_len):
                deletable_bag += [{"node": "_S_",
                                   "selected_idx": set(range(root_len)) -
                                                   set(range(tree.start_idx, tree.end_idx))}]
            if (tree.children[0].tag == "VP") and (sibling is not None) and (sibling.tag not in ["WHNP", "IN"]):
                deletable_bag += [{"node": tree.tag, "selected_idx": set(range(tree.start_idx, tree.end_idx))}]

    if tree.children is not None:
        lrb, rrb = -1, -1
        for idx, child in enumerate(tree.children):
            if child.tag == "-LRB-":
                lrb = child.start_idx
            elif child.tag == '-RRB-':
                rrb = child.end_idx
        if (lrb >= 0) and (rrb >= 0) and (lrb < rrb):
            deletable_bag += [{"node": "LRRB", "selected_idx": set(range(lrb, rrb))}]
            # print({"par_text":parent.text,"text":tree.children,"node": "LRRB", "selected_idx": set(range(lrb, rrb))})

    if tag in ["PP", "SBAR", "ADVP", "ADJP", "NP-TMP", "PRN"]:
        deletable_bag += [{"node": tree.tag, "selected_idx": set(range(tree.start_idx, tree.end_idx))}]
    # if ' '.join(tree.text) == '-LRB- CNN -RRB-':
    #     deletable_bag += [{"node": "PRN", "selected_idx": set(range(tree.start_idx, tree.end_idx))}]

    if parent is not None:
        if (parent.tag == "NP") and (tree.children is not None):
            for idx, child in enumerate(tree.children):
                if child.tag == 'JJ':
                    deletable_bag += \
                        [{"node": "JJ", "selected_idx": set(range(child.start_idx, child.end_idx))}]
                    # print({"par_text":parent.text,"text":child.text,"node": "JJ", "selected_idx": set(range(child.start_idx, child.end_idx))})
    if tree.children is not None:
        for idx, child in enumerate(tree.children):
            if idx == 0:
                deletable_bag += find_deletable_span_rule_based(child, root_len, None, parent=tree)
            else:
                deletable_bag += find_deletable_span_rule_based(child, root_len, tree.children[idx - 1], parent=tree)

    return deletable_bag


def check_if_redundant(text_bit, pool):
    if any([True for p in pool if p['text_bits'] == text_bit]):
        return True
    else:
        return False


def check_if_empty_lists(inp):
    if inp == []:
        return True
    for i in inp:
        if i != []:
            return False
    return True


"""
def length_compensation(doc, abs: List) -> str:
    if type(doc) is str:
        doc = doc.split(" ")
    l_abs = len(abs)
    l_doc = len(doc)
    if l_abs > l_doc:
        gap = l_abs - l_doc
        backup = CONTENT_LIB[:gap]
    else:
        backup = []
    doc = doc + backup
    return " ".join(doc)
"""


def comp_document_oracle(doc_list: List, abs_str, beam_sz=6):
    """
    Given a document and the abstraction string, return the oracle set combination
    :param doc_list: List of strings.
    :param abs_str: a single string with \n
    :param name:
    :return:
    """
    doc_list = [d.strip() for d in doc_list]  # trim
    # for d in doc_list:
    #     assert len(d) > 0
    len_of_doc = len(doc_list)
    doc_as_readable_list = doc_list
    abs_as_readable_list = [x for x in abs_str.split("\n") if (x != "") and (x != " ")]  # no SS and \n

    f_score_list = []
    for i in range(len_of_doc):
        f1 = get_rouge_est_str_2gram_smart(gold=abs_str, pred=doc_as_readable_list[i])
        # len_compensat_inp = length_compensation(doc=doc_as_readable_list[i], abs=abs_as_readable_list)
        # f1 = get_rouge_est_str_2gram(gold='\n'.join(abs_as_readable_list),
        #                              pred=len_compensat_inp)

        f_score_list.append(f1)
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
        combination_data = comp_num_seg_out_of_p_sent_beam(_filtered_doc_list=filtered_doc_list,
                                                           _num_edu=num_of_edu,
                                                           _absas_read_str=abs_str,
                                                           abs_as_read_list=abs_as_readable_list,
                                                           map_from_new_to_ori_idx=map_from_new_to_ori_idx,
                                                           beam_sz=beam_sz)
        combination_data_dict[num_of_edu] = combination_data
    # json_str = json.dumps(combination_data_dict)
    return combination_data_dict


def comp_num_seg_out_of_p_sent_beam(_filtered_doc_list,
                                    _num_edu,
                                    _absas_read_str,
                                    abs_as_read_list,
                                    map_from_new_to_ori_idx,
                                    beam_sz=8):
    beam = []
    if len(_filtered_doc_list) < _num_edu:
        return {"nlabel": _num_edu,
                "data": {},
                "best": None
                }

    combs = list(range(1, len(_filtered_doc_list)))
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
                average_f_score = get_rouge_est_str_2gram_smart(_absas_read_str, _tmp)
                # average_f_score = get_rouge_est_str_2gram(_absas_read_str, _tmp)

                leaderboard[to_add] = average_f_score
            sorted_beam = [(k, leaderboard[k]) for k in sorted(leaderboard, key=leaderboard.get, reverse=True)]

            for it in sorted_beam:
                new_in = already_in_beam + [it[0]]

                sorted_new_in = sorted(new_in)
                str_new_in = [str(x) for x in sorted_new_in]
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
        # n_comb_original.sort()  # json label
        n_comb_original = [int(x) for x in n_comb_original]
        # print(n_comb_original)
        _tmp = assemble_doc_list_from_idx(_filtered_doc_list, n_comb)
        # score = rouge_protocol([[_tmp]], [[abs_as_read_list]])
        _tmp = '\n'.join(_tmp)
        f1 = get_rouge_est_str_2gram_smart(_absas_read_str, _tmp)

        _comb_bag[f1] = {"label": n_comb_original,
                         "R1": f1,
                         "nlabel": _num_edu,
                         "sent":_tmp}
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


import multiprocessing, os, hashlib


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


def move_file_to_dir_url(url_file, path_read, file_to_write):
    with open(url_file, 'r', encoding='utf-8') as fd:
        lines = fd.read().splitlines()
        url_names = get_url_hashes(lines)
        print("len of urls {}".format(len(url_names)))

    url_names = [os.path.join(path_read, url) for url in url_names]
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    rt_bag = pool.map(read_one_file, url_names)

    pool.close()
    pool.join()

    rt_bag = [x for x in rt_bag if x is not None]
    wt_string = "\n".join(rt_bag)
    with open(file_to_write, 'w') as fd:
        fd.write(wt_string)


def move_file_to_dir_file_name(file_list, path_read, file_to_write):
    # with open(url_file, 'r', encoding='utf-8') as fd:
    #     lines = fd.read().splitlines()
    #     url_names = get_url_hashes(lines)
    #     print("len of urls {}".format(len(url_names)))
    print("Len of file list: {}".format(len(file_list)))
    url_names = [os.path.join(path_read, url) for url in file_list]
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    rt_bag = pool.map(read_one_file, url_names)

    pool.close()
    pool.join()

    rt_bag = [x for x in rt_bag if x is not None]
    wt_string = "\n".join(rt_bag)
    with open(file_to_write, 'w') as fd:
        fd.write(wt_string)


def read_one_file(fname):
    if os.path.isfile(fname + '.data'):
        with open(fname + '.data', 'r') as fd:
            line = fd.read().splitlines()
            assert len(line) == 1
        return line[0]
    else:
        return None
