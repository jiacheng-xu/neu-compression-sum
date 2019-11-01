import os
import shutil
import re
from typing import List
from neusum.evaluation.rough_rouge import get_rouge_est_str_4gram

top_k_combo = 5
P_SENT = 10
NUM_EDU = [2, 3, 4, 5]
from neusum.data.generate_oracle_with_dplp import comp_num_seg_out_of_p_sent_beam
import numpy


def clear_dir(_path):
    if os.path.isdir(_path):
        shutil.rmtree(_path)
    os.mkdir(_path)


class TreeNode():
    def __init__(self, tag, text, children, depth):
        self.text = text
        self.tag = tag
        self.children = children
        self.depth = depth
        self.start_idx = -1
        self.end_idx = -1

    def __repr__(self):
        return "Text: {}\tTag:{}\tDepth:{}".format(" ".join(self.text), self.tag, self.depth)


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


def read_single_parse_tree(inp_str) -> TreeNode:
    inp_str = inp_str.replace("\n", "")
    # inp_str = re.split('\(|\)|\s', inp_str)
    inp_str = re.sub(' +', ' ', inp_str)
    # inp = inp.split(" ")
    out = parse_subtree(inp_str, depth=0)
    out = add_idx_of_tree(out, start_idx=0, end_idx=len(out.text))
    return out


def find_deletable_span_rule_based(rule, tree: TreeNode) -> List[TreeNode]:
    # Rule is a list with TAG names
    # return a list of TreeNodes which are deletable
    tag = tree.tag
    deletable_bag = []
    if tree.children is not None:
        for child in tree.children:
            deletable_bag += find_deletable_span_rule_based(rule, child)
    if tag in rule:
        deletable_bag += [tree]
    if ' '.join(tree.text) == '-LRB- CNN -RRB-':
        deletable_bag += [tree]
    return deletable_bag


import json


def generate_span_segmentation(doc_list: List[str]) -> List[int]:
    span = []
    jdx = 0
    for d in doc_list:
        num = len([x for x in d.split(' ') if x != ''])
        span.append(jdx)
        span.append(jdx + num - 1)
        jdx += num
    return span


def extract_parse(snlp_dict) -> List[str]:
    sentences = snlp_dict['sentences']
    buff_parse = [s['parse'] for s in sentences]
    return buff_parse


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


def comp_oracle_delete_one_unit(tree: TreeNode, del_spans: List, topk):
    del_spans.sort(key=lambda x: x['rouge'], reverse=True)
    del_spans = del_spans[:topk]
    most_deletable = del_spans[0]  # {[sidx][eidx][node][rouge]}
    return del_spans, most_deletable


def BFS(block_map, current_member: List[int]):
    combinations = []
    if current_member == []:
        cur_max = -1
    else:
        cur_max = max(current_member)
        combinations.append(current_member)

    member_set = set(current_member)
    l = len(block_map)
    for i in range(cur_max + 1, l):
        if i in member_set:
            continue  # already in
        b = block_map[i]
        friend = all([b[m] for m in current_member])
        if not friend:
            continue  # not compatible with exsisting
        combinations += BFS(block_map, current_member + [i])
    return combinations


def comp_oracle_delete_multiple_unit(abs_str: str, tree: TreeNode, del_spans: List, topk):
    full_text_list = tree.text
    del_spans.sort(key=lambda x: x['rouge'], reverse=True)
    del_spans = del_spans[:topk]
    l = len(del_spans)
    block_map = [[True] * l for _ in range(l)]
    for idx, sp in enumerate(del_spans):
        s = sp['sidx']
        e = sp['eidx']
        for jdx in range(idx, l):
            _s, _e = del_spans[jdx]['sidx'], del_spans[jdx]['eidx']
            if _s >= e or s >= _e:
                pass
            else:
                block_map[idx][jdx] = False
                block_map[jdx][idx] = False
    possible_combinations = BFS(block_map, [])
    del_multi_units = []
    for p in possible_combinations:
        full_set = set(range(len(full_text_list)))
        for x in p:
            this_start = del_spans[x]["sidx"]
            this_end = del_spans[x]["eidx"]
            this_set = set(range(this_start, this_end))
            full_set = full_set - this_set

        full_list = list(full_set)
        full_list.sort()
        text_chunks = [full_text_list[w] for w in full_list]
        _rouge = get_rouge_est_str_4gram(gold=abs_str, pred=" ".join(text_chunks))
        tmp = {"del_spans": [del_spans[x] for x in p], "rouge": _rouge, "selected_idx": full_list}
        del_multi_units.append(tmp)
    del_multi_units.sort(key=lambda x: x["rouge"], reverse=True)
    del_multi_units = del_multi_units[:topk]
    most_trash_multi_unit = del_multi_units[0]
    # return [{"del_spans", rouge}]
    return del_multi_units, most_trash_multi_unit


def gen_span_segmentation(doc_list: List[List]):
    # generate inclusive span representation
    point = 0
    span_pairs = []
    for d in doc_list:
        l = len(d)
        span_pairs.append([point, point + l - 1])
        point += l
    return span_pairs


def convert_document_to_read_ready_string(path_read, path_write,
                                          fname_without_suffix: str
                                          ):
    doc_file = os.path.join(path_read, fname_without_suffix + ".doc.json")
    abs_file = os.path.join(path_read, fname_without_suffix + ".abs.json")
    if (not os.path.isfile(doc_file)) or (not os.path.isfile(abs_file)):
        raise TypeError
    with open(doc_file, 'r') as fd:
        doc_str = fd.read()
    with open(abs_file, 'r') as fd:
        abs_str = fd.read()

    doc_dict = json.loads(doc_str)
    abs_dict = json.loads(abs_str)
    doc_parse = extract_parse(doc_dict)
    abs_token, abs_str = extract_tokens(abs_dict)

    rt_sentences = []

    dft = {"sidx": 0,
           "eidx": 1,
           "node": "BASELINE",
           "rouge": 0,
           "selected_idx": [0]}
    # <SOS> Used for pred the end of decoding
    sent_sos_dict = {"token": ["<SOS>"],
                     "del_span": [dft],
                     "1-del": [dft], "n-del": [dft], "n-del-best": dft, "1-del-best": dft}

    rt_sentences.append(sent_sos_dict)

    for sent_parse in doc_parse:

        sent_tree = read_single_parse_tree(sent_parse)
        tree_len = len(sent_tree.text)
        rt_del_spans = []
        del_spans = find_deletable_span_rule_based(rules, sent_tree)

        # List of tree nodes
        for del_sp in del_spans:
            full_set = set(range(len(sent_tree.text)))
            selected_set = list(full_set - set(range(del_sp.start_idx, del_sp.end_idx)))
            selected_set.sort()
            text_left = sent_tree.text[0:del_sp.start_idx] + sent_tree.text[del_sp.end_idx:]
            _txt = " ".join(text_left)
            _rouge1 = get_rouge_est_str_4gram(gold=abs_str, pred=_txt)
            # to prevent nothing to compress, always add the whole sentence itself to the del list?
            if len(selected_set) > 2:  # TODO you have to keep something
                rt_del_spans.append({"sidx": del_sp.start_idx,
                                     "eidx": del_sp.end_idx,
                                     "node": del_sp.tag,
                                     "rouge": _rouge1,
                                     "selected_idx": selected_set})
        rt_del_spans.append({"sidx": tree_len - 1,
                             "eidx": tree_len,
                             "node": "BASELINE",
                             "rouge": get_rouge_est_str_4gram(gold=abs_str, pred=" ".join(sent_tree.text[:-1])),
                             "selected_idx": list(range(tree_len - 1))})
        # rt_del_spans.append({"sidx": sent_tree.start_idx,
        #                      "eidx": sent_tree.end_idx,
        #                      "node": "ALL",
        #                      "rouge": 0,
        #                      "selected_idx": []})

        del_multi_units, most_trash_multi_unit = comp_oracle_delete_multiple_unit(abs_str, sent_tree, rt_del_spans,
                                                                                  topk)  # delete multi and best delete multi

        # delete 1 and best delete 1
        del_single_units, most_trash_single_unit = comp_oracle_delete_one_unit(sent_tree, rt_del_spans, topk)

        sent_pack = {"token": sent_tree.text, "del_span": rt_del_spans,
                     "1-del": del_single_units, "1-del-best": most_trash_single_unit,
                     "n-del": del_multi_units, "n-del-best": most_trash_multi_unit
                     }
        rt_sentences.append(sent_pack)
    # print("Finding Oracle ...")
    # Sentence Level Oracle
    doc_list_for_oracle = [" ".join(x['token']) for x in rt_sentences]
    sent_ora_json = comp_document_oracle(doc_list_for_oracle, abs_str)
    #

    # Subsentence Level Oracle
    doc_list_trimmed_for_oracle = []
    for x in rt_sentences:
        doc_list_trimmed_for_oracle.append(" ".join([x["token"][kdx] for kdx in x['1-del-best']['selected_idx']]))
    trim_ora_json = comp_document_oracle(doc_list_trimmed_for_oracle, abs_str)
    # print("Oracle found")
    # Return the datapack
    rt = {}
    rt["name"] = fname_without_suffix
    span_pairs = gen_span_segmentation([x['token'] for x in rt_sentences])
    rt["abs"] = abs_str
    rt["abs_token"] = abs_token
    rt["token"] = " ".join(doc_list_for_oracle)
    rt["token_list"] = [x['token'] for x in rt_sentences]
    rt["span"] = span_pairs
    rt["sentences"] = rt_sentences
    rt["sent_oracle"] = sent_ora_json
    rt["sent_oracle_trim"] = trim_ora_json

    # output format: document{ name:str,
    # '<SOS>' + @@SS@@'.join tokens_all, sentence segmentation span,
    # sentences:[
    # start idx,
    # end idx,
    # token list,
    # deletable spans:[relative startidx in the sentence, relative endidx, nodetype, rouge(slot kept)],
    # what if nothing deletable?  always add the the whole sentence
    # top k  trimed version (multiple oracle) [ combination of deletable spans]
    # oracle: sentence level oracle (n of n)
    #           trimed sentence level (n of n)
    # TODO compression version: [ tokens+license+rouge(slot), tokens+license+rouge,... ]
    # TODO or trash candidate : [tokens+rouge,...]
    #  ] }
    json_rt = json.dumps(rt)
    with open(os.path.join(path_write, fname_without_suffix + '.data'), 'w') as fd:
        fd.write(json_rt)


def comp_document_oracle(doc_list: List, abs_str, beam_sz=5):
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
        f1 = get_rouge_est_str_4gram(gold='\n'.join(abs_as_readable_list),
                                     pred=doc_as_readable_list[i])

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


def find_deletable_span_ungrammar(merge_size, depth, tree: TreeNode):
    un_unified_bag = []
    if tree.depth < depth:
        if tree.children is not None:
            for child in tree.children:
                un_unified_bag += find_deletable_span_ungrammar(merge_size, depth, child)

        else:
            # Reach leaf node
            return [tree]
    elif tree.depth == depth:
        return [tree]
    else:
        raise NotImplementedError
    if tree.depth != 0:
        return un_unified_bag
    unified_bag = []
    buff = []
    buff_node = []
    buff_l = 0
    buff_start = -1
    buff_end = -1
    for item in un_unified_bag:
        if buff_start == -1:
            buff_start = item.start_idx
        buff_end = item.end_idx
        l = item.end_idx - item.start_idx
        buff_l += l
        buff += item.text
        buff_node.append(item.tag)
        if buff_l > merge_size:
            newnode = TreeNode(tag="-".join(buff_node), text=buff,
                               children=None, depth=depth)
            newnode.start_idx = buff_start
            newnode.end_idx = buff_end
            unified_bag.append(newnode)
            buff = []
            buff_node = []
            buff_l = 0
            buff_start = -1
    if buff != []:
        newnode = TreeNode(tag="-".join(buff_node), text=buff,
                           children=None, depth=depth)
        newnode.start_idx = buff_start
        newnode.end_idx = buff_end
        assert buff_end > buff_start
        unified_bag.append(newnode)
    return unified_bag


import sys
import os
import multiprocessing
import hashlib


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


def read_one_file(fname):
    if os.path.isfile(fname + '.data'):
        with open(fname + '.data', 'r') as fd:
            line = fd.read().splitlines()
            assert len(line) == 1
        return line[0]
    else:
        return None


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


import random

if __name__ == '__main__':
    gram = sys.argv[1]
    full_dataname = 'cnn'
    if str.lower(gram) == 'true':
        grammar = True
    else:
        grammar = False
    print("grammar: {}".format(grammar))
    path_to_snlp_output = "/home/cc/snlp-output-cnn"
    path_to_write_processed_dataset = "/home/cc/distributed_read_ready-grammar{}".format(grammar)

    clear_dir(path_to_write_processed_dataset)
    # rules = ["PP", "SBAR", "ADVP", "ADJP", "S"]
    rules = ["PP", "SBAR", "ADVP", "ADJP"]

    files = [i.split(".")[0] for i in os.listdir(path_to_snlp_output) if i.endswith('.doc.json')]
    total_num = len(files)
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    pool.starmap(convert_document_to_read_ready_string, zip([path_to_snlp_output] * total_num,
                                                            [path_to_write_processed_dataset] * total_num,
                                                            files
                                                            # ,
                                                            # [grammar] * total_num,
                                                            # [rules] * total_num
                                                            ))
    pool.close()
    pool.join()

    # read snlp parse tree
    # figure out deletable unit
    # generate all compression version with length and rouge
    # document: {sentence:{full_text, deletable_bag, compression version }}
    path_to_categories_data = "/home/cc/read_ready-grammar{}".format(grammar)
    clear_dir(path_to_categories_data)

    train_urls = '/home/cc/cnn-dailymail/url_lists/{}_wayback_training_urls.txt'.format(full_dataname)
    dev_urls = '/home/cc/cnn-dailymail/url_lists/{}_wayback_validation_urls.txt'.format(full_dataname)
    test_urls = '/home/cc/cnn-dailymail/url_lists/{}_wayback_test_urls.txt'.format(full_dataname)

    move_file_to_dir_url(url_file=train_urls, path_read=path_to_write_processed_dataset,
                         file_to_write=os.path.join(path_to_categories_data, 'train.txt'))

    move_file_to_dir_url(url_file=dev_urls, path_read=path_to_write_processed_dataset,
                         file_to_write=os.path.join(path_to_categories_data, 'dev.txt'))

    move_file_to_dir_url(url_file=test_urls, path_read=path_to_write_processed_dataset,
                         file_to_write=os.path.join(path_to_categories_data, 'test.txt'))
