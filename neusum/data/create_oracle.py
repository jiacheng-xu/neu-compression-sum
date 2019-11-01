import os
import shutil
import re
from typing import List
from neusum.evaluation.rough_rouge import get_rouge_est_str_4gram
import sys
from neusum.service.basic_service import clear_dir
import json
import itertools
from neusum.data.generate_compression_based_data import comp_document_oracle
from neusum.data.generate_compression_based_data import move_file_to_dir_url

topk = 10


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


def split_branch_of_cc(tree: TreeNode) -> List:
    # where is cc
    left, right = 0, 0
    seg_idx = 0
    for idx, c in enumerate(tree.children):
        if c.tag == 'CC':
            left = c.start_idx
            right = c.end_idx
            seg_idx = idx
            break
    left_tree = TreeNode(tag="CC_branch", text=list(itertools.chain([child for child in tree.children[:seg_idx + 1]])),
                         children=tree.children[:seg_idx + 1], depth=tree.depth)
    left_tree.start_idx = tree.start_idx
    left_tree.end_idx = right
    right_tree = TreeNode(tag="CC_branch", text=list(itertools.chain([child for child in tree.children[seg_idx:]])),
                          children=tree.children[seg_idx:], depth=tree.depth)
    right_tree.start_idx = left
    right_tree.end_idx = tree.end_idx
    return [left_tree, right_tree]


def comp_oracle_delete_one_unit(tree: TreeNode, del_spans: List, topk):
    del_spans.sort(key=lambda x: x['rouge'], reverse=True)
    del_spans = del_spans[:topk]
    most_deletable = del_spans[0]  # {[sidx][eidx][node][rouge]}
    return del_spans, most_deletable


def find_deletable_span_rule_based(rule, tree: TreeNode) -> List[TreeNode]:
    # Rule is a list with TAG names
    # return a list of TreeNodes which are deletable
    tag = tree.tag
    deletable_bag = []
    if tree.children is not None:
        # CC
        flag_cc = any([c.tag == 'CC' for c in tree.children])
        if flag_cc:
            deletable_bag += split_branch_of_cc(tree)
        for child in tree.children:
            deletable_bag += find_deletable_span_rule_based(rule, child)
    if tag in rule:
        deletable_bag += [tree]
    if ' '.join(tree.text) == '-LRB- CNN -RRB-':
        deletable_bag += [tree]

    return deletable_bag


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


def convert_document_to_read_ready_string(path_read, path_write,
                                          fname_without_suffix: str,
                                          grammar, rules=None, max_sent=40, data_name='dm', merge_sz=5, depth=3,
                                          topk=10,
                                          set_of_del=[1, 2]):
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
    doc_parse = extract_parse(doc_dict)[:max_sent]
    abs_token, abs_str = extract_tokens(abs_dict)

    rt_sentences = []

    # dft = CompressionSpan(sidx=0, eidx=1, node="BASELINE", rouge=0, selected_idx=[])
    dft = {"sidx": 0, 'eidx': 1, 'node': "BASELINE", 'rouge': 0, 'selected_idx': []}
    # <SOS> Used for pred the end of decoding
    # sent_sos_dict = SentDataWithOracle(token=["<SOS>"],del_span= [dft],single_del=[dft],single_del_best=dft)
    sent_sos_dict = {'token': ["<SOS>"], 'del_span': [dft], 'single_del': [dft], 'single_del_best': dft}

    rt_sentences.append(sent_sos_dict)

    for sent_parse in doc_parse:

        sent_tree = read_single_parse_tree(sent_parse)
        tree_len = len(sent_tree.text)
        rt_del_spans = []
        del_spans = find_deletable_span_rule_based(rules, sent_tree)

        # List of tree nodes
        for del_sp in del_spans:
            if len(del_sp.text) < 2:  # ASSUM
                continue
            full_set = set(range(len(sent_tree.text)))
            selected_set = list(full_set - set(range(del_sp.start_idx, del_sp.end_idx)))
            selected_set.sort()
            text_left = sent_tree.text[0:del_sp.start_idx] + sent_tree.text[del_sp.end_idx:]
            _txt = " ".join(text_left)
            _rouge1 = get_rouge_est_str_4gram(gold=abs_str, pred=_txt)
            # to prevent nothing to compress, always add the whole sentence itself to the del list?
            if len(selected_set) >= 2:  # TODO you have to keep something   ASSUM
                # rt_del_spans.append(CompressionSpan(sidx=del_sp.start_idx, eidx=del_sp.end_idx,
                #                                     node=del_sp.tag, rouge=_rouge1, selected_idx=selected_set))
                rt_del_spans.append(
                    {'sidx': del_sp.start_idx, 'eidx': del_sp.end_idx, 'node': del_sp.tag, 'rouge': _rouge1,
                     'selected_idx': selected_set})
        rt_del_spans.append({'sidx': tree_len - 1, 'eidx': tree_len,
                             'node': "BASELINE", 'rouge': get_rouge_est_str_4gram(
                gold=abs_str, pred=" ".join(sent_tree.text[:-1])), 'selected_idx': list(range(tree_len - 1))})

        # del_multi_units, most_trash_multi_unit = comp_oracle_delete_multiple_unit(abs_str, sent_tree, rt_del_spans,
        #                                                                           topk)  # delete multi and best delete multi

        # delete 1 and best delete 1
        del_single_units, most_trash_single_unit = comp_oracle_delete_one_unit(sent_tree, rt_del_spans, topk)
        # TODO
        # sent_pack = SentDataWithOracle(sent_tree.text, del_span=rt_del_spans, single_del=del_single_units,
        #                                single_del_best=most_trash_single_unit)
        sent_pack = {"token": sent_tree.text, "del_span": rt_del_spans,
                     "single_del": del_single_units, "single_del_best": most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack)
    # print("Finding Oracle ...")
    # Sentence Level Oracle
    doc_list = [" ".join(x['token']) for x in rt_sentences]
    sent_ora_json = comp_document_oracle(doc_list, abs_str)
    #

    # Subsentence Level Oracle
    # doc_list_trimmed_for_oracle = []
    # for x in rt_sentences:
    #     doc_list_trimmed_for_oracle.append(" ".join([x['token'][kdx] for kdx in x['single_del_best']['selected_idx']]))
    # trim_ora_json = comp_document_oracle(doc_list_trimmed_for_oracle, abs_str)
    # print("Oracle found")
    # Return the datapack
    rt = {}
    rt["name"] = fname_without_suffix
    rt['part'] = data_name
    # span_pairs = gen_span_segmentation([x['token'] for x in rt_sentences])
    rt["abs"] = abs_str
    rt["abs_list"] = abs_token
    rt["doc"] = " ".join(doc_list)
    rt["doc_list"] = [x['token'] for x in rt_sentences]
    rt["sentences"] = rt_sentences

    # rt["sent_oracle"] = trim_ora_json
    rt["sent_oracle"] = sent_ora_json
    # rt["sent_oracle_trim"] = trim_ora_json

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
    out = parse_subtree(inp_str, depth=0)
    out = add_idx_of_tree(out, start_idx=0, end_idx=len(out.text))
    return out


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


import multiprocessing

if __name__ == '__main__':
    print("arg1: grammar: true or false; beam size (number of samples): 3, 5, 10, ...; max sent")
    full_dataname = 'dailymail'
    brief_dataname = 'dm'
    root = "/home/cc/"
    # root = "/backup2/jcxu/data/"
    mini = False

    gram = sys.argv[1]
    if str.lower(gram) == 'true':
        grammar = True
    else:
        grammar = False

    beam = int(sys.argv[2])

    max_sent = int(sys.argv[3])
    print("grammar: {}".format(grammar))
    print("max sent: {}".format(max_sent))
    path_to_snlp_output = root + "snlp-output-{}".format(brief_dataname)

    path_to_write_processed_dataset = root + "distributed{}-gram{}-mini{}-beam{}-maxsent{}".format(brief_dataname,
                                                                                                   grammar,
                                                                                                   mini,
                                                                                                   beam,
                                                                                                   max_sent)

    clear_dir(path_to_write_processed_dataset)
    # rules = ["PP", "SBAR", "ADVP", "ADJP", "S"]
    rules = ["PP", "SBAR", "ADVP", "ADJP"]

    files = [i.split(".")[0] for i in os.listdir(path_to_snlp_output) if i.endswith('.doc.json')]
    if mini:
        files = files[:200]
    # files = files[:100]
    total_num = len(files)
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    pool.starmap(convert_document_to_read_ready_string, zip([path_to_snlp_output] * total_num,
                                                            [path_to_write_processed_dataset] * total_num,
                                                            files, [grammar] * total_num,
                                                            [rules] * total_num,
                                                            [max_sent] * total_num,
                                                            [brief_dataname] * total_num))
    pool.close()
    pool.join()
    path_to_categories_data = root + "{}-gram{}-mini{}-maxsent{}-beam{}".format(brief_dataname, grammar, mini, max_sent,
                                                                                beam)
    clear_dir(path_to_categories_data)

    train_urls = root + 'cnn-dailymail/url_lists/{}_wayback_training_urls.txt'.format(full_dataname)
    dev_urls = root + 'cnn-dailymail/url_lists/{}_wayback_validation_urls.txt'.format(full_dataname)
    test_urls = root + 'cnn-dailymail/url_lists/{}_wayback_test_urls.txt'.format(full_dataname)

    move_file_to_dir_url(url_file=train_urls, path_read=path_to_write_processed_dataset,
                         file_to_write=os.path.join(path_to_categories_data, 'train.txt'))

    move_file_to_dir_url(url_file=dev_urls, path_read=path_to_write_processed_dataset,
                         file_to_write=os.path.join(path_to_categories_data, 'dev.txt'))

    move_file_to_dir_url(url_file=test_urls, path_read=path_to_write_processed_dataset,
                         file_to_write=os.path.join(path_to_categories_data, 'test.txt'))
