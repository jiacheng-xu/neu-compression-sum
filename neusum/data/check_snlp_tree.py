path = "/backup3/jcxu/data/sample-snlp-output"
from neusum.orac.util import extract_tokens, extract_parse, read_single_parse_tree



from typing import List
import json


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


# {} deletion
# [] optional


# (PP 3178
# (PP (VB? VBG VBN 44   always safe to delete
# (PP (TO 298       don't del if under ADJP, ADVP or PP
# (PP (IN 2801
# (PP (IN of  683   don't del if under ADJP or NP or ADVP
# (PP (IN in 593
# (PP (IN on 191
# (PP (IN with 202
# (PP (IN from 138
# (PP (IN from 104
# (PP (IN by 108
# (PP (IN as 59     don't del if under VP or ADVP

# JJ or [JJ , JJ] or [JJ CC JJ] under NP before NNP or NN or NNS or NNS


RULES = {"RB": "{-LRB * -RRB}",  # done
         "PRN": "PRN",  # done
         "CCS": "{ S [,] (CC but/and) } (S ..)",  # done
         # ":": "{{S} (  : , )} (S)",
         "VP->VBG": "(VP (VBG ) ) under NP or S (but S not under PP) or between split tokens",  # done
         "VP->VBN": "(VP (VBN ) ) under NP or between split tokens",  # done
         "SBAR": "between split tokens or first child=IN or WH* ",  # done
         "SBAR-sel": "(SBAR (IN) (S ))  keep the S here",  # done
         "JJ": "JJ or [JJ , .. , JJ] or [JJ CC JJ] under NP before NNP or NN or NNS NNPS ",  # done
         "(, ,)": "Between same depth split tokens, "
                  "SINGLE ADVP or SBAR or PP or [(VP)(NP) or (NP)(VP) ](len<8) or S(len<5)   or NP {, NP ,}",
         "PP": "Del PP unless 1) (PP(TO under ADJP ADVP or PP   2)(PP(IN of under ADJP NP ADVP "
               "3) PP(IN as   under VP or ADVP"}

SPLIT = [",", ".", ":"]  # and the head


def det_pp(node: TreeNode, parent: TreeNode):
    if node.tag == 'PP' and len(node.text) > 1:
        first_child = node.children[0]
        if parent is not None and ("to" in first_child.text) and (parent.tag in ["VP", "ADJP", "ADVP"]):
            # if ("to" in first_child.text) and (parent.tag in ["VP", "ADJP", "ADVP", "PP"]):       # conservative
            return []
        if parent is not None and ("as" in first_child.text) and (parent.tag in ["VP", "ADVP"]):
            return []
        if parent is not None and ("of" in first_child.text) and (parent.tag in ["ADJP", "ADVP"]):
            # if ("of" in first_child.text) and (parent.tag in ["NP", "ADJP", "ADVP"]):
            return []
        if parent is not None and parent.tag == 'VP' and parent.children[0].text in ["is", "was", "are", "were", "be"]:
            return []
        return [{"node": node.tag, "selected_idx": set(range(node.start_idx, node.end_idx)),
                 "text": node.text}]
    return []


def det_sbar(node: TreeNode, root_len, parent: TreeNode = None):
    # "SBAR": "between split tokens or first child=IN or WH* ",
    # "SBAR-sel": "(SBAR (IN) (S ))  keep the S here",
    bag = []
    if node.tag == 'SBAR' and node.children:
        if parent is not None and (parent.children[0].tag != 'VBG'):
            bag += [{"node": node.tag, "selected_idx": set(range(node.start_idx, node.end_idx)),
                     "text": " ".join(node.text)}]
        # if parent is not None and (parent.tag == "VP"):
        #     return []
        first_child = node.children[0]
        first_child_tag = first_child.tag
        # if first_child_tag.startswith("W") or first_child_tag == 'IN':
        #     bag += [{"node": node.tag, "selected_idx": set(range(node.start_idx, node.end_idx)),
        #              "text": " ".join(node.text)}]
        if first_child_tag == 'IN' and len(node.children) > 1:
            sec_child_tag = node.children[1].tag
            if sec_child_tag == "S":
                bag += [{"node": "_S_",
                         "selected_idx": set(range(root_len)) - set(
                             range(node.children[1].start_idx, node.children[1].end_idx)),
                         "kep_text": " ".join(node.children[1].text)}]
    return bag


def det_JJ(node: TreeNode):
    bag = []
    if node.tag == 'NP':
        if node.children:
            for idx, child in enumerate(node.children):
                if child.tag == "JJ" or child.tag == "ADJP":
                    bag += [{"node": child.tag, "selected_idx": set(range(child.start_idx, child.end_idx)),
                             "text": " ".join(child.text), "par_text": " ".join(node.text)}]
    return bag


def det_vp_vbg_vbn(tree: TreeNode, parent: TreeNode, grand_parent: TreeNode = None):
    if tree.tag == 'VP':
        if tree.children[0].tag == 'VBG':
            if parent and parent.tag == 'NP':
                return [{"node": 'VBG',
                         "selected_idx": set(range(tree.start_idx, tree.end_idx)),
                         "text": " ".join(tree.text),
                         "par_text": " ".join(parent.text)
                         }]
            if parent and parent.tag == 'S' and grand_parent and grand_parent.tag != "PP":
                return [{"node": 'VBG',
                         "selected_idx": set(range(tree.start_idx, tree.end_idx)),
                         "text": " ".join(tree.text),
                         "par_text": " ".join(parent.text)
                         }]
        if tree.children[0].tag == 'VBN':
            if parent and parent.tag == 'NP':
                return [{"node": 'VBN',
                         "selected_idx": set(range(tree.start_idx, tree.end_idx)),
                         "text": " ".join(tree.text),
                         "par_text": " ".join(parent.text)
                         }]
    return []


def det_advp(tree: TreeNode, parent: TreeNode):
    if tree.tag == 'ADVP':
        return [{"node": 'ADVP',
                 "selected_idx": set(range(tree.start_idx, tree.end_idx)),
                 "text": " ".join(tree.text)
                 }]
    return []


def det_np_np(tree: TreeNode):
    bag = []
    if not tree.children:
        return []
    flag = any([x.tag in SPLIT for x in tree.children])
    if not flag:
        return []
    child_seq = []
    for idx, child in enumerate(tree.children):
        if child.tag in SPLIT:
            child_seq.append("|")
        elif child.tag == 'NP':
            child_seq.append("NP")
        else:
            child_seq.append("*")
    for idx in range(len(tree.children) - 3):
        if child_seq[idx] == 'NP' and child_seq[idx + 2] == 'NP' and child_seq[idx + 1] == "|" and child_seq[
            idx + 3] == "|":
            bag += [{"node": "NP_np",
                     "selected_idx": set(range(tree.children[idx + 1].start_idx, tree.children[idx + 3].end_idx)),
                     "text": " ".join(
                         tree.children[idx + 1].text + tree.children[idx + 2].text + tree.children[idx + 3].text),
                     "par_text": " ".join(tree.text)
                     }
                    ]

            _txt = tree.children[idx + 2].text
            for x in _txt:
                if x.isupper():
                    bag += [{"node": "np_NP",
                             "selected_idx": set(
                                 range(tree.children[idx + 0].start_idx, tree.children[idx + 1].end_idx)).union(
                                 set(range(tree.children[idx + 3].start_idx, tree.children[idx + 3].end_idx)))
                                ,
                             "text": " ".join(
                                 tree.children[idx + 0].text + tree.children[idx + 1].text + [" ___"] + tree.children[
                                     idx + 3].text),
                             "par_text": " ".join(tree.text)
                             }
                            ]
                    break
    return bag


def det_between_split(tree: TreeNode, rootlen: int):
    # "(, ,)": "Between same depth split tokens, "
    # "SINGLE ADVP or SBAR or PP or [(VP)(NP) or (NP)(VP) ](len<8)
    # or S(len<5)   or NP {, NP ,}",

    def _easy_match(tag):
        if tag in ["ADVP", "SBAR", "PP"]:
            return True
        return False

    def _easy_match_len_limit(_node: TreeNode, tag="S", l=5):
        if tag == _node.tag and (len(_node.text) <= l):
            return True
        return False

    def _easy_two_step_match(_node: TreeNode, match_tag=["VP"], match_subtag=['VBG', 'VBN']):
        if _node.tag in match_tag:
            if _node.children:
                if _node.children[0].tag in match_subtag:
                    return True
        return False

    def _mix_np_vp(_node_a, _node_b):
        if (_node_a.tag == "NP") and (_node_b.tag == "VP") and len(_node_a.text + _node_b.text) <= 6:
            return True
            # return [{"node":"NP_VP",
            #          "selected_idx":
            #          }]
        if (_node_b.tag == "NP") and (_node_a.tag == "VP") and len(_node_a.text + _node_b.text) <= 6:
            return True
        return False

    bag = []
    if not tree.children:
        return []
    flag = any([x.tag in SPLIT for x in tree.children])
    if not flag:
        return []
    split_child_id = []
    for idx, child in enumerate(tree.children):
        if child.tag in SPLIT:
            split_child_id.append(idx)
    # split_child_id = [ 3, 6] __ __ __ , __ __ .

    last = -1
    # single rule check: VP (VBG)  VP(VBN) ADVP SBAR PP
    for idx_c, c in enumerate(split_child_id):
        if last == -1:
            cand_child = tree.children[:c]
        else:
            cand_child = tree.children[last + 1:c]
        last = c

        if len(cand_child) == 1:
            candi = cand_child[0]
            if _easy_match(candi.tag) or _easy_two_step_match(candi) or _easy_match_len_limit(candi):
                right, left = False, False
                if idx_c == 0:
                    right = True
                elif c == rootlen - 1:
                    left = True
                else:
                    right, left = True, True

                sel_idx = set()
                if left:
                    sel_idx = sel_idx.union(set(range(tree.children[split_child_id[idx_c - 1]].start_idx,
                                                      tree.children[split_child_id[idx_c - 1]].end_idx)))
                if right:
                    sel_idx = sel_idx.union(set(range(tree.children[split_child_id[idx_c]].start_idx,
                                                      tree.children[split_child_id[idx_c]].end_idx)))
                sel_idx = sel_idx.union(set(range(candi.start_idx, candi.end_idx)))
                bag += [{"node": "Split_" + candi.tag,
                         "selected_idx": sel_idx,
                         "text": " ".join(candi.text),
                         "par_text": " ".join(tree.text)}]

        elif len(cand_child) == 3:
            if cand_child[0].tag.startswith('\'') and _mix_np_vp(cand_child[1], cand_child[2]):
                bag += [{"node": "NPxVP",
                         "selected_idx": set(range(cand_child[1].start_idx, cand_child[2].end_idx)),
                         "text": " ".join(cand_child[1].text + cand_child[2].text),
                         "par_text": " ".join(tree.text)}]
    return bag
    # [(VP)(NP) or (NP)(VP)](len < 8) or S(len < 5) or NP
    # {, NP,}


def match_children_tag(tree: TreeNode, tag_name, word_name=None):
    """
    Match the tag of children. If found, return start_idx , end_idx and child_idx
    :param tree:
    :param tag_name:
    :return:
    """
    if tree.children is not None:
        for idx, child in enumerate(tree.children):
            if child.tag == tag_name:
                if word_name is None:
                    return child.start_idx, child.end_idx, idx
                elif word_name in child.text:
                    return child.start_idx, child.end_idx, idx
    return -1, -1, -1


def match_list_nodes(list_of_nodes, tag_name):
    pass


def det_ccs(tree: TreeNode, root_len):
    # { S [,] (CC but/and) } (S ..)
    sid, eid, child_idx = match_children_tag(tree, 'CC', word_name="and")
    if (sid >= 0 and eid >= 0 and child_idx >= 0):
        if len(tree.children) > child_idx + 1:
            if tree.children[child_idx + 1].tag == "S":
                return [{"node": "CCS", "selected_idx": set(range(root_len)) - set(range(
                    tree.children[child_idx + 1].start_idx, tree.children[child_idx + 1].end_idx)),
                         "kep_text": " ".join(tree.children[child_idx + 1].text)}]

    sid, eid, child_idx = match_children_tag(tree, 'CC', word_name="but")
    if (sid >= 0 and eid >= 0 and child_idx >= 0):
        if len(tree.children) > child_idx + 1:
            if tree.children[child_idx + 1].tag == "S":
                return [{"node": "CCS", "selected_idx": set(range(root_len)) - set(range(
                    tree.children[child_idx + 1].start_idx, tree.children[child_idx + 1].end_idx)),
                         "kep_text": " ".join(tree.children[child_idx + 1].text)}]
    return []


def det_rb(tree: TreeNode):
    if tree.tag == 'RB' and tree.text[0] in ['very', 'quite', "much", "also", "still", "just"]:
        return [{"node": "PRN", "selected_idx": set(range(tree.start_idx, tree.end_idx)),
                 "text": " ".join(tree.text)}]
    return []


def det_PRN(tree: TreeNode):
    if tree.tag == 'PRN':
        return [{"node": "PRN", "selected_idx": set(range(tree.start_idx, tree.end_idx)),
                 "text": " ".join(tree.text)}]
    return []


def det_RB(tree: TreeNode, parent: TreeNode):
    ls, le, lid = match_children_tag(tree, '-LRB-')
    if ls >= 0 and le >= 0:
        rs, re, rid = match_children_tag(tree, '-RRB-')
        if rs >= 0 and re >= 0:

            if ls <= 5 and 'CNN' in tree.text:
                return [{"node": "RB", "selected_idx": set(range(0, re)),
                         "text": " ".join(tree.text)}]
            else:
                return [{"node": "RB", "selected_idx": set(range(ls, re)),
                         "text": " ".join(tree.text)}]
    else:
        return []
    return []


def det_SS(tree: TreeNode):
    if tree.children:
        total_child_num = len(tree.children)
        first_child = tree.children[0]
        if total_child_num >= 3 and first_child.tag == "S":
            sec_child = tree.children[1]
            if sec_child.tag in SPLIT:
                print(" ".join(tree.text))

    return []


def find_deletable_span_rule_based_updated(tree: TreeNode,
                                           root_len: int,
                                           parent=None,
                                           grand_parent=None):
    next_parent = tree
    next_grandparent = parent

    deletable_bag = []
    deletable_bag += det_JJ(tree)
    deletable_bag += det_PRN(tree)
    deletable_bag += det_ccs(tree, root_len)
    deletable_bag += det_pp(node=tree, parent=parent)
    deletable_bag += det_sbar(node=tree, root_len=root_len, parent=parent)
    deletable_bag += det_vp_vbg_vbn(tree=tree, parent=parent, grand_parent=grand_parent)
    deletable_bag += det_np_np(tree)
    deletable_bag += det_RB(tree, parent)
    deletable_bag += det_between_split(tree, root_len)
    deletable_bag += det_advp(tree, parent)
    deletable_bag += det_rb(tree)
    # if len(deletable_bag) > 0:
    #     print(deletable_bag)
    if tree.children is not None:
        for idx, child in enumerate(tree.children):
            deletable_bag += find_deletable_span_rule_based_updated(child, root_len, parent=next_parent,
                                                                    grand_parent=next_grandparent)
    return deletable_bag


def read_one_file(dir, fp):
    with open(os.path.join(dir, fp), 'r') as fd:
        lines = fd.read()
    line_dict = json.loads(lines)
    doc_parse = extract_parse(line_dict)
    for sent_parse in doc_parse:
        # sent_parse = "(S (, ,) ('' '') (NP (NNP Armstrong))    (VP (VBD said))    (. .))"
        sent_tree = read_single_parse_tree(sent_parse)
        print(sent_parse)
        tree_len = len(sent_tree.text)
        # print(sent_tree)
        # continue
        del_bag = find_deletable_span_rule_based_updated(sent_tree, root_len=tree_len, parent=None, grand_parent=None)
        # exit()
        print(" ".join(sent_tree.text))
        print(del_bag)
        print('-' * 50)


import os


def remove_redundant_del(del_spans):
    d = {}
    new_list = []
    for del_sp in del_spans:
        txt = del_sp['text']
        if txt in d:
            continue
        else:
            new_list.append(del_sp)
            d[txt] = True
    return new_list


if __name__ == '__main__':
    # s="(ROOT\n  (S\n    (NP (PRP It))\n    (VP (VBZ 's)\n      (VP (VBN reported)\n        (SBAR (IN that)\n          (S\n            (NP\n              (NP (DT the) (JJ young) (NN bride))\n              (, ,)\n              (VP (VBN known)\n                (ADVP (RB only))\n                (PP (IN as)\n                  (NP (NNP Wang))))\n              (, ,))\n            (VP (VBZ is)\n              (ADVP (RB also))\n              (ADJP (RB mentally) (JJ impaired)))))))\n    (. .)))"
    # print(s)
    # a = "(ROOT\n  (S\n    (ADJP (JJ Desperate)\n      (PP (IN for)\n        (NP (DT a) (NN grandchild))))\n    (, ,)\n    (NP\n      (NP (NN Xu))\n      (PRN (-LRB- -LRB-)\n        (VP (VBN pictured))\n        (-RRB- -RRB-)))\n    (VP (VBD bought)\n      (NP\n        (NP (DT a) (NN wife))\n        (PP (IN for)\n          (NP\n            (NP (PRP$ his) (NN son))\n            (, ,)\n            (SBAR\n              (WHNP (WP who))\n              (S\n                (VP (VBZ has)\n                  (S\n                    (VP (VBG learning)\n                      (NP (NNS difficulties)))))))))))\n    (. .)))"
    # print(a)
    # exit()

    sent_parse = [
        "(S (NP (NP (NNP Conley)) (PRN (-LRB- () (VP (VBN pictured) (PP (IN on) (NP (DT the) (NN right))) (, ,) (PP (IN with) (NP (DT a) (NN friend)))) (-RRB- )) )) (VP (VBD tried) (PRT (RP out)) (NP (JJ several) (JJ different) (NNS religions)) (PP (IN in) (NP (NN college))) (PP (IN before) (S (VP (VBG selecting) (NP (NNP Islam)))))) (. .))",
        "(S (`` ‘) (S (NP (PRP I)) (VP (VBD asked) (NP (PRP them)) (S (VP (TO to) (VP (VB check) (ADVP (RB again)) (SBAR (RB just) (IN before) (S (NP (PRP we)) (VP (VBD boarded))))))))) (, ,) (CC but) (S (NP (PRP they)) (ADVP (RB still)) (VP (VBD said) (SBAR (S (NP (PRP I)) (VP (VBD was) (PP (IN in) (NP (DT the) (JJ right) (NN place)))))))) (. .))",
        "(S (NP (DT The) (NN plane)) (VP (VBD was) (ADJP (RB quite) (JJ empty)) (NP (NNS anyway.’))))",
        "(S (PP (IN Despite) (NP (DT the) (NNS assurances) (SBAR (IN that) (S (NP (PRP she)) (VP (VBD was) (PP (IN on) (NP (DT the) (JJ right) (NN flight)))))))) (, ,) (NP (NP (NNP Stacey) (POS 's)) (NN nightmare) (NN situation)) (VP (VBD was) (VP (VBN confirmed) (SBAR (WHADVP (WRB when)) (S (NP (DT the) (NN captain)) (VP (VBD told) (NP (DT the) (NNS passengers)) (SBAR (S (NP (PRP they)) (VP (VBD were) (NP (NP (RB just) (CD one) (NN hour)) (PP (IN from) (S (VP (VBG touching) (PRT (RP down)) (PP (IN in) (NP (NNP England))))))))))))))) (. .))",
        "(S (S (VP (VBN Forced) (S (VP (TO to) (VP (VB apologise)))))) (: :) (S (NP (NN Budget) (NN airline) (NNP Jet2)) (VP (VBD put) (NP (NNP Stacey) (NNP McGuinness)) (PP (IN onto) (NP (DT the) (JJ wrong) (NN flight))) (PP (IN at) (NP (NNP Dalaman) (NNP Airport))))) (. .))",
        "(S (S (`` ‘) (S (NP (PRP I)) (VP (VBD got) (ADVP (RB back)) (PP (IN in) (NP (DT the) (NN end)))))) (, ,) (CC but) (S (S (S (NP (PDT all) (PRP$ my) (NNS clothes) (CC and) (NNS valuables)) (VP (VBP are) (PP (IN in) (NP (PRP$ my) (NN suitcase))) (SBAR (WHNP (WDT that)) (S (NP (PRP I)) (ADVP (RB still)) (VP (VBP have) (RB n't) (VP (VBD got) (ADVP (RB back)))))))) (, ,)) (CC and) (S (NP (DT the) (NN customer) (NNS services)) (VP (VBP are) (VP (VBG telling) (NP (PRP me)) (NP (NN nothing)))))) (. .) ('' '))",

    ]
    for s in sent_parse:
        sent_tree = read_single_parse_tree(s)
        tree_len = len(sent_tree.text)
        del_spans = find_deletable_span_rule_based_updated(sent_tree, root_len=tree_len, parent=None, grand_parent=None)
        del_spans = remove_redundant_del(del_spans)
        print(del_spans)
    # files = os.listdir(path)
    #
    # for f in files:
    #     read_one_file(path, f)
