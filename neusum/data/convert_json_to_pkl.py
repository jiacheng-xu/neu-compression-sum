import os
import json
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, SpanField, ListField, LabelField, \
    IndexField, ArrayField, SequenceField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import numpy as np
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from typing import List
import random
from allennlp.data.vocabulary import Vocabulary
import pickle

name = 'cnn'
vocab = None


def set_data_name(_n, servername):
    global name
    name = _n
    global vocab
    print(name)
    if servername == 'titan':
        root = "/scratch/cluster/jcxu/exComp/"
    elif servername == 'eve':
        root = "/backup3/jcxu/exComp/"
    elif servername == 'cc':
        root = '/home/cc/exComp/'
    else:
        pass
    if name == 'nyt':

        vocab_path = root + "nyt_vocab"
    else:
        # vocab_path = "/home/cc/exComp/cnndm_vocab"
        vocab_path = root + "cnndm_vocab"
    # vocab_path = "/home/cc/nyt_vocab"
    # vocab_path = "/backup3/jcxu/data/cnndm_vocab"
    print("reading vocab {}".format(vocab_path))
    vocab = Vocabulary.from_files(vocab_path)


def read_sent_object(sentence, word_token_indexers):
    txt_list = sentence['token']
    sent_len = len(txt_list)
    del_span = sentence['del_span']
    # baseline = sentence['baseline']
    random.shuffle(del_span)
    # del_span = del_span[:25]  # TODO assum
    l = len(del_span)
    txt_token_obj = [Token(word) for word in txt_list]
    txt_field = TextField(txt_token_obj, word_token_indexers)
    txt_field.index(vocab)
    # l = len(del_span)
    # random.shuffle(del_span)
    bag_of_compression_bit = []
    bag_of_rouge = []
    bag_of_rouge_ratio = []
    bag_of_span_meta = []

    for index in range(l):
        sp = del_span[index]
        # sp['ratio'] = sp['rouge'] / baseline
        del_indexs = sp['selected_idx']
        blank = np.zeros(sent_len, dtype=np.int)
        for _idx in del_indexs:
            blank[_idx] = 1
        del_mask = ArrayField(array=blank)
        bag_of_compression_bit.append(del_mask)
        bag_of_rouge.append(sp['rouge'])
        bag_of_rouge_ratio.append(sp['ratio'])
        # bag_of_span.append(span)
        bag_of_span_meta.append(
            [sp['node'], sp['selected_idx'], sp['rouge']
                , sp['ratio']
             ])
    bag_of_rouge = ArrayField(np.asarray(bag_of_rouge))
    bag_of_rouge_ratio = ArrayField(np.asarray(bag_of_rouge_ratio))
    bag_of_compression_bit = ListField(bag_of_compression_bit)
    bag_of_span_meta = MetadataField(bag_of_span_meta)
    # sort rouge and get label
    scores = [x['rouge'] for x in del_span]
    r_idx = np.argsort(scores)[::-1]
    seq_label = LabelField(label=int(r_idx[0]), skip_indexing=True)

    return txt_token_obj, bag_of_rouge, bag_of_rouge_ratio, bag_of_compression_bit, bag_of_span_meta, seq_label


def read_doc_object(sentences: List, word_token_indexers):
    txt_token_objs, list_rouge, list_rouge_ratio, list_span, list_span_meta, list_seq_label = [], [], [], [], [], []
    assert len(sentences) != 0
    for s in sentences:
        txt_token_obj, bag_of_rouge, bag_of_rouge_ratio, bag_of_span, bag_of_span_meta, seq_label \
            = read_sent_object(s, word_token_indexers)
        # TextField, ArrayField, ListField, MetadataField
        txt_token_objs.append(txt_token_obj)
        list_rouge.append(bag_of_rouge)
        list_rouge_ratio.append(bag_of_rouge_ratio)
        list_span.append(bag_of_span)
        list_span_meta.append(bag_of_span_meta)
        list_seq_label.append(seq_label)
    # txt_token_obj = ListField(txt_token_objs)
    list_rouge = ListField(list_rouge)
    list_span = ListField(list_span)
    list_span_meta = ListField(list_span_meta)
    list_seq_label = ListField(list_seq_label)
    list_rouge_ratio = ListField(list_rouge_ratio)
    return txt_token_objs, list_rouge, list_rouge_ratio, list_span, list_span_meta, list_seq_label


def convert_one_line(line):
    data_dict = json.loads(line)
    sentences = data_dict['sentences']
    if len(sentences) <= 1:
        return None

    name = data_dict['name']
    doc_str = data_dict['doc']
    abs_str = data_dict['abs']
    # span_pairs = data_dict['span']
    doc_list = data_dict['doc_list']
    abs_list = data_dict['abs_list']
    # print(name)
    if "part" in data_dict:
        part = data_dict['part']
    else:
        part = 'cnn'
    sent_oracle = data_dict['sent_oracle']
    non_compression_sent_oracle = data_dict['non_compression_sent_oracle']
    word_token_indexers = {"tokens": SingleIdTokenIndexer()}
    instance_fields = {}
    txt_token_obj, list_rouge, list_rouge_ratio, list_span, list_span_meta, list_seq_label = read_doc_object(sentences,
                                                                                                             word_token_indexers)

    txt_field = ListField([TextField(obj, word_token_indexers) for obj in txt_token_obj])
    txt_field.index(vocab)
    # print(txt_field.get_padding_lengths())
    # txt_field.get_padding_lengths()
    # instance_fields["token"] = TextField(tokens,self.word_token_indexers)
    instance_fields["text"] = txt_field
    instance_fields["comp_rouge"] = list_rouge
    instance_fields["comp_msk"] = list_span
    instance_fields["comp_meta"] = list_span_meta
    instance_fields["comp_seq_label"] = list_seq_label
    instance_fields["comp_rouge_ratio"] = list_rouge_ratio

    # def edit_label_rouge(_label_list, _rouge_list):
    #     _label_list = [x for x in _label_list]
    #     _max_len = max([len(x) for x in _label_list])
    #     for idx, label in enumerate(_label_list):
    #         if len(label) < _max_len:
    #             _label_list[idx] = _label_list[idx] + [-1] * (_max_len - len(label))
    #     np_gold_label = np.asarray(_label_list, dtype=np.int)
    #     f = ArrayField(array=np_gold_label, padding_value=-1)
    #     r = ArrayField(np.asarray(_rouge_list, dtype=np.float32))
    #     return f, r

    instance_fields["metadata"] = MetadataField(
        {
            "doc_list": doc_list,
            "abs_list": abs_list,
            "name": name,
            "part": part
        })
    instance_fields['_sent_oracle'] = sent_oracle
    instance_fields['_non_compression_sent_oracle'] = non_compression_sent_oracle
    return instance_fields


import multiprocessing


def convert_json_file_to_pkl_dump(path, txt_fname, part: str):
    chunk_size = 5000
    # vocab = Vocabulary.from_files(vocab_path)
    # pkl_fname = os.path.join(path, txt_fname + '.pkl')
    with open(os.path.join(path, txt_fname + '.txt'), 'r') as fd:
        lines = fd.read().splitlines()

    glob_cnt = 0
    total_l = len(lines)
    assertion_cnt = 0
    l = total_l
    while l > 0:
        todo = lines[:chunk_size * 5]
        lines = lines[chunk_size * 5:]
        cnt = int(multiprocessing.cpu_count() / 4)
        pool = multiprocessing.Pool(processes=cnt)
        pairs = pool.map(convert_one_line, todo)
        pool.close()
        pool.join()
        pairs = [x for x in pairs if x is not None]
        print(pairs[4])
        this_l = len(pairs)
        import math
        num_of_files = math.ceil(this_l / chunk_size)

        for idx in range(num_of_files):
            print("Writing to disk")
            pkl_fname = os.path.join(path, txt_fname + '.pkl.{}.'.format(part) + str(glob_cnt) + str(idx))
            if idx == num_of_files - 1:
                wt_content = pairs[idx * chunk_size:]
            else:
                wt_content = pairs[idx * chunk_size: (idx + 1) * chunk_size]
            assertion_cnt += len(wt_content)
            f = open(pkl_fname, 'wb')
            pickle.dump(wt_content, f)
            f.close()
        glob_cnt += 1
        print("reaming {}".format(l))
        l = len(lines)
    print("Writing done.")
    # assert assertion_cnt == total_l
