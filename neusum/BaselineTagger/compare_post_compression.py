import json, pickle, os
import shutil

from allennlp.common.file_utils import cached_path
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from typing import Dict

from allennlp.predictors import SentenceTaggerPredictor
from overrides import overrides

from neusum.BaselineTagger.my_sentence_tagger import MySentenceTaggerPredictor

flatten = lambda l: [item for sublist in l for item in sublist]
from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.common.file_utils import cached_path
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training.metrics import CategoricalAccuracy
import torch, os
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

import random
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.modules.attention import LegacyAttention, DotProductAttention
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, weighted_sum

import numpy as np

import tempfile
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.data.iterators import DataIterator

from allennlp.commands.evaluate import evaluate
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.archival import load_archive
from typing import Any, Dict, List, Optional, Tuple

from allennlp.nn import InitializerApplicator, util, RegularizerApplicator


def find_original_sents(doc_list, pred_list):
    rt_strings = []
    outputs = []
    for pred in pred_list:

        pred = [x.lower() for x in pred if len(x) > 0]
        # if pred[-1].endswith(".") and len(pred[-1])>1:

        pred_set = set(pred)
        success = False
        for idx, doc_sent in enumerate(doc_list):
            doc_sent_set = [x.lower() for x in doc_sent if len(x) > 0]
            doc_sent_set = set(doc_sent_set)
            if len(pred_set) == 0:
                ratio = 0
            else:
                ratio = len(pred_set.intersection(doc_sent_set)) / len(pred_set)
            if ratio > 0.6:
                outputs.append(idx)
                success = True
                break
        ratio_thres = 0.5
        while not success:
            for idx, doc_sent in enumerate(doc_list):
                doc_sent_set = [x.lower() for x in doc_sent if len(x) > 0]
                doc_sent_set = set(doc_sent_set)
                if len(pred_set) == 0:
                    ratio = 0
                else:
                    ratio = len(pred_set.intersection(doc_sent_set)) / len(pred_set)
                if ratio > ratio_thres:
                    outputs.append(idx)
                    success = True
                    break
            ratio_thres -= .1
    rt_strings = ([doc_list[idx] for idx in outputs])
    return outputs, rt_strings


def write_list_of_strings_to_a_file(list_of_strs, file_name):
    if isinstance(list_of_strs[0], str):
        pass
    elif isinstance(list_of_strs[0], list):
        list_of_strs = [" ".join(x) for x in list_of_strs]
    else:
        raise TypeError
    with open(file_name, 'w')as fd:
        fd.write("\n".join(list_of_strs))
    print('Finish writing to {}'.format(file_name))


def match_original_sents(dataset, dataset_name, my_output, root, dir_sel_sents, dir_our_compression, dir_reference,
                         dir_lead3):
    with open(dataset, 'r') as fd:
        lines = fd.read().splitlines()
    raw_dataset = []
    for l in lines:
        d = json.loads(l)
        out = {
            'name': d['name'],
            'doc_list': d['doc_list'],
            'abs_list': d['abs_list']
        }
        raw_dataset.append(out)

    with open(my_output, 'rb') as fd:
        x = pickle.load(fd)
    pred, ref = x['pred'], x['ref']

    print(len(ref))
    print(len(raw_dataset))

    # try to match
    our_data = []
    for p, r in zip(pred, ref):
        r_tokens = [x.split(" ") for x in r[0]]
        toks = flatten(r_tokens)
        our_data.append([toks, p])

    for idx, data in enumerate(raw_dataset):
        print(data['name'])
        name = data['name'] + '.txt'
        flatten_abs = flatten(data['abs_list'])
        flatten_abs = [x.lower() for x in flatten_abs]
        flatten_abs = set(flatten_abs)
        predction = None

        for rate_thres in [0.7, 0.5, 0.3]:
            _pop_index = -1
            for jdx, candidate in enumerate(our_data):
                candidate_abs = set([x.lower() for x in candidate[0]])
                ratio = len(flatten_abs.intersection(candidate_abs)) / len(flatten_abs.union(candidate_abs))
                if ratio > rate_thres:
                    predction = candidate[1]
                    _pop_index = jdx
                    break
                # elif ratio > 0.5:
                #     print(candidate_abs)
                #     print(flatten_abs)
            if predction is not None:
                assert _pop_index != -1
                our_data.pop(_pop_index)
                break
        data['pred'] = predction
        raw_dataset[idx] = data
        # print(data['abs_list'])
        assert predction is not None
    #
    # exit()
    for data in raw_dataset:
        # print(r[0])
        print(data['abs_list'])

        print(data['name'])
        name = data['name'] + '.txt'

        ref = [" ".join(x) for x in data['abs_list']]
        p = data['pred']
        pred_tokens = [x.split(" ") for x in p]
        index, extraction_str = find_original_sents(data['doc_list'], pred_tokens)
        # lead3
        lead3_str = data['doc_list'][:3]
        write_list_of_strings_to_a_file(ref, os.path.join(root, dir_reference, name))
        write_list_of_strings_to_a_file(extraction_str, os.path.join(root, dir_sel_sents, name))
        write_list_of_strings_to_a_file(p, os.path.join(root, dir_our_compression, name))
        write_list_of_strings_to_a_file(lead3_str, os.path.join(root, dir_lead3, name))


from neusum.BaselineTagger.BLSTMSimpleTagger import PosDatasetReader
from allennlp.common.util import prepare_environment


def load_offshelf_compression_model(
        cuda_device=2,
        archive_file="/backup3/jcxu/exComp/tmp_expsc74o5pf7/model.tar.gz",
        weights_file="/backup3/jcxu/exComp/tmp_expsc74o5pf7/best.th",

):
    archive = load_archive(archive_file, cuda_device, "", weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()
    predictor = MySentenceTaggerPredictor(model, dataset_reader=PosDatasetReader())
    return model, predictor


def predict_compression(predictor, inp_batch_json, margin=0):
    output = predictor.predict_batch_json(inp_batch_json)
    rts = []
    comps = []
    for idx, out in enumerate(output):
        tag_logits = out['tag_logits']
        # tokens = out['tokens']
        tokens = inp_batch_json[idx]['sentence']
        original_len = len(tokens)
        rt = []
        for logit, tok in zip(tag_logits, tokens):
            del_score = logit[0]
            retain_score = logit[1]
            if del_score <= retain_score + margin:
                rt.append(tok.text)
        compressed_len = len(rt)
        rt = " ".join(rt)
        compression_rate = compressed_len / original_len
        comps.append(compression_rate)
        rts.append(rt)
    return rts, comps


from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


def compress(predictor, root, dir_sel_sents, tgt_dir, margin):
    print(tgt_dir)
    tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)
    path = os.path.join(root, dir_sel_sents)
    files = os.listdir(path)
    meta_comps = []
    for f in files:
        with open(os.path.join(path, f), 'r') as fd:
            lines = fd.read().splitlines()

        json_lines = [{"sentence": tokenizer.split_words(sent)} for sent in lines]
        output, comps = predict_compression(predictor, json_lines, margin)
        # print(comps)
        with open(os.path.join(root, tgt_dir, f), 'w') as fd:
            fd.write("\n".join(output))
        # print("Finish writing {}".format(f))
        meta_comps += comps
    print(sum(meta_comps) / len(meta_comps))


from pythonrouge.pythonrouge import Pythonrouge


def rouge(pred_dir, tgt_dir):
    preds = os.listdir(pred_dir)
    tgts = os.listdir(tgt_dir)
    pred_str_bag, ref_str_bag = [], []
    for p, t in zip(preds, tgts):
        with open(os.path.join(pred_dir, p), 'r') as fd:
            prediction = fd.read().splitlines()
        with open(os.path.join(tgt_dir, t), 'r') as fd:
            reference = fd.read().splitlines()
        pred_str_bag.append(prediction)
        ref_str_bag.append([reference])
    print('Finish reading')
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=pred_str_bag, reference=ref_str_bag,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=True, ROUGE_W=True,
                        ROUGE_W_Weight=1.2,
                        recall_only=False, stemming=True, stopwords=False,
                        word_level=True, length_limit=False, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5, default_conf=True)
    score = rouge.calc_score()
    print(score)

def grammarly(dir='/backup3/jcxu/exComp/xu-durrett-output/lead3'):
    files = os.listdir(dir)
    random.shuffle(files)
    files=files[:200]
    for f in files:
        with open(os.path.join(dir, f),'r') as fd:
            lines = fd.read().splitlines()
            print("\n".join(lines))
    exit()
if __name__ == '__main__':
    grammarly()
    data_file = "/backup3/jcxu/data/2merge-cnn/test.txt"
    # data_file = "/backup3/jcxu/data/2merge-dm/test.txt"
    f = "/backup3/jcxu/exComp/0.327,0.122,0.290-cnnTrue1.0-1True3-1093-cp_0.5"
    # f = "/backup3/jcxu/exComp/0.427,0.192,0.388-dmTrue10.0-1True3-10397-cp_0.7"
    # mkdir lead3  lstm-comp-lead3  lstm-comp-sel-sents  lstm-comp-sel-sents-del  lstm-comp-sel-sents-ret  ref  sel-sents  xu-compressions
    # rm -r lead3  lstm-comp-lead3  lstm-comp-sel-sents  lstm-comp-sel-sents-del  lstm-comp-sel-sents-ret  ref  sel-sents  xu-compressions
    root = "/backup3/jcxu/exComp/xu-durrett-output/"
    dir_sel_sents = 'sel-sents'
    dir_our_compression = 'xu-compressions'
    dir_reference = 'ref'
    dir_lead3 = 'lead3'
    dir_offshelf_compress_sel = 'lstm-comp-sel-sents'
    dir_offshelf_compress_sel_more_retain = 'lstm-comp-sel-sents-ret'
    dir_offshelf_compress_sel_more_del = 'lstm-comp-sel-sents-del'
    dir_offshelf_compress_lead3 = 'lstm-comp-lead3'

    # match_original_sents(data_file, 'cnn', f, root, dir_sel_sents, dir_our_compression, dir_reference, dir_lead3)

    model, predictor = load_offshelf_compression_model()
    # compress(predictor, root, dir_sel_sents, tgt_dir=dir_offshelf_compress_sel_more_retain, margin=2.3)
    # compress(predictor, root, dir_sel_sents, tgt_dir=dir_offshelf_compress_sel, margin=3.5)
    # compress(predictor, root, dir_sel_sents, tgt_dir=dir_offshelf_compress_sel_more_del, margin=2.7)

    rouge(os.path.join(root, dir_sel_sents), os.path.join(root, dir_reference))
    rouge(os.path.join(root, dir_our_compression), os.path.join(root, dir_reference))
    rouge(os.path.join(root, dir_offshelf_compress_sel_more_del), os.path.join(root, dir_reference))
    rouge(os.path.join(root, dir_offshelf_compress_sel), os.path.join(root, dir_reference))
    rouge(os.path.join(root, dir_offshelf_compress_sel_more_retain), os.path.join(root, dir_reference))

    # todo
    #  match the original sentence
    #  write down the original sentence in json format
    # write down lead3
    # run off shelf compressor and evaluate rouge
    #
