import json
from typing import Dict
import allennlp
import numpy as np
from allennlp.data.vocabulary import Vocabulary
from typing import Dict, List, Any, Tuple
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from typing import Iterable, Iterator, Callable
import logging
from allennlp.data.instance import Instance
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, SpanField, ListField, LabelField, \
    IndexField, ArrayField
import numpy
import random
import os
import multiprocessing

root = "/backup3/jcxu/data"
cnn_dir = "2merge-cnn"
dm_dir = "2merge-dm"

dm_train = os.path.join(root, dm_dir, 'train.txt')
dm_dev = os.path.join(root, dm_dir, 'dev.txt')
dm_test = os.path.join(root, dm_dir, 'test.txt')

cnn_train = os.path.join(root, cnn_dir, 'train.txt')
cnn_dev = os.path.join(root, cnn_dir, 'dev.txt')
cnn_test = os.path.join(root, cnn_dir, 'test.txt')
word_token_indexers = {"tokens": SingleIdTokenIndexer()}


def read_one(file_path):
    instances = []
    with open(file_path, 'r') as fd:
        for line in fd:
            data_dict = json.loads(line)
            doc_str = data_dict['doc']
            allen_token_word_in_doc = TextField([Token(word) for word in doc_str.split()], word_token_indexers)
            instances.append(Instance({"text": allen_token_word_in_doc}))
    return instances


out = read_one(dm_test) + read_one(cnn_test) + read_one(dm_dev) + read_one(cnn_dev) +read_one(dm_train) + read_one(cnn_train)
vocab = Vocabulary.from_instances(out, min_count={'tokens': 80})
print(vocab.get_vocab_size())
vocab.save_to_files(root+"/cnndm_vocab_sm")
print(root+"/cnndm_vocab_sm")
# out = read_one(dm_train) + read_one(cnn_train)
# vocab = Vocabulary.from_instances(out, min_count={'tokens': 14})
# print(vocab.get_vocab_size())
# vocab.save_to_files(root+"/cnndm_vocab_lg")
