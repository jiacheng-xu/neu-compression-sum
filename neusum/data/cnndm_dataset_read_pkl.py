import json
from typing import Dict
import allennlp
import pickle
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
import multiprocessing

from neusum.data.generate_compression_based_data import TreeNode
def filter_oracle_label(single_oracle, fix_edu_num, labels: Dict):
    """

    :param single_oracle:
    :param fix_edu_num:
    :param labels:
    :return: List[List], List[float]
    """
    if single_oracle:
        raise NotImplementedError
    else:
        if fix_edu_num > 0:
            rt_data, rt_rouge = [], []
            if str(fix_edu_num) in labels:
                cur_data = labels[str(fix_edu_num)]['data']
                for k, v in cur_data.items():
                    rt_data.append(v['label'])
                    rt_rouge.append(v['R1'])
        elif fix_edu_num == -1:
            fix_edu_num = 2  # just in case
            rt_data, rt_rouge = [], []
            for n, lab in labels.items():
                data = lab['data']
                for k, v in data.items():
                    rt_data.append(v['label'])
                    rt_rouge.append(v['R1'])
        else:
            raise NotImplementedError("Fix_edu_num == -1 or 2,3,4,5..")
        if not rt_rouge:
            return [[0] * fix_edu_num], [0.1]
        else:
            sort_idx = [i[0] for i in sorted(enumerate(rt_rouge), key=lambda x: x[1], reverse=True)]
            sort_rt_data = [rt_data[sort_idx[idx]] for idx in range(len(rt_data))]
            sort_rt_rouge = [rt_rouge[sort_idx[idx]] for idx in range(len(rt_rouge))]
            return sort_rt_data, sort_rt_rouge


import os


class SummarizationDatasetReaderPkl(DatasetReader):
    def __init__(self,
                 source_token_indexers: Dict[str, TokenIndexer] = None,  # word token indexer
                 dir: str = None,
                 lazy: bool = True,
                 single_oracle=True,
                 fix_edu_num=None,
                 trim_sent_oracle: bool = True,
                 vocab_path: str = "",
                 save_to: str = None,
                 dbg: bool = False
                 ) -> None:
        super().__init__(lazy)
        self.word_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.single_oracle = single_oracle
        self.fix_edu_num = fix_edu_num
        self.vocab = None
        self.trim_oracle = trim_sent_oracle
        self.build_vocab(vocab_path, save_to=save_to)
        self.dir = dir
        self.dbg = dbg

    def build_vocab(self, file_path, save_to: str = None):
        if os.path.isdir(save_to):
            try:
                self.vocab = Vocabulary.from_files(save_to)
                print("Load vocab success")
                logging.info("Vocab size: {}".format(int(self.vocab.get_vocab_size())))
                return
            except:
                print("Failed in loading pre-saved vocab.")
        raise NotImplementedError
        instances = []

        with open(file_path, 'r') as fd:
            for line in fd:
                data_dict = json.loads(line)
                doc_str = data_dict['doc']
                allen_token_word_in_doc = TextField([Token(word) for word in doc_str.split()], self.word_token_indexers)
                instances.append(Instance({"text": allen_token_word_in_doc}))
        self.vocab = Vocabulary.from_instances(instances, min_count={'tokens': 10})
        if save_to is not None:
            self.vocab.save_to_files(save_to)
        logging.info("Vocab size: {}".format(int(self.vocab.get_vocab_size())))

    def _read(self, file_path: str) -> Iterable[Instance]:
        assert self.vocab is not None
        # patten: file_path= train.pkl    .* pattern!!!!

        print("Into the read function. FP: {}".format(file_path))

        if file_path.startswith("train"):
            files = [x for x in os.listdir(self.dir) if x.startswith(file_path)]
            random.shuffle(files)
            if self.dbg:
                files = files[:1]
            else:
                files = files[:6]
        elif file_path.startswith("test") or file_path.startswith("dev"):
            files = [x for x in os.listdir(self.dir) if x.startswith(file_path)]
            # files = files[:1]

        for file in files:
            print("Reading {}".format(file))
            logging.info("Reading {}".format(file))
            f = open(os.path.join(self.dir, file), 'rb')

            data = pickle.load(f)
            for instance_fields in data:

                if self.trim_oracle:
                    sent_oracle = instance_fields['_sent_oracle']
                else:
                    sent_oracle = instance_fields['_non_compression_sent_oracle']
                instance_fields.pop('_sent_oracle')
                instance_fields.pop('_non_compression_sent_oracle')
                # instance_fields['_sent_oracle'] = None
                # instance_fields['_non_compression_sent_oracle'] = None

                sent_label_list, sent_rouge_list = filter_oracle_label(self.single_oracle,
                                                                       self.fix_edu_num, sent_oracle)

                def edit_label_rouge(_label_list, _rouge_list):
                    _label_list = [x for x in _label_list]
                    _max_len = max([len(x) for x in _label_list])
                    for idx, label in enumerate(_label_list):
                        if len(label) < _max_len:
                            _label_list[idx] = _label_list[idx] + [-1] * (_max_len - len(label))
                    np_gold_label = np.asarray(_label_list, dtype=np.int)
                    f = ArrayField(array=np_gold_label, padding_value=-1)

                    r = ArrayField(np.asarray(_rouge_list, dtype=np.float32))
                    return f, r

                if sent_label_list and sent_rouge_list:
                    label, rouge = edit_label_rouge(_label_list=sent_label_list,
                                                    _rouge_list=sent_rouge_list)
                    instance_fields["sent_label"] = label
                    instance_fields["sent_rouge"] = rouge
                else:
                    raise NotImplementedError

                yield self.text_to_instance(instance_fields
                                            )

    def text_to_instance(self, instance_fields
                         ) -> Instance:
        return Instance(instance_fields)
