import json
from typing import Dict
import allennlp
import numpy as np

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


def filter_oracle_label(single_oracle, fix_edu_num, labels: Dict):
    """

    :param single_oracle:
    :param fix_edu_num:
    :param labels:
    :return: List[List], List[float]
    """
    if single_oracle:
        raise NotImplementedError
        # if fix_edu_num:
        #     cur_label = labels[str(fix_edu_num)]
        #     l = cur_label["best"]
        #     if l:
        #         return [l["label"]], [l["R"]]
        #     else:
        #         return [[1] * fix_edu_num], [0.1]  # TODO
        # else:
        #     _best_label = []
        #     _best_rouge = 0
        #     for k, v in labels.items():
        #         if v["best"]:
        #             if v["best"]["R"] > _best_rouge:
        #                 _best_label = v["best"]["label"]
        #                 _best_rouge = v["best"]["R"]
        #     if _best_rouge == 0:
        #         return [[1] * 2], [0.1]
        #     else:
        #         return [_best_label], [_best_rouge]
    else:
        if fix_edu_num:
            rt_data, rt_rouge = [], []
            cur_data = labels[str(fix_edu_num)]['data']
            for k, v in cur_data.items():
                rt_data.append(v['label'])
                rt_rouge.append(v['R1'])
        else:
            fix_edu_num = 2  # just in case
            rt_data, rt_rouge = [], []
            for n, lab in labels.items():
                data = lab['data']
                for k, v in data.items():
                    rt_data.append(v['label'])
                    rt_rouge.append(v['R1'])
        if not rt_rouge:
            return [[1] * fix_edu_num], [0.1]
        else:
            sort_idx = [i[0] for i in sorted(enumerate(rt_rouge), key=lambda x: x[1], reverse=True)]
            sort_rt_data = [rt_data[sort_idx[idx]] for idx in range(len(rt_data))]
            sort_rt_rouge = [rt_rouge[sort_idx[idx]] for idx in range(len(rt_rouge))]
            return sort_rt_data, sort_rt_rouge


def read_sent_object(sentence: dict):
    token = sentence['token']


class SummarizationDatasetReader(DatasetReader):
    def __init__(self,
                 source_token_indexers: Dict[str, TokenIndexer] = None,  # word token indexer
                 lazy: bool = False,
                 single_oracle=True,
                 fix_edu_num=None,
                 trim_oracle=False) -> None:
        super().__init__(lazy)
        self.word_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.single_oracle = single_oracle
        self.fix_edu_num = fix_edu_num
        self.trim_oracle = trim_oracle

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as fd:
            for line in fd:
                data_dict = json.loads(line)
                name = data_dict['name']
                doc_str = data_dict['token']
                abs_str = data_dict['abs']
                span_pairs = data_dict['span']
                doc_list = data_dict['token_list']
                abs_list = data_dict['abs_token']
                sentences = data_dict['sentences']
                sent_oracle = data_dict['sent_oracle']
                sent_oracle_trim = data_dict['sent_oracle_trim']

                allen_token_word_in_doc = [Token(word) for word in doc_str.split()]

                # word_in_abs = [Token(word) for word in abst.split()]# TODO
                # assert int(span_info.split()[-1]) + 1 == len(word_in_doc)
                # span_info = [int(w) for w in span_info.split()]
                # assert len(span_info) % 2 == 0
                # idx_in_span = list(zip(span_info[0::2], span_info[1::2]))

                if self.trim_oracle:
                    sent_label_list, sent_rouge_list = filter_oracle_label(self.single_oracle,
                                                                           self.fix_edu_num, sent_oracle_trim)

                else:
                    sent_label_list, sent_rouge_list = filter_oracle_label(self.single_oracle,
                                                                           self.fix_edu_num, sent_oracle)

                yield self.text_to_instance(name, allen_token_word_in_doc,
                                            doc_list,
                                            abs_list,
                                            doc_str,
                                            abs_str,
                                            sentences,
                                            span_pairs, sent_label_list, sent_rouge_list

                                            )

    def text_to_instance(self, name: str,
                         allen_token_word_in_doc: List[Token],
                         doc_list: List[List[str]],
                         abs_list: List[List[str]],
                         doc_str: str,
                         abs_str: str,
                         sentences: List[dict],
                         span_pairs: List[List[int]],
                         sent_label_list: List[List[int]],
                         sent_rouge_list: List[float]
                         # word_in_doc_in_list: List,
                         # word_in_abs_in_list: List,
                         # document: List[Token],
                         # abstract: List[Token],
                         # span: List[Tuple[int, int]],
                         # label_list: List = None,
                         # rouge_list: List = None,
                         # name: str = None,
                         # _max_len: int = 5
                         ) -> Instance:
        instance_fields = {}
        doc_text_field = TextField(allen_token_word_in_doc, self.word_token_indexers)
        # print(doc_text_field.sequence_length())
        instance_fields["text"] = doc_text_field

        def edit_label_rouge(_label_list, _rouge_list):
            _label_list = [x + [0] for x in _label_list]
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
            instance_fields["label"] = label
            instance_fields["rouge"] = rouge
        else:
            raise NotImplementedError

        spans = []
        # print(span)
        for s in span_pairs:
            _sf = SpanField(span_start=s[0], span_end=s[1],
                            sequence_field=doc_text_field)
            spans.append(_sf)
        span_field = ListField(spans)
        instance_fields["spans"] = span_field

        instance_fields["metadata"] = MetadataField(
            {
                "doc_list": doc_list,
                "abs_list": abs_list,
                "name": name,
                "sentences": sentences,
                # "label": label,
                # "rouge": rouge
            }
        )
        return Instance(instance_fields)
