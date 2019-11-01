import shutil

from allennlp.common.file_utils import cached_path
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from typing import Dict

from allennlp.predictors import SentenceTaggerPredictor
from overrides import overrides
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
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


def unfold_sentence(node):
    pair_list = []
    for n in node:
        pair_list.append((n['parent_id'], n['child_id']))
    return pair_list


import json


def load_data():
    # transfer the data to tsv format
    dir = '/backup3/jcxu/data/compression-data.json'
    train_file = '/backup3/jcxu/data/compression-train.tsv'
    test_file = '/backup3/jcxu/data/compression-test.tsv'
    with open(dir, 'r') as fd:
        lines = fd.read().splitlines()
    line_num = [idx for idx, x in enumerate(lines) if x == ""]
    line_num = [0] + line_num + [-1]
    dataset = []
    for idx in range(len(line_num) - 1):
        start_line = line_num[idx]
        if line_num[idx + 1] == -1:
            end_line = -1
            tmp_lines = lines[start_line:]
        else:
            end_line = line_num[idx + 1]
            tmp_lines = lines[start_line:end_line]
        str_lines = r' '.join(tmp_lines)
        data = json.loads(str_lines)

        compress_edges = unfold_sentence(data['compression']['edge'])
        original_edges = unfold_sentence(data['graph']['edge'])
        node_lists = [(n['word'][n['head_word_index']]['id'],
                       n['word']) for idx, n in
                      enumerate(data['graph']['node'])]

        delta_edges = list(set(original_edges) - set(compress_edges))
        compressed_nodes = [c[1] for c in delta_edges]  # all of the compressed child_ids
        # print(compressed)

        from operator import itemgetter
        node_lists.sort(key=itemgetter(0))
        max_idx = node_lists[-1][1][-1]['id'] + 20
        # print(node_lists)

        tags = ["" for _ in range(max_idx)]
        sents = ["" for _ in range(max_idx)]
        for node in node_lists:
            idx = node[0]
            if idx == -1:
                continue
            words = node[1]
            for w_dict in words:
                sents[w_dict['id']] = w_dict['form']
            # words = [w['form'] for w in words]
            l = len(words)
            if idx in compressed_nodes:
                for w_dict in words:
                    tags[w_dict['id']] = 'B'
                tags[words[0]['id']] = 'B'
            else:
                for w_dict in words:
                    tags[w_dict['id']] = 'O'

        this_example = []
        for t, s in zip(tags, sents):
            if t == "":
                continue
            this_example.append("{}#@#@#{}".format(s, t))

        #     print(t,s)
        ex = "\t".join(this_example)
        dataset.append(ex)

    # random.shuffle(dataset)
    test = dataset[:1000]
    train = dataset[1000:9000]
    # test = dataset[8000:]
    print(test[100])
    print(test[101])
    with open(train_file, 'w') as fd:
        fd.write("\n".join(train))

    with open(test_file, 'w') as fd:
        fd.write("\n".join(test))


from typing import Iterator, List, Dict


@DatasetReader.register('pos-tutorial')
class PosDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(cached_path(file_path)) as f:
            for line in f:
                pairs = line.strip().split("\t")
                sentence, tags = zip(*(pair.split("#@#@#") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)


from allennlp.training.metrics import F1Measure

from torch import nn


@Model.register('lstm-tagger')
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 sec_encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.sec_encoder = sec_encoder
        self.hidden2tag = torch.nn.Linear(in_features=sec_encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        print(vocab.get_token_from_index(2))
        print(vocab.get_token_from_index(14))
        print(vocab.get_vocab_size('labels'))
        print(vocab.get_token_from_index(0, 'labels'))
        print(vocab.get_token_from_index(1, 'labels'))

        self.accuracy = CategoricalAccuracy()
        self.f1_ret = F1Measure(1)
        self.f1_del = F1Measure(0)
        self.drop = nn.Dropout(p=0.5)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        encoder_out = self.drop(encoder_out)
        encoder_out = self.sec_encoder(encoder_out, mask)
        encoder_out = self.drop(encoder_out)
        tag_logits = self.hidden2tag(encoder_out)

        token_list = tokens['tokens'].tolist()
        lexico_token = []
        for sent in token_list:
            lexico_token.append([self.vocab.get_token_from_index(x) for x in sent if x != 0])
        output = {"tag_logits": tag_logits,
                  "tokens": lexico_token}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            self.f1_ret(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, fscore = self.f1_ret.get_metric(reset)
        return {"accuracy": self.accuracy.get_metric(reset),
                "f1": fscore,
                "prec": precision,
                "recall": recall}


from allennlp.commands.predict import _predict
from allennlp.commands.predict import _PredictManager

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance


def pred(cuda_device=0,
         archive_file="/backup3/jcxu/exComp/tmp_expsc74o5pf7/model.tar.gz",
         weights_file="/backup3/jcxu/exComp/tmp_expsc74o5pf7/best.th",
         predictor='lstm-tagger',
         input_file="/backup3/jcxu/exComp/example.txt"
         ):
    with open(input_file, 'w') as fd:
        json.dump(
            {"sentence": "This is a useful sentence."}, fd)
        fd.write("\n")
        json.dump(
            {"sentence": "This is a gree, blue and useful sentence."}, fd)
        fd.write("\n")
        json.dump(
            {"sentence": "This is a useless sentence."}, fd)
    check_for_gpu(cuda_device)
    archive = load_archive(archive_file,
                           weights_file=weights_file,
                           cuda_device=cuda_device,
                           overrides="")
    # predictor = SentenceTaggerPredictor(archive, dataset_reader=PosDatasetReader())
    predictor = Predictor.from_archive(archive, 'sentence-tagger')

    manager = _PredictManager(predictor,
                              input_file,
                              None,
                              1,
                              not False,
                              False)
    manager.run()


import pickle

if __name__ == '__main__':
    # simple_tagger
    # load_data()
    # exit()
    # root = "/scratch/cluster/jcxu/exComp/"
    import allennlp

    f = "/backup3/jcxu/exComp/0.319-0.115-0.284-cnnTrue1.0-1True1093"
    with open(f, 'rb') as fd:
        x = pickle.load(fd)

    pred()
    exit()
    print(allennlp.__version__)
    root = "/backup3/jcxu/exComp/"
    # dataset_dir = "/scratch/cluster/jcxu/data/SegSum/abc/"
    jsonnet_file = os.path.join(root, 'neusum/BaselineTagger/experiment.jsonnet')

    params = Params.from_file(jsonnet_file)

    serialization_dir = tempfile.mkdtemp(prefix=os.path.join(root, 'tmp_exps'))
    model = train_model(params, serialization_dir)


    print(serialization_dir)
    predictor = SentenceTaggerPredictor(model, dataset_reader=PosDatasetReader())
    output = predictor.predict(
        "Five people have been taken to hospital with minor injuries following a crash on the A17 near Sleaford this morning.")
    logits = output['tag_logits']
    # print(logits)
    tag_ids = np.argmax(logits, axis=-1)
    print(tag_ids)
    print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
