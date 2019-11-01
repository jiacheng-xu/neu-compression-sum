import shutil
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from typing import Dict

from overrides import overrides

import torch,os
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
@Model.register('Seq2IdxSumInit')
class Seq2IdxSum(Model):
    def __init__(self,

                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,

                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:
        super(Seq2IdxSum, self).__init__(vocab, regularizer)

if __name__ == '__main__':
    root = "/scratch/cluster/jcxu/exComp/"
    dataset_dir = "/scratch/cluster/jcxu/data/SegSum/abc/"
    jsonnet_file = os.path.join(root, 'neusum/remade/model.jsonnet')
    params = Params.from_file(jsonnet_file)

    serialization_dir = tempfile.mkdtemp(prefix=os.path.join(root, 'tmp_exps'))
    model = train_model(params, serialization_dir)