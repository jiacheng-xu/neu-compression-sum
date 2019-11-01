import allennlp
import pickle
import os
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

import json
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, SpanField, ListField, LabelField, \
    IndexField, ArrayField, SequenceField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import numpy as np
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from typing import List
import random
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
import pickle
from neusum.data.generate_compression_based_data import TreeNode

if __name__ == '__main__':
    f_path = "/backup3/jcxu/data/2merge-cnndm/train.pkl.dm.00"
    # x = pickle.loads(open(f_path))
    with open(f_path, 'rb') as fd:
        pickle.load(fd)
    # pickle.load(open(f_path,'rb'))
