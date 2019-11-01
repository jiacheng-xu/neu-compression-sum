import logging
import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from neusum.nn_modules.enc.gather import masked_sum, select_gather
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
import torch.nn.functional as F
from allennlp.common.params import Params
from neusum.nn_modules.enc.enc_word2sent import EncWord2Sent
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator, util, RegularizerApplicator
from typing import Any, Dict, List, Optional, Tuple

@Model.register('EncDoc')
class EncDoc(torch.nn.Module):
    # Encode a document.
    def __init__(self,

                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,

                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,

                 inp_dim,
                 hid_dim,
                 dropout: float = 0.4,
                 dropout_emb: float = 0.2,
                 pretrain_embedding_file=None,
                 gather='sum'):
        super(EncDoc, self).__init__(vocab, regularizer)
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=inp_dim)

        if dropout_emb > 0:
            self._lexical_dropout = torch.nn.Dropout(p=dropout_emb)
        else:
            self._lexical_dropout = lambda x: x

        self.hid_dim = hid_dim
        self.sent_enc = EncWord2Sent(inp_dim=inp_dim,
                                     hid_dim=hid_dim, dropout=dropout, gather=gather)

        if pretrain_embedding_file is not None:
            logger = logging.getLogger()
            logger.info("Loading word embedding: {}".format(pretrain_embedding_file))
            token_embedding.from_params(vocab=vocab,
                                        params=Params({"pretrained_file": pretrain_embedding_file,
                                                       "embedding_dim": inp_dim})
                                        )
            print("token_embedding size: {}".format(token_embedding.num_embeddings))
        self._text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        self.sent2doc = EncWord2Sent(inp_dim=self.sent_enc.get_output_dim(), hid_dim=hid_dim,
                                     nenc_lay=1, dropout=dropout)

    def get_output_dim(self):
        return self.sent2doc.get_output_dim()

    def forward(self, inp, context_msk):
        """

        :param inp: [batch, max_sent, max_word_num]
        :param context_msk:
        :return:
        """
        #
        # inp: batch, max_sent, max_t LongTensor
        # [sent_0, sent_1, ... sent_n]
        # for every sent, word_emb, word_embed_msk, span
        # sent and compression encoding is based on every sent
        # doc_rep is based on the whole batch
        # context: batch, max_sent, max_word, inp_dim
        # context_msk: batch, max_sent, max_word {1,0}
        # spans: batch, max_sent, nspan, 2

        context = self._text_field_embedder(inp)
        context = self._lexical_dropout(context)
        num_doc, max_sent, max_word, inp_dim = context.size()
        _num_doc, _max_sent, _max_word = context_msk.size()
        # num_doc_, max_sent_, nspan = spans.size()[0:-1]
        assert num_doc == _num_doc
        assert max_sent == _max_sent

        mix_batch_and_sent = num_doc * max_sent
        flattened_context = context.view(mix_batch_and_sent, max_word, inp_dim)
        flattened_context_msk = context_msk.view(mix_batch_and_sent, max_word)
        # flattened_spans = spans.view(mix_batch_and_sent, nspan, 2)
        _, gathered_output = self.sent_enc.forward(context=flattened_context,
                                                   context_msk=flattened_context_msk
                                                   )
        sent_reps = gathered_output.view(num_doc, max_sent, self.sent_enc.get_output_dim())
        sent_mask = context_msk[:, :, 0]  # [batch, sent]
        # print(sent_mask.size())
        sent_blstm_output, document_rep = self.sent2doc.forward(context=sent_reps, context_msk=sent_mask)
        # sent_blstm_output: batch, max_sent, hdim*2
        # document_rep:     batch, hdim*2
        # attn_dist_of_phrases: batch, max_sent, max_num_of_span
        # reorg_spans_rep:         batch, max_sent, max_num_of_span, hid*2
        # reorg_span_msk:       batch, max_sent, max_num_of_span
        return sent_blstm_output, document_rep
