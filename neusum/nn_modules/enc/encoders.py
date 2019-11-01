import torch.nn
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
import torch.nn.functional as F
from allennlp.modules.span_extractors import EndpointSpanExtractor, BidirectionalEndpointSpanExtractor, \
    SelfAttentiveSpanExtractor
from neusum.nn_modules.attention import NewAttention
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data import Vocabulary
from neusum.nn_modules.NormalSpanExtractor import EasySpanExtractor

import logging
from allennlp.common.params import Params

class EncWord2Sent(torch.nn.Module):
    # Normal BLSTM encoder from word embedding to hidden rep
    def __init__(self, device, inp_dim, hidden_dim, nenc_lay, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enc_blstm = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(inp_dim, hidden_dim,
                          batch_first=True, bidirectional=True,
                          num_layers=nenc_lay))
        self._span_encoder = BidirectionalEndpointSpanExtractor(
            self.enc_blstm.get_output_dim()
        )
        self._dropout = torch.nn.Dropout(p=dropout)
        self.device = device

    def get_output_dim(self):
        return self.enc_blstm.get_output_dim()

    def forward(self, context, context_msk):
        """

        :param context: [batch, t, inp_dim]
        :param context_msk: [batch, t] [0,1]
        :return: blstm_output [batch, t, hid_dim*2]
                avg_blstm_out [batch, hid*2]
        """
        batch_size = context.size()[0]
        blstm_output = self._dropout(self.enc_blstm(context, context_msk))      # blstm
        context_msk[:,0] = 1
        # context_msk = [batch, t]
        context_mask_sum = torch.sum(context_msk, dim=1) - 1
        # context_mask_sum = [batch]   [ 40, 4, 9, 0(sent len=1) , -1(sent len=0), -1]
        sum_of_mask = context_mask_sum.unsqueeze(dim=1).unsqueeze(dim=2).long()

        span_idx = torch.ones((batch_size), device=self.device, dtype=torch.long) * -1
        # [ [-1] [-1] [-1] [-1] ]
        # according to sum_of_mask
        valid_bit = (context_mask_sum >= 0).long()
        span_idx = span_idx + valid_bit
        span_idx = span_idx.view((batch_size, 1, 1))
        span_idx = torch.cat([span_idx, sum_of_mask], dim=2).long()
        # Span module: (batch_size, sequence_length, embedding_size)
        #                (batch_size, num_spans, 2)
        attended_text_embeddings = self._span_encoder.forward(sequence_tensor=blstm_output,
                                                              span_indices=span_idx,
                                                              sequence_mask=context_msk,
                                                              span_indices_mask=valid_bit.unsqueeze(1))
        # attended_text_embeddings: batch, 1, dim


        attended_text_embeddings = attended_text_embeddings.squeeze(1)
        # valid_len = context_msk.sum(dim=1).unsqueeze(1)  # batchsz,1.
        # context_msk = context_msk.unsqueeze(2)
        # msked_blstm_out = context_msk * blstm_output
        attended_text_embeddings = self._dropout(attended_text_embeddings)
        return blstm_output, attended_text_embeddings


class EncWord2PhrasesViaMapping(torch.nn.Module):
    # Step1: BLSTM(a b c d e f g)
    # Step2: Mapping: [a:g] [a:c] [f:g]
    # Step3: h[a:g] = BLSTM(a b c d e f g) * [1, 1, 1, 1, 1, 1, 1]
    #         h[f:g] = BLSTM(a b c d e f g) * [0, 0, 0, 0, 0, 1, 1]
    # Input: word_embedding: batch, t, dim
    #        word_embedding_msk: batch, t
    #       span: batch, n_span, 2
    #       span_msk: batch, n_span : n_span varies so we need to know the num of span for every sample in the batch
    #       span_ready_to_use: batch, n_span, t: like [1,1,1,1,0,0,0]
    # Output:
    #       Phrases: batch, n_span, dim
    #       phrases_msk: batch, n_span
    def __init__(self, inp_dim, hidden_dim, nenc_lay, dropout):
        super().__init__()

        self.enc_blstm = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(inp_dim, hidden_dim,
                          batch_first=True, bidirectional=True,
                          num_layers=nenc_lay))
        self._dropout = torch.nn.Dropout(p=dropout)
        self._span_encoder = BidirectionalEndpointSpanExtractor(
            self.enc_blstm.get_output_dim()
        )

    def forward(self, context, context_msk, span):
        """

        :param context:
        :param context_msk:
        :param span:
        :return: spans_rep: [batch, n_span, dim], float
                span_msk: [batch, n_span], binary [1,1,0,0]...
        """
        blstm_output = self._dropout(self.enc_blstm(context, context_msk))
        span_msk = (span[:, :, 0] >= 0).long()

        # tensor([[1., 1., 1., 0.],
        # [1., 1., 1., 1.],
        # [1., 1., 1., 0.]])
        spans_for_comp = F.relu(span.float()).long()

        spans_rep = self._span_encoder(
            sequence_tensor=blstm_output,
            span_indices=spans_for_comp,
            sequence_mask=context_msk,
            span_indices_mask=span_msk)

        valid_len = context_msk.sum(dim=1).unsqueeze(1)  # batchsz,1.
        context_msk = context_msk.unsqueeze(2)
        msked_blstm_out = context_msk * blstm_output
        avg_blstm_out = torch.sum(msked_blstm_out, dim=1) / valid_len

        # for name, param in self._span_encoder.named_parameters():
        #     if 'bias' in name:
        #         print("name: {}".format(name))
        #         print("Param: {}".format(param))

        return blstm_output, avg_blstm_out, spans_rep, span_msk


class EncPhrases2CondensedPhraseViaAttn(torch.nn.Module):
    # InputL
    # phrases, [batch, n_span, dim], float
    # phrases_msk, [batch, n_span], binary [1,1,0,0]...
    # attn_sent: [batch, dim]
    def __init__(self, context_dim, sent_dim):
        super().__init__()
        self.attn = NewAttention(enc_dim=context_dim, dec_dim=sent_dim)

    def forward(self, phrases, phrases_msk, attn_sent):
        """

        :param phrases:
        :param phrases_msk:
        :param attn_sent:
        :return: reweighted_phrases: use attn to reweight phrasese: [batch, dim]
                attn_dist: attn dist of phrases given doc: [batch, nspan=src_len]
        """
        attn_dist, score = self.attn.forward_one_step(enc_state=phrases,
                                                      dec_state=attn_sent,
                                                      enc_mask=phrases_msk.float())
        unsqueezed_attn_dist = attn_dist.unsqueeze(-1)
        reweighted_phrases = unsqueezed_attn_dist * phrases
        reweighted_phrases = torch.sum(reweighted_phrases, dim=1)
        # print(reweighted_phrases.size())
        return reweighted_phrases, attn_dist, score


class EncSent(torch.nn.Module):
    # given a sentence and its compressions, output representation and attn dist.
    def __init__(self, device, inp_dim, hid_dim, compression=True, dropout=0.4):
        super().__init__()
        self.hid_dim = hid_dim
        self.compression = compression
        if compression:
            self.plain_enc = EncWord2PhrasesViaMapping(inp_dim=inp_dim, hidden_dim=hid_dim,
                                                       nenc_lay=2, dropout=dropout)
            self.condensed_phrases = EncPhrases2CondensedPhraseViaAttn(context_dim=hid_dim * 2, sent_dim=hid_dim * 2)
        else:
            self.enc = EncWord2Sent(device=device, inp_dim=inp_dim, hidden_dim=hid_dim,
                                    nenc_lay=1, dropout=dropout)

    def get_output_dim(self):
        if self.compression:
            return self.hid_dim * 4
        else:
            return self.hid_dim * 2

    def forward(self, word_emb, word_emb_msk, span):
        """

        :param word_emb: batch, t, dim
        :param word_emb_msk: [batch, t]
        :param span: batch, n_span, 2
        :return:
        """
        # word_emb: batch, t, dim
        # word_emb_msk: [batch, t] [0,1]
        # span: batch, n_span, 2
        if self.compression:
            blstm_output, avg_blstm_out, spans_rep, span_msk = self.plain_enc.forward(context=word_emb,
                                                                                      context_msk=word_emb_msk,
                                                                                      span=span)

            reweighted_phrases, attn_dist, score = self.condensed_phrases.forward(phrases=spans_rep,
                                                                                  phrases_msk=span_msk,
                                                                                  attn_sent=avg_blstm_out)
            # print(reweighted_phrases.size())
            # print(avg_blstm_out.size())
            cat_rep = torch.cat([reweighted_phrases, avg_blstm_out], dim=1)
            # print(cat_rep.size())
            return cat_rep, attn_dist, spans_rep, span_msk, score
        else:
            blstm_output, avg_blstm_out = self.enc.forward(context=word_emb, context_msk=word_emb_msk)
            return avg_blstm_out, None, None, None, None


class EncDoc(torch.nn.Module):
    # Encode a document.
    def __init__(self, device, inp_dim, hid_dim, compression, vocab, dropout: float = 0.4,
                 dropout_emb: float = 0.2, pretrain_embedding_file=None):
        super().__init__()
        self.compression = compression
        self.hid_dim = hid_dim
        self.sent_enc = EncSent(device=device, inp_dim=inp_dim, hid_dim=hid_dim, compression=compression)
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=inp_dim)

        if dropout_emb > 0:
            self._lexical_dropout = torch.nn.Dropout(p=dropout_emb)
        else:
            self._lexical_dropout = lambda x: x

        if pretrain_embedding_file is not None:
            logger = logging.getLogger()
            logger.info("Loading word embedding: {}".format(pretrain_embedding_file))
            token_embedding.from_params(vocab=vocab,
                                        params=Params({"pretrained_file": pretrain_embedding_file,
                                                       "embedding_dim": inp_dim})
                                        )
        self._text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        self.sent2doc = EncWord2Sent(device=device, inp_dim=self.sent_enc.get_output_dim(), hidden_dim=hid_dim,
                                     nenc_lay=2, dropout=dropout)

    def forward(self, context, context_msk, spans):
        #
        # inp: batch, max_sent, max_t LongTensor
        # [sent_0, sent_1, ... sent_n]
        # for every sent, word_emb, word_embed_msk, span
        # sent and compression encoding is based on every sent
        # doc_rep is based on the whole batch
        # context: batch, max_sent, max_word, inp_dim
        # context_msk: batch, max_sent, max_word {1,0}
        # spans: batch, max_sent, nspan, 2


        context = self._text_field_embedder(context)
        context = self._lexical_dropout(context)
        num_doc, max_sent, max_word, inp_dim = context.size()
        num_doc_, max_sent_, nspan = spans.size()[0:-1]
        assert num_doc == num_doc_
        assert max_sent == max_sent_

        mix_batch_and_sent = num_doc * max_sent
        flattened_context = context.view(mix_batch_and_sent, max_word, inp_dim)
        flattened_context_msk = context_msk.view(mix_batch_and_sent, max_word)
        flattened_spans = spans.view(mix_batch_and_sent, nspan, 2)
        flattened_enc, attn_dist, spans_rep, span_msk, score = self.sent_enc.forward(word_emb=flattened_context,
                                                                                     word_emb_msk=flattened_context_msk,
                                                                                     span=flattened_spans)
        if spans_rep is not None:
            reorg_spans_rep = spans_rep.view(num_doc, max_sent, nspan, self.hid_dim * 2)
            reorg_span_msk = span_msk.view(num_doc, max_sent, nspan)
        else:
            reorg_spans_rep, reorg_span_msk = None, None
        if attn_dist is not None:
            # print(attn_dist.size())
            attn_dist_of_phrases = attn_dist.view(num_doc, max_sent, nspan)
        else:
            attn_dist_of_phrases = None
        enc = flattened_enc.view(num_doc, max_sent, -1)
        # print(enc.size())
        sent_mask = context_msk[:, :, 0]
        # print(sent_mask.size())
        sent_blstm_output, document_rep = self.sent2doc.forward(context=enc, context_msk=sent_mask)
        # sent_blstm_output: batch, max_sent, hdim*2
        # document_rep:     batch, hdim*2
        # attn_dist_of_phrases: batch, max_sent, max_num_of_span
        # reorg_spans_rep:         batch, max_sent, max_num_of_span, hid*2
        # reorg_span_msk:       batch, max_sent, max_num_of_span
        return sent_blstm_output, document_rep, attn_dist_of_phrases, reorg_spans_rep, reorg_span_msk
