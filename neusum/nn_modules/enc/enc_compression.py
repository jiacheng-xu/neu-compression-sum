import torch
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from neusum.nn_modules.enc.gather import select_gather, GatherCNN

from neusum.service.shared_asset import get_device
from neusum.nn_modules.attention import NewAttention

s = lambda x: print(x.size())


class EncCompression(torch.nn.Module):
    # given a sentence and its compressions, output representation and attn dist.

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
    def __init__(self,
                 inp_dim, hid_dim, dropout=0.2, gather='sum', nenc_lay=1):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = get_device()
        self.enc_blstm = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(inp_dim, hid_dim,
                          batch_first=True, bidirectional=True,
                          num_layers=nenc_lay))
        self._dropout = torch.nn.Dropout(p=dropout)
        self.gather_func = select_gather(gather)

        # self.attn = NewAttention(enc_dim=self.enc_blstm.get_output_dim() * 3,
        #                          dec_dim=self.enc_blstm.get_output_dim())
        self.cnn_sent_enc = GatherCNN(input_dim=self.enc_blstm.get_output_dim(),
                                      num_filters=3, output_dim=self.enc_blstm.get_output_dim())
        self.cnn_comp_enc = GatherCNN(input_dim=self.enc_blstm.get_output_dim(),
                                      num_filters=3, output_dim=self.enc_blstm.get_output_dim())

    def get_output_dim(self):
        return self.enc_blstm.get_output_dim() * 3

    def get_output_dim_unit(self):
        return self.enc_blstm.get_output_dim()

    def forward(self, word_emb, word_emb_msk, span):
        """

        :param word_emb: [batch, t, dim]
        :param word_emb_msk: [batch, t]
        :param span: [batch, n_span, t] like 000111110000
        :return: concat_rep: [batch, n_span, output dim]
                mask_of_span: [batch, n_span]
                original_sent_rep: [batch, out dim]
        """
        batch_z, t_, dim = word_emb.size()
        hiddim = self.enc_blstm.get_output_dim()
        blstm_output = self._dropout(self.enc_blstm(word_emb, word_emb_msk))
        batch, num_of_compression, t = span.size()
        word_emb = blstm_output.unsqueeze(1)
        expanded_word_emb = word_emb.expand(batch, num_of_compression, t, self.enc_blstm.get_output_dim()).contiguous()

        # these are compressions
        # easy version
        # enc_compressions = self.gather_func(inp=expanded_word_emb, dim=2, msk=span)
        # cnn version
        sqzed_expanded_word_emb = expanded_word_emb.view(batch * num_of_compression, t,
                                                         self.enc_blstm.get_output_dim()).contiguous()
        sqzed_span = span.view(batch * num_of_compression, t)
        comp_out = self.cnn_comp_enc.forward(tokens=sqzed_expanded_word_emb, msk=sqzed_span)
        enc_compressions = comp_out.view(batch, num_of_compression, self.cnn_comp_enc.get_output_dim())
        # enc_compressions: batch, num_of_compression, dim

        original_sent_rep = self.cnn_sent_enc.forward(tokens=blstm_output, msk=word_emb_msk)
        # original_sent_rep = self.gather_func(inp=blstm_output, dim=1)
        # original_sent_rep: batch, dim
        sqz_original_sent_rep = original_sent_rep.unsqueeze(1)
        sqz_original_sent_rep = sqz_original_sent_rep.expand(batch, num_of_compression, self.enc_blstm.get_output_dim())

        full_minus_compression = sqz_original_sent_rep - enc_compressions
        concat_rep = torch.cat([enc_compressions, sqz_original_sent_rep, full_minus_compression]
                               , dim=-1)

        sum_of_span = torch.sum(input=span, dim=2)
        mask_of_span = (sum_of_span > 0).float()
        # s(concat_rep)
        # s(original_sent_rep)
        # s(mask_of_span)
        # torch.Size([7, 17, 222])
        # torch.Size([7, 74])
        # torch.Size([7, 17])
        # attention_distribution, penaltied_score = self.attn.forward_one_step(enc_state=concat_rep,
        #                                                                      dec_state=original_sent_rep,
        #                                                                      enc_mask=mask_of_span)
        # s(attention_distribution)
        # s(penaltied_score)
        return concat_rep, mask_of_span, original_sent_rep
        # , attention_distribution, penaltied_score

        # span_msk = (span[:, :, 0] >= 0).long()
        #
        # # tensor([[1., 1., 1., 0.],
        # # [1., 1., 1., 1.],
        # # [1., 1., 1., 0.]])
        # spans_for_comp = F.relu(span.float()).long()
        #
        # spans_rep = self._span_encoder(
        #     sequence_tensor=blstm_output,
        #     span_indices=spans_for_comp,
        #     sequence_mask=context_msk,
        #     span_indices_mask=span_msk)
        #
        # blstm_output, avg_blstm_out, spans_rep, span_msk = self.plain_enc.forward(context=word_emb,
        #                                                                           context_msk=word_emb_msk,
        #                                                                           span=span)
        #
        # reweighted_phrases, attn_dist, score = self.condensed_phrases.forward(phrases=spans_rep,
        #                                                                       phrases_msk=span_msk,
        #                                                                       attn_sent=avg_blstm_out)
        # # print(reweighted_phrases.size())
        # # print(avg_blstm_out.size())
        # cat_rep = torch.cat([reweighted_phrases, avg_blstm_out], dim=1)
        # # print(cat_rep.size())
        # return cat_rep, attn_dist, spans_rep, span_msk, score
