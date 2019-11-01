import logging
import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from neusum.nn_modules.enc.gather import masked_sum, select_gather, GatherCNN
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
import torch.nn.functional as F
from allennlp.common.params import Params
from neusum.service.shared_asset import get_device

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator, util, RegularizerApplicator
from typing import Any, Dict, List, Optional, Tuple


@Model.register('EncWord2Sent')
class EncWord2Sent(torch.nn.Module):
    # Normal BLSTM encoder from word embedding to hidden rep
    def __init__(self, inp_dim, hid_dim, dropout, nenc_lay=1, gather='sum'):
        super().__init__()
        self.hidden_dim = hid_dim
        self.enc_blstm = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(inp_dim, hid_dim,
                          batch_first=True, bidirectional=True,
                          num_layers=nenc_lay))

        # self._span_encoder = select_gather(gather)
        self._span_encoder = GatherCNN(input_dim=self.enc_blstm.get_output_dim(),
                                       num_filters=5, output_dim=self.enc_blstm.get_output_dim())
        self._dropout = torch.nn.Dropout(p=dropout)
        self.device = get_device()

    def get_output_dim(self):
        assert self.enc_blstm.get_output_dim() == self.hidden_dim * 2
        return self.enc_blstm.get_output_dim()

    def forward(self, context, context_msk):
        """

        :param context: [batch, t, inp_dim]
        :param context_msk: [batch, t] [0,1]
        :return: blstm_output [batch, t, hid_dim*2]
                avg_blstm_out [batch, hid*2]
        """
        batch_size = context.size()[0]
        b_size = context_msk.size()[0]
        assert context.size()[1] == context_msk.size()[1]
        assert batch_size == b_size

        blstm_output = self._dropout(self.enc_blstm(context, context_msk))  # blstm
        # blstm_output: batch, t, dim
        gathered_output = self._span_encoder.forward(tokens=blstm_output, msk=context_msk)
        # gathered_output: batch, dim
        return blstm_output, gathered_output
        # context_msk[:, 0] = 1
        # # context_msk = [batch, t]
        # context_mask_sum = torch.sum(context_msk, dim=1) - 1
        # # context_mask_sum = [batch]   [ 40, 4, 9, 0(sent len=1) , -1(sent len=0), -1]
        # sum_of_mask = context_mask_sum.unsqueeze(dim=1).unsqueeze(dim=2).long()
        #
        # span_idx = torch.ones((batch_size), device=self.device, dtype=torch.long) * -1
        # # [ [-1] [-1] [-1] [-1] ]
        # # according to sum_of_mask
        # valid_bit = (context_mask_sum >= 0).long()
        # span_idx = span_idx + valid_bit
        # span_idx = span_idx.view((batch_size, 1, 1))
        # span_idx = torch.cat([span_idx, sum_of_mask], dim=2).long()
        # # Span module: (batch_size, sequence_length, embedding_size)
        # #                (batch_size, num_spans, 2)
        # attended_text_embeddings = self._span_encoder.forward(sequence_tensor=blstm_output,
        #                                                       span_indices=span_idx,
        #                                                       sequence_mask=context_msk,
        #                                                       span_indices_mask=valid_bit.unsqueeze(1))
        # # attended_text_embeddings: batch, 1, dim
        #
        # attended_text_embeddings = attended_text_embeddings.squeeze(1)
        # # valid_len = context_msk.sum(dim=1).unsqueeze(1)  # batchsz,1.
        # # context_msk = context_msk.unsqueeze(2)
        # # msked_blstm_out = context_msk * blstm_output
        # attended_text_embeddings = self._dropout(attended_text_embeddings)
        # return blstm_output, attended_text_embeddings
