import torch
import torch.nn as nn
from neusum.nn_modules.enc.transformer import MultiHeadedAttention

import copy
import math


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


import numpy as np


def convert_span_indices_to_mask(span_indices, span_indices_mask: torch.LongTensor = None, max_len=1):
    """

    :param span_indices: torch.LongTensor {batch, span_num, 2} [3,5] [-1,-1]
    :param span_indices_mask:  torch.LongTensor {batch, span_num} [1,0]
    :return: context_mask {batch, span_num, t} [ 0000111110000, 1111100000,....]
    """
    batch_size, span_num = span_indices.size()[0:2]
    # if span_indices_mask is None:
    #     virtual_mask = torch.ge(span_indices[:, :, 0], 0).long()
    mask = np.zeros((batch_size, span_num, max_len), dtype=np.int)
    # [  [for sp_idx in range(span_num)] for batch_sz in range(batch_size)]
    for batchsz in range(batch_size):
        for sp_idx in range(span_num):
            start, end = span_indices[batchsz, sp_idx]
            mask[batchsz, sp_idx, start:end] = 1
    mask = torch.from_numpy(mask)
    return mask


class SelfAttnExtractor(nn.Module):
    def __init__(self, h=8, d_model=512, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                # sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None):
        """
        convert span_indices to masks. eg. [1,4] (inclusive start, exclusive end) = mask [0 1 1 1 0 0 0 0]

        multi-head self attention to aggragate rep {batch, span_num, t, dim} => {batch, span_num, dim}
        for those invalid span_indices, span is set to be [0, t]

        :param sequence_tensor: torch.FloatTensor {batch, t, d_model}
        :param span_indices: torch.LongTensor {batch, span_num, 2}
        # :param sequence_mask: torch.LongTensor {batch, t} [1,0]
        :param span_indices_mask: torch.LongTensor {batch, span_num} [1,0]
        :return:
        """

        nbatch, t, dim = sequence_tensor.size()
        nb, span_num, _ = span_indices.size()
        spans = convert_span_indices_to_mask(span_indices, max_len=t)  # batch, span_num, t [111110000, 00011000,]
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (sequence_tensor, sequence_tensor, sequence_tensor))]
        # nbatch, t, d_model = > nbatch, t, h, d_k => nbatch, h, t, d_k
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)  # batch, h, t, t
        scores = scores.unsqueeze(1)
        scores = scores.expand((nbatch, span_num, self.h, t, t)).contiguous()
        scores = scores.view((nbatch * span_num, self.h, t, t))
        print(scores.size())

        spans = spans.view((nbatch, span_num, 1, t, 1))
        spans = spans.expand((nbatch, span_num, self.h, t, 1))
        spans = spans.expand((nbatch, span_num, self.h, t, t))
        spans = spans.view((nbatch * span_num, self.h, t, t))
        # print(scores[0])
        # print(spans_for_masking[0])
        # print(scores[2])
        # print(spans_for_masking[2])
        print(spans.size())
        print(spans[0, 0])
        print(scores[0, 0])
        scores = scores.masked_fill(spans == 0, -1e9)
        print(scores[0, 0])
        print(scores.size())


if __name__ == '__main__':
    inp_dim = 11
    hid_dim = 128
    batch = 3
    max_sent = 2
    max_t = 19

    span = torch.ones((batch, max_sent, 2), dtype=torch.long)
    span[:, 0, 0] = 5
    span[:, 0, 1] = 17
    span[:, 1, 0] = 2
    span[:, 1, 1] = 11
    span[batch - 1, :] = -1
    print(span)
    print(span.size())
    convert_span_indices_to_mask(span, None, max_t)

    context = torch.ones((batch, max_t, hid_dim))
    context.uniform_()

    attn = SelfAttnExtractor(h=4, d_model=hid_dim)
    attn.forward(sequence_tensor=context, span_indices=span)
