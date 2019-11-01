# Mainly adapted from OpenNMT.

import torch
import torch.nn as nn
import torch.nn.functional as F
from neusum.service.basic_service import checkNaN


class NewAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_type="dot"):
        super(NewAttention, self).__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_type = attn_type

        # for prev_attn
        # self.gate_prev_attn = nn.Linear(dec_dim, 1
        #                                 )
        self._relu = nn.ReLU()
        if self.attn_type == 'general':
            # self.W_squeeze = nn.Linear(dim * 2, dim, bias=True)
            self.W_h = nn.Linear(enc_dim, dec_dim, bias=True)
            self.W_s = nn.Linear(dec_dim, dec_dim, bias=True)
            self.v = nn.Linear(dec_dim, 1)
            raise NotImplementedError
        elif self.attn_type == 'dot':
            self.W_h = nn.Linear(enc_dim, dec_dim, bias=True)
        else:
            raise NotImplementedError

    def forward_one_step(self, enc_state, dec_state, enc_mask,
                         prev_attn=None, penalty_val=10):
        """

        :param enc_state: batch_size_, src_len, enc_dim
        :param dec_state: batch_size, dec_dim
        :param enc_mask: batch_size_, src_len
        :param prev_attn:
        :param penalty_val:
        :return:
        """
        batch_size_, src_len, enc_dim = enc_state.size()
        batch_size, dec_dim = dec_state.size()
        assert batch_size == batch_size_
        assert enc_dim == self.enc_dim
        assert dec_dim == self.dec_dim
        if prev_attn is not None:
            # prev_attn
            gating_prev_attn = self._relu(self.gate_prev_attn(dec_state))  # batch, 1
            gated_prev_attn = gating_prev_attn * prev_attn  # batch, src_len <= a distribution

        if self.attn_type == 'dot':
            _middle = self.W_h(enc_state)  # batch_size, src_len, dec_dim
            unsqueezed_dec_state = dec_state.unsqueeze(2)
            score = torch.matmul(_middle, unsqueezed_dec_state)  # batch, src_len, 1
            score = score.squeeze(2)  # batch, src
        elif self.attn_type == 'general':
            w_enc = self.W_h(enc_state)  # batch, src, decdim
            w_dec = self.W_s(dec_state).unsqueeze(1)  # batch, [1], decdim
            _middle = torch.tanh(w_enc + w_dec)  # batch, src, decdim
            score = self.v(_middle)
            score = score.squeeze(2)  # batch, src
        else:
            raise NotImplementedError

        # checkNaN(score)
        if prev_attn is not None:
            # print(enc_mask.size())
            penaltied_score = score + (enc_mask - 1) * penalty_val - gated_prev_attn
        else:
            penaltied_score = score + (enc_mask - 1) * penalty_val
        attention_distribution = self.masked_softmax(penaltied_score, enc_mask)
        # x = self.named_parameters()
        # for k, v in x:
        #     print("k: {}\nv: {}".format(k,v))
        # checkNaN(attention_distribution)
        return attention_distribution, penaltied_score

    @staticmethod
    def masked_softmax(score, mask):
        """

        :param score: [batch_size, src_len]
        :param mask:
        :return:
        """
        """Take softmax of e then apply enc_padding_mask and re-normalize"""
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        # mask = mask + 0.00001
        attn_dist = F.softmax(score, dim=1)  # take softmax. shape (batch_size, attn_length)
        return attn_dist

        # attn_dist = attn_dist * mask  # apply mask
        # masked_sums = torch.sum(attn_dist, dim=1, keepdim=True)  # shape (batch_size)
        # masked_sums = masked_sums.expand_as(attn_dist)
        # return attn_dist / masked_sums
