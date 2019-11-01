import random
from typing import List, Tuple

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from neusum.nn_modules.attention import NewAttention
from neusum.service.nn_services import ptr_network_index_select


# RNN decoder with Attention without copy and coverage
# Not fixed classed

class InputFeedRNNDecoder(nn.Module):
    def __init__(self, device: torch.device,
                 rnn_type: str = 'lstm',
                 dec_hidden_size: int = 100,
                 dec_input_size: int = 50,
                 attn_type: str = 'dot',
                 dropout: float = 0.1,
                 flexible_dec_step: bool = False,
                 max_dec_steps: int = 3,
                 min_dec_steps: int = 3,
                 schedule_ratio_from_ground_truth: float = 0.5,
                 mult_orac_sampling: bool = True):

        super().__init__()
        self.device = device
        self._rnn_type = rnn_type
        self._dec_input_size = dec_input_size
        self._dec_hidden_size = dec_hidden_size

        self._max_dec_steps = max_dec_steps
        self.rnn = self.build_rnn(self._rnn_type, self._dec_input_size,
                                  self._dec_hidden_size,
                                  )
        self.attn = NewAttention(enc_dim=dec_input_size, dec_dim=dec_hidden_size,
                                 attn_type=attn_type)

        self.sampling = schedule_ratio_from_ground_truth
        self.mult_orac_sampling = mult_orac_sampling
        self.CELoss = torch.nn.CrossEntropyLoss(ignore_index=-1,
                                                reduction='none')
        self._dropout = nn.Dropout(dropout)

    def comp_loss_with_oracle(self, _single_oracle_flag: bool,
                              batch_size: int,
                              decoder_outputs_prob: torch.Tensor,
                              max_dec_steps: int,
                              tgt: torch.Tensor,
                              rouge: torch.Tensor,
                              mult_orac_sampling: bool = True
                              ):
        """

        :param _single_oracle_flag:
        :param batch_size:
        :param decoder_outputs_prob: dec_time_step, batchsize, src_len
        :param max_dec_steps:
        :param tgt: batchsize, n_of_oracles, dec_time_step
        :param tgt_mask: batchsize, n_of_oracles, dec_time_step
        :param rouge: batchsize, n_of_oracles
        :param mult_orac_sampling: True=only sample one from the bag;
                    False=use all the oracles and average
        :return:
        """
        batch_size_, n_oracles, dec_step = tgt.size()
        batch_sizee_, n_ora = rouge.size()
        assert batch_sizee_ == batch_size_ == batch_size

        if mult_orac_sampling:
            # sample one from all possible data uniformly
            lists = [[0] * n_ora for _ in range(batch_size_)]
            for idx, _ in enumerate(lists):
                lists[idx][random.randint(0, n_oracles - 1)] = 1
            # for rouge
            torch_rand_idx = torch.ByteTensor(lists).to(self.device)
            # selected_rouge = torch.masked_select(rouge, torch_rand_idx)  # size: batchsz
            # for tgt, do broadcast first
            torch_rand_idx_bc = torch_rand_idx.unsqueeze(2)
            selected_tgt = torch.masked_select(tgt, torch_rand_idx_bc).view(batch_size_, dec_step)  # batch, step
            # comp loss
            batch_first_decoder_outputs_prob = torch.transpose(decoder_outputs_prob, 0, 1).contiguous()
            flatten_prob = batch_first_decoder_outputs_prob.view(batch_size * max_dec_steps, -1)
            flatten_tgt = selected_tgt.view(-1).long()
            loss_bf_msk = self.CELoss(flatten_prob, flatten_tgt)
            tgt_mask = (selected_tgt[:, :] >= 0).float()
            flatten_tgt_msk = tgt_mask.view(-1)
            loss = loss_bf_msk * flatten_tgt_msk
            total_loss = torch.mean(loss)
        else:
            batch_first_decoder_outputs_prob = torch.transpose(decoder_outputs_prob, 0, 1).contiguous()
            batch_first_decoder_outputs_prob = batch_first_decoder_outputs_prob.unsqueeze(1)
            assert batch_first_decoder_outputs_prob.size()[1] == 1
            batch_first_decoder_outputs_prob = batch_first_decoder_outputs_prob.expand(batch_size, n_ora, dec_step,
                                                                                       -1).contiguous()
            flatten_prob = batch_first_decoder_outputs_prob.view(batch_size * max_dec_steps * n_ora, -1)
            flatten_tgt = tgt.view(-1).long()  # batch, nora, step
            loss_bf_msk = self.CELoss(flatten_prob, flatten_tgt)

            tgt_mask = (tgt[:, :, :] >= 0).float()
            flatten_tgt_msk = tgt_mask.view(-1)
            loss = loss_bf_msk * flatten_tgt_msk

            rouge = rouge.unsqueeze(2).expand_as(tgt).contiguous()
            rouge = rouge.view(-1) - torch.min(rouge) + 0.1  # baseline= 0.1
            rouge_mean = torch.mean(rouge)
            weighted_loss = loss * rouge * (1 / rouge_mean)
            total_loss = torch.mean(weighted_loss)
        return total_loss

    def compute_loss(self, decoder_outputs_prob, tgt, rouge):
        max_dec_steps, batch_size, src_len = decoder_outputs_prob.size()
        loss = self.comp_loss_with_oracle(False, batch_size,
                                          decoder_outputs_prob,
                                          max_dec_steps,
                                          tgt,
                                          rouge,
                                          mult_orac_sampling=self.mult_orac_sampling
                                          )
        return loss

    def forward(self,
                context: torch.FloatTensor,  # batch size, enc sent num, dim;
                context_mask,  # batch size, enc sent num; [1,0]
                last_state: torch.FloatTensor,  # batch size, dim;
                tgt: torch.LongTensor = None,  # batch size, time step, num of samples
                # rouge=None,  # batch size, num of samples
                ):
        """

        :param context:
        :param context_mask:
        :param last_state:
        :param tgt: If tgt is available (training mode), then use tgt sometimes as the input for next step.
                    self.sampling=schedule_ratio_from_ground_truth
        :return:
        """
        batch_size, src_len, encoder_dim = context.size()
        batch_size_, num_oracles, max_dec_t = tgt.size()
        _batch_size, hidden_size = last_state.size()
        assert batch_size_ == batch_size == _batch_size

        if tgt is not None:
            max_dec_steps = tgt.size()[2]
        else:
            max_dec_steps = self._max_dec_steps

        if tgt is not None:
            if tgt.size()[1] == 1:
                # _single_oracle_flag = True
                best_tgt_as_ground_truth = tgt.squeeze(1).long()
            else:
                # _single_oracle_flag = False
                best_tgt_as_ground_truth = tgt[:, 0, :].long()  # already sorted
            assert len(best_tgt_as_ground_truth.size()) == 2
            best_tgt_as_ground_truth_mask = (best_tgt_as_ground_truth[:, :] >= 0)
        else:
            best_tgt_as_ground_truth = None
            best_tgt_as_ground_truth_mask = None

        # assert
        if best_tgt_as_ground_truth is not None:
            best_tgt_as_ground_truth = best_tgt_as_ground_truth + (best_tgt_as_ground_truth_mask.long() - 1) * (-1)
            # invalid positions in best_tgt_as_ground_truth was -1. add (0-1)*(-1)=1 to make it -1+1 = 0 = <SOS>

        decoder_outputs_logit = torch.LongTensor(max_dec_steps, batch_size).to(self.device)
        decoder_outputs_prob = torch.zeros(size=(max_dec_steps, batch_size, src_len),
                                           dtype=torch.float32, device=self.device)
        decoder_states_h = torch.zeros(size=(max_dec_steps, batch_size, hidden_size),
                                       dtype=torch.float32, device=self.device)
        decoder_states_c = torch.zeros(size=(max_dec_steps, batch_size, hidden_size),
                                       dtype=torch.float32, device=self.device)
        # decoder input now is the state
        decoder_input = last_state

        prev_attn = torch.zeros((batch_size, src_len)).to(self.device)
        prev_attn[:, 0] = 1.  # never attend to <SOS>!

        # all_attentions = torch.empty((max_dec_steps, batch_size, src_len),
        #                              device=self.device)

        _init_zero = torch.zeros_like(last_state, device=self.device)
        state = (_init_zero, _init_zero)  # TODO use zero as h_0?

        for t in range(max_dec_steps):
            state, prev_attn, attn, score, weighted_context = \
                self.run_forward_one(inp=decoder_input,
                                     context=context,
                                     context_mask=context_mask,
                                     prev_state=state,
                                     prev_attn=prev_attn,
                                     timestep=t
                                     )
            decoder_states_h[t] = state[0]
            decoder_states_c[t] = state[1]
            decoder_outputs_prob[t] = score
            topv, topi = score.data.topk(1)
            topi = topi.squeeze()
            # Record
            decoder_outputs_logit[t] = topi
            # all_attentions[t] = attn

            if best_tgt_as_ground_truth is not None:
                if random.random() >= self.sampling:
                    decoder_input_idx = topi
                else:
                    decoder_input_idx = best_tgt_as_ground_truth[:, t]
            else:
                decoder_input_idx = topi

            decoder_input = ptr_network_index_select(self.device,
                                                     context, decoder_input_idx)
            # decoder_input = torch.index_select(context, 1, decoder_input_idx)

        output_dict = {}
        output_dict["decoder_outputs_logit"] = decoder_outputs_logit
        output_dict["decoder_outputs_prob"] = decoder_outputs_prob
        output_dict["best_tgt_as_ground_truth"] = best_tgt_as_ground_truth
        output_dict["decoder_states"] = [decoder_states_h, decoder_states_c]
        # output_dict['all_attentions'] = all_attentions

        return output_dict

    def build_rnn(self, rnn_type: str, input_size: int,
                  hidden_size: int):
        if rnn_type == "lstm":
            return torch.nn.LSTMCell(input_size, hidden_size)
        elif rnn_type == 'gru':
            return torch.nn.GRUCell(input_size, hidden_size)
        else:
            raise NotImplementedError

    def run_forward_one(self,
                        inp: torch.FloatTensor,
                        context: torch.FloatTensor,
                        context_mask: torch.FloatTensor,
                        prev_state: Tuple[torch.FloatTensor, torch.FloatTensor],
                        prev_attn: torch.FloatTensor,
                        timestep: int
                        ):
        """

        :param inp: batch x hidden_size
        :param context: batch x src_len x hidden_size
        :param context_mask: batch x src_len
        :param prev_state: (batch x hidden), (batch x hidden)
        :param prev_attn: batch x src_len: prev_attn is the average of previous attns
        :return: current_raw_state,
                prev_attn,
                attention_distribution,
                penaltied_score,
                weighted_context: batch, dim. use attn distribution to reweight context
        """

        batch_size, src_len, hidden_size = context.size()

        current_raw_state = self.rnn(inp,
                                     prev_state)  # rnn_output: batch x hiddensize. hidden batch x hiddensize

        if self._rnn_type == 'lstm':
            assert type(current_raw_state) == tuple
            current_state_h = current_raw_state[0]
            current_state_c = current_raw_state[1]
        elif self._rnn_type == 'gru':
            current_state_h = current_raw_state
            current_state_c = current_raw_state
        else:
            raise NotImplementedError

        current_state_h = self._dropout(current_state_h)
        attention_distribution, penaltied_score = \
            self.attn.forward_one_step(context,
                                       current_state_h,
                                       context_mask,
                                       prev_attn)
        # attention_distribution: batch, context len
        # penaltied_score: batch, context len
        prev_attn = (prev_attn * (timestep + 1) + attention_distribution) / \
                    (timestep + 2)
        weighted_context = attention_distribution.unsqueeze(2) * context
        weighted_context = torch.sum(weighted_context, dim=1)
        # attn_h_weighted: batch, dim
        # a:               batch, src

        ####
        # prob = torch.log(prob + 0.0000001)
        ####
        return current_raw_state, prev_attn, attention_distribution, penaltied_score, weighted_context
