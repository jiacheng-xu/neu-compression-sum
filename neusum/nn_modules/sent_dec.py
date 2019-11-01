import random
from typing import List, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from neusum.nn_modules.attention import NewAttention
from neusum.service.nn_services import ptr_network_index_select
from neusum.service.basic_service import flip_first_two_dim, log_predict_example, checkNaN
from neusum.evaluation.rouge_with_pythonrouge import RougeStrEvaluation
from neusum.service.shared_asset import get_device


class SentRNNDecoder(nn.Module):
    def __init__(self,
                 rnn_type: str = 'lstm',
                 dec_hidden_size: int = 100,
                 dec_input_size: int = 50,
                 dropout: float = 0.1,
                 fixed_dec_step: int = -1,
                 max_dec_steps: int = 2,
                 min_dec_steps: int = 2,
                 schedule_ratio_from_ground_truth: float = 0.5,
                 dec_avd_trigram_rep: bool = True,
                 mult_orac_sample_one: bool = True,
                 abs_board_file="/home/cc/exComp/board.txt",
                 valid_tmp_path='/scratch/cluster/jcxu/exComp',
                 serilization_name: str = ""
                 ):
        super().__init__()
        self.device = get_device()
        self._rnn_type = rnn_type
        self._dec_input_size = dec_input_size
        self._dec_hidden_size = dec_hidden_size

        self.fixed_dec_step = fixed_dec_step
        if fixed_dec_step == -1:
            self.min_dec_steps = min_dec_steps
            self.max_dec_steps = max_dec_steps
        else:
            self.min_dec_steps, self.max_dec_steps = fixed_dec_step, fixed_dec_step
        self.schedule_ratio_from_ground_truth = schedule_ratio_from_ground_truth
        self.mult_orac_sample_one_as_gt = mult_orac_sample_one
        self._dropout = nn.Dropout(dropout)

        self.rnn = self.build_rnn(self._rnn_type, self._dec_input_size,
                                  self._dec_hidden_size,
                                  )
        self.rnn_init_state_h = torch.nn.Linear(dec_hidden_size, dec_hidden_size)
        self.rnn_init_state_c = torch.nn.Linear(dec_hidden_size, dec_hidden_size)

        self.attn = NewAttention(enc_dim=dec_input_size, dec_dim=dec_hidden_size
                                 )
        self.CELoss = torch.nn.CrossEntropyLoss(ignore_index=-1,
                                                reduction='none')  # TODO
        self.rouge_metrics_sent = RougeStrEvaluation(name='sent',
                                                     path_to_valid=valid_tmp_path,
                                                     writting_address=valid_tmp_path,
                                                     serilization_name=serilization_name)
        self.dec_avd_trigram_rep = dec_avd_trigram_rep

    def get_output_dim(self):
        return self._dec_hidden_size

    def forward(self, context: torch.FloatTensor,  # batch size, enc sent num, dim;
                context_mask,  # batch size, enc sent num; [1,0]
                last_state: torch.FloatTensor,  # batch size, dim;
                tgt: torch.LongTensor = None  # batch size,  num of samples,time step
                ):
        batch_size, src_len, encoder_dim = context.size()
        _batch_size, hidden_size = last_state.size()
        assert batch_size == _batch_size
        if tgt is not None:
            batch_size_, num_oracles, max_dec_t = tgt.size()
            assert batch_size_ == _batch_size

        if tgt is not None:
            max_dec_steps = tgt.size()[2]
        else:
            max_dec_steps = self.max_dec_steps

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

        decoder_outputs_logit = torch.zeros(size=(max_dec_steps, batch_size), device=self.device)
        decoder_outputs_score = torch.zeros(size=(max_dec_steps, batch_size, src_len),
                                            dtype=torch.float32, device=self.device)
        decoder_states_h = torch.zeros(size=(max_dec_steps, batch_size, hidden_size),
                                       dtype=torch.float32, device=self.device)
        decoder_states_c = torch.zeros(size=(max_dec_steps, batch_size, hidden_size),
                                       dtype=torch.float32, device=self.device)
        decoder_input = last_state

        # prev_attn = torch.zeros((batch_size, src_len)).to(self.device)
        # prev_attn[:, 0] = 1.  # never attend to <SOS>!
        prev_attn = None
        # _init_unif = torch.rand_like(last_state, device=self.device) - 0.5
        # _init_unif = torch.zeros_like(last_state, device=self.device)
        _ones = torch.ones((_batch_size, hidden_size), device=self.device)
        init_h = self.rnn_init_state_h(_ones)
        init_c = self.rnn_init_state_c(_ones)
        state = (init_h, init_c)  # TODO use unif(+-0.5) as h_0?
        # print(max_dec_steps)
        for t in range(max_dec_steps):
            state, prev_attn, attn, score, weighted_context = \
                self._forward_step(inp=decoder_input,
                                   context=context,
                                   context_mask=context_mask,
                                   prev_state=state,
                                   prev_attn=prev_attn,
                                   timestep=t
                                   )
            # checkNaN(attn)
            decoder_states_h[t] = state[0]
            decoder_states_c[t] = state[1]
            decoder_outputs_score[t] = score
            topv, topi = score.data.topk(1)
            topi = topi.squeeze(1)
            # Record
            decoder_outputs_logit[t] = topi
            # all_attentions[t] = attn

            if random.random() < 0.0005:
                # print("\nAttn: {}".format(attn[1]))
                # print("Score: {}".format(score[1]))
                pass
            if best_tgt_as_ground_truth is not None:
                if random.random() < self.schedule_ratio_from_ground_truth:
                    decoder_input_idx = topi
                else:
                    decoder_input_idx = best_tgt_as_ground_truth[:, t]
            else:
                decoder_input_idx = topi
            try:
                decoder_input = ptr_network_index_select(self.device,
                                                         context, decoder_input_idx)
                decoder_input = self._dropout(decoder_input)
            except IndexError:
                print('-' * 50)
                print(batch_size_)
                print(context)
                print(decoder_input_idx)
                print(context.size())
                print(decoder_input_idx.size())
                exit()
        # print(self.rnn.weight_hh)
        # checkNaN(decoder_outputs_logit)
        # checkNaN(decoder_outputs_score)
        return decoder_outputs_logit, decoder_outputs_score, [decoder_states_h, decoder_states_c]

    def _forward_step(self, inp: torch.FloatTensor,
                      context: torch.FloatTensor,
                      context_mask: torch.FloatTensor,
                      prev_state: Tuple[torch.FloatTensor, torch.FloatTensor],
                      prev_attn: torch.FloatTensor,
                      timestep: int):

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
        # prev_attn = (prev_attn * (timestep + 1) + attention_distribution) / \
        #             (timestep + 2)
        weighted_context = attention_distribution.unsqueeze(2) * context
        weighted_context = torch.sum(weighted_context, dim=1)
        # attn_h_weighted: batch, dim
        # a:               batch, src
        # print(attention_distribution)
        ####
        # prob = torch.log(prob + 0.0000001)
        ####

        return current_raw_state, prev_attn, attention_distribution, penaltied_score, weighted_context

    def build_rnn(self, rnn_type: str, input_size: int,
                  hidden_size: int):
        if rnn_type == "lstm":
            return torch.nn.LSTMCell(input_size, hidden_size)
        elif rnn_type == 'gru':
            return torch.nn.GRUCell(input_size, hidden_size)
        else:
            raise NotImplementedError

    def extract_trigram_feature(self, inp: List[str]):
        l = len(inp)
        features = ["_".join([inp[i], inp[i + 1], inp[i + 2]]) for i in range(l - 2)]
        return features

    def _decode_one_sample(self, name: str, abs_list: List,
                           doc_list: List,
                           sent_label, dec_prob_t, max_dec_step: int, min_dec_step: int):
        _abs = [" ".join(a_list) for a_list in abs_list]

        total_sent_num = len(doc_list)
        dec_prob_t[:, total_sent_num:] = -100
        max_idx = torch.argmax(dec_prob_t, dim=1).tolist()  # first choice. [time, source sentence]
        # dec_prob_t[:, 0] = -100
        # backup_max_idx = torch.argmax(dec_prob_t, dim=1).tolist()
        sent_ids = []
        sent_words = []
        _max_count = 8
        t = 0

        while (t < max_dec_step):
            # propose

            trail_num = 0
            while True:
                try:
                    trail_num += 1
                    sent_id = max_idx[t]

                    proposed_words = doc_list[sent_id]
                    if sent_id in sent_ids:
                        raise IndexError
                except IndexError:
                    if trail_num == _max_count:
                        # print(dec_prob_t)
                        print("Not enough sents selected!")
                        break
                    dec_prob_t[t, sent_id] -= 10
                    max_idx = torch.argmax(dec_prob_t, dim=1).tolist()
                    continue
                sent_ids.append(sent_id)
                sent_words.append(proposed_words)
                break
            t += 1
            #     if sent_id == 0:
            #         if t < min_dec_step:
            #             sent_id = backup_max_idx[t]
            #             # assert sent_id != 0
            #             if sent_id == 0:
            #                 break
            #         else:
            #             stop_flag = True
            #             sent_ids.append(sent_id)
            #             break
            #     else:
            #         try:
            #             proposed_words = doc_list[sent_id]
            #         except IndexError:
            #             # print("Invalid choice. {}".format(sent_id))
            #             pass
            #             break
            #
            #         # if self.dec_avd_trigram_rep:
            #         #     proposed_features = self.extract_trigram_feature(proposed_words)
            #         #     trigram_flag = any([True for f in proposed_features if f in trigram_features])
            #         # else:
            #         #     trigram_flag = False
            #         if trigram_flag or (sent_id in sent_ids):  # no rep
            #             dec_prob_t[t, sent_id] -= 1
            #             backup_max_idx = torch.argmax(dec_prob_t, dim=1).tolist()
            #             sent_id = backup_max_idx[t]
            #         else:
            #             sent_ids.append(sent_id)
            #             sent_words.append(proposed_words)
            #             # trigram_features += proposed_features
            #             break
            # t += 1
        # _pred = [" ".join(doc_list[i]) for i in sent_ids]
        _pred = [" ".join(x) for x in sent_words]
        # print to screen
        if random.random() < 0.01:
            # print("sent_ids: {}".format(sent_ids))
            # print("sent_lab: {}".format(sent_label))
            log_predict_example(name, sent_ids, sent_label, _pred, _abs)
        # sent_ids.append(0)
        # Add to globad evaluation bag
        self.rouge_metrics_sent(pred=_pred, ref=[_abs])
        return _pred, _abs, sent_ids

    def decode(self, decoder_outputs_prob, metadata, sent_label=None):

        batch, timestep, sent_num = decoder_outputs_prob.size()
        batch_ = len(metadata)
        part = metadata[0]['part']
        assert batch == batch_
        if part == 'cnn':
            # max_dec_steps, min_dec_steps = self.max_dec_steps - 1, self.min_dec_steps - 1
            max_dec_steps, min_dec_steps = self.max_dec_steps , self.min_dec_steps
        elif part == 'dm':
            max_dec_steps, min_dec_steps = self.max_dec_steps, self.min_dec_steps
        elif part == 'nyt':
            max_dec_steps, min_dec_steps = self.max_dec_steps, self.min_dec_steps
        else:
            raise NotImplementedError
        dec_prob = decoder_outputs_prob.cpu().data
        batch_sent_decoding_result = torch.ones((max_dec_steps, batch_), dtype=torch.long, device=self.device) * -1
        if sent_label is not None:
            sent_label = sent_label[:, 0, :].cpu().data

        for idx, m in enumerate(metadata):
            # _pred = []
            name = m['name']
            part = m['part']
            dec_prob_t = dec_prob[idx]  # timestep, source len
            if sent_label is not None:
                label = sent_label[idx].tolist()
            else:
                label = []

            _pred, _abs, sent_ids = self._decode_one_sample(name=name,
                                                            abs_list=m["abs_list"], doc_list=m["doc_list"],
                                                            sent_label=label,
                                                            dec_prob_t=dec_prob_t, max_dec_step=max_dec_steps,
                                                            min_dec_step=min_dec_steps)

            # record
            for wt_step in range(len(sent_ids)):
                batch_sent_decoding_result[wt_step, idx] = int(sent_ids[wt_step])

        # print(batch_sent_decoding_result)
        return batch_sent_decoding_result

    def comp_loss(self,
                  decoder_outputs_prob: torch.Tensor,  # batch, timestep, prob_over_source
                  oracles: torch.Tensor,  # batchsize, n_of_oracles, dec_time_step
                  rouge: torch.Tensor,  # batchsize, n_of_oracles
                  ):
        batch, time, _ = decoder_outputs_prob.size()
        batch_size_, n_oracles, dec_step = oracles.size()  # invalid position = -1
        batch_sizee_, n_ora = rouge.size()  # invalid position = 0
        assert batch_sizee_ == batch_size_ == batch
        assert time == dec_step
        try:
            assert n_ora == n_oracles
        except AssertionError:
            print(oracles)
            print(rouge)
        decoder_outputs_prob = decoder_outputs_prob.contiguous()
        # batch_first_decoder_outputs_prob = torch.transpose(decoder_outputs_prob, 0, 1).contiguous()

        if self.mult_orac_sample_one_as_gt:
            lists = [[0] * n_ora for _ in range(batch_size_)]
            for idx, _ in enumerate(lists):
                lists[idx][random.randint(0, n_oracles - 1)] = 1
            torch_rand_idx = torch.ByteTensor(lists).to(self.device)
            torch_rand_idx_bc = torch_rand_idx.unsqueeze(2)
            selected_tgt = torch.masked_select(oracles, torch_rand_idx_bc).view(batch_size_, dec_step)  # batch, step
            flatten_prob = decoder_outputs_prob.view(batch_size_ * dec_step, -1)
            flatten_tgt = selected_tgt.view(-1).long()
            loss_bf_msk = self.CELoss(flatten_prob, flatten_tgt)
            tgt_mask = (selected_tgt[:, :] >= 0).float()
            flatten_tgt_msk = tgt_mask.view(-1)
            loss = loss_bf_msk * flatten_tgt_msk
            total_loss = torch.mean(loss)
            return total_loss, total_loss
        else:
            batch_first_decoder_outputs_prob = decoder_outputs_prob.unsqueeze(1)
            assert batch_first_decoder_outputs_prob.size()[1] == 1
            batch_first_decoder_outputs_prob = batch_first_decoder_outputs_prob.expand(batch_size_, n_ora, dec_step,
                                                                                       -1).contiguous()
            flatten_prob = batch_first_decoder_outputs_prob.view(batch_size_ * dec_step * n_ora, -1)
            flatten_tgt = oracles.contiguous().view(-1).long()  # batch, nora, step
            # print(flatten_tgt)
            # TODO why do i need this....
            flatten_tgt = torch.nn.functional.relu(flatten_tgt.float()).long()
            # print(flatten_prob)
            loss_bf_msk = self.CELoss(flatten_prob, flatten_tgt)
            # print(loss_bf_msk)
            tgt_mask = (oracles[:, :, :] >= 0).float()
            flatten_tgt_msk = tgt_mask.view(-1)
            loss_ori = loss_bf_msk * flatten_tgt_msk
            rouge = rouge.unsqueeze(2).expand_as(oracles).contiguous()
            rouge = (rouge.view(-1))  # baseline= 0.1
            # rouge_mean = torch.mean(rouge)
            # weighted_loss = loss * rouge * (1 / rouge_mean)
            weighted_loss = loss_ori * rouge
            # print("="*100)
            loss_ori = torch.mean(loss_ori)
            total_loss = torch.mean(weighted_loss)
            return total_loss, loss_ori  # use original loss!!!
