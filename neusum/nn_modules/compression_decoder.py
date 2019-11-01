import itertools

import torch
import torch.nn

from neusum.nn_modules.attention import NewAttention
from neusum.evaluation.rouge_with_pythonrouge import RougeStrEvaluation
import random
import allennlp
from neusum.nn_modules.enc.enc_compression import EncCompression
from neusum.service.basic_service import log_predict_example, log_compression_example, log_universal
from allennlp.modules.feedforward import FeedForward
from typing import List
from neusum.service.shared_asset import get_device

import numpy as np
from neusum.service.basic_service import easy_post_processing

# Given infomation, output the compressions
# with iterative removal <= need to be a param?
import copy, json

sp_tok = "_-"
sp_tok_rep = "^_"
import os


class CompExecutor():

    def __init__(self, span_meta,
                 sent_idxs,  # #t
                 prediction_score,  # t, # max_comp, 2
                 abs_str: List[str],
                 name, doc_list, keep_threshold: List[float], part: str,
                 ser_fname, ser_dir):
        self.abs_str = abs_str
        self.span_meta = span_meta
        self.sent_idxs = sent_idxs
        self.prediction_score = prediction_score
        self.name = name
        self.doc_list = doc_list
        self.keep_threshold = keep_threshold
        self.dec_sents = []

        self.full_sents = []
        self.compressions: List[OrderedDict] = []
        self.del_record = [[] for _ in range(len(self.keep_threshold))]
        self.part = part
        self.ser_fname = os.path.join(ser_dir, ser_fname)

    def run(self) -> (List, List, List, List):
        # go through everything
        _pred = [[] for _ in range(len(self.keep_threshold))]
        # _visuals = [[] for _ in range(len(self.keep_threshold))]

        # first keep all of the compressions in record
        self.read_sent_record_compressions(self.sent_idxs)

        # start diverging
        # delete those under threshold
        self.del_under_threshold()  # diverge!
        # iterate: delete those already covered in context
        processed_words = self.iterative_rep_del()
        # reorder
        # sent_order: List[int]
        order = np.argsort(self.sort_order)
        # output
        for kepidx, kep in enumerate(self.keep_threshold):
            processed_words[kepidx] = [processed_words[kepidx][o] for o in order]

        # output something for evaluation
        # bag_pred_eval = [[] for _ in range(len(self.keep_threshold))]
        bag_pred_eval = []
        for i, words in enumerate(processed_words):
            _tmp = []
            for j, sent in enumerate(words):
                sent = [x for x in sent if (not x.startswith(sp_tok)) and (not x.startswith(sp_tok_rep))]
                out = easy_post_processing(" ".join(sent))
                _tmp.append(out)
            bag_pred_eval.append(_tmp)

        # (optional) visualization

        if random.random() < 0.005:
            try:
                logger = logging.getLogger()
                logger.info("Prob\t\tType\t\tRatio\t\tRouge\t\tLen\t\tContent")
                for idx, d in enumerate(self.compressions):
                    for key, value in d.items():
                        wt = [value['prob'], value['type'], value['ratio'], value['rouge'], value['len'], key]
                        wt = "\t\t".join([str(x) for x in wt])
                        logger.info(wt)
                log_universal(Partition=self.part, Name=self.name,
                              Abs=self.abs_str
                              )
                for idx in range(len(self.keep_threshold)):
                    lis = processed_words[idx]
                    lis_out = [" ".join(x) for x in lis]
                    log_universal(Kep=self.keep_threshold[idx],
                                  Visual=" | ".join(lis_out))
                # write del_record to disk
                f = open(self.ser_fname, 'a')
                js = json.dumps(self.del_record)
                f.write("\n")
                f.write(js)
                f.close()
            except ZeroDivisionError:
                pass

        # return processed_words, self.del_record, self.compressions, self.full_sents, bag_pred_eval
        return bag_pred_eval

    def del_under_threshold(self):
        for idx, sent_stat_dict in enumerate(self.compressions):  # idx is the num of sent
            for key, value in sent_stat_dict.items():
                p = value['prob']
                sel = value['sel_idx']
                word_sent = self.full_sents[idx]
                selected_words = [x for k, x in enumerate(word_sent) if k in sel]
                for th_idx, thres in enumerate(self.keep_threshold):
                    if p > thres:
                        self.removal[th_idx][idx] = set(value['sel_idx']).union(self.removal[th_idx][idx])
                        self.del_record[th_idx].append({'type': value['type'],
                                                        'len': value['len'],
                                                        'active': 1,
                                                        'word': selected_words,
                                                        'ratio': value['ratio'], 'prob': p})
                    else:
                        self.consider[th_idx][idx] += [
                            {'type': value['type'],
                             "sel_word": selected_words,
                             "sel_idx": value['sel_idx'],
                             "len": value['len'], 'ratio': value['ratio'],
                             'prob': p
                             }
                        ]
                    # self consider is a List[List[ dict, dict,...]]

    def iterative_rep_del(self):

        # apply self.removal
        curret_words = [copy.deepcopy(self.full_sents) for _ in range(len(self.keep_threshold))]
        for idx, rm in enumerate(self.removal):  # th first, then sent idx
            for sent_idx, rm_of_sent in enumerate(rm):
                list_rm_of_sent = list(rm_of_sent)

                _tmp = []
                for word_id, word in enumerate(curret_words[idx][sent_idx]):
                    if word_id not in list_rm_of_sent:
                        _tmp.append(word)
                    else:
                        _tmp.append(sp_tok + word + sp_tok)
                curret_words[idx][sent_idx] = _tmp

        for thidx, consider in enumerate(self.consider):  # List[th List[sent[ [dict, dict,..]]]
            for sent_idx, consider_of_sent in enumerate(consider):
                consider_of_sent.sort(key=lambda x: x['len'], reverse=True)
                for c in consider_of_sent:
                    c_words = c['sel_word']
                    c_words = [x.lower() for x in c_words]
                    mini = Counter(c_words)
                    current_full = list(itertools.chain(*curret_words[thidx]))
                    current_full = [x.lower() for x in current_full]
                    cnt = Counter(current_full)
                    if (len(cnt) == len(cnt - mini)) and (cnt != (cnt - mini)):
                        self.del_record[thidx].append({'type': c['type'],
                                                       'len': c['len'],
                                                       'active': 0,
                                                       'word': c['sel_word'],
                                                       'ratio': c['ratio'], 'prob': c['prob']})
                        todo = c['sel_idx']
                        for t in todo:
                            curret_words[thidx][sent_idx][t] = sp_tok_rep + curret_words[thidx][sent_idx][
                                t] + sp_tok_rep
        return curret_words

    def read_sent_record_compressions(self, sent_idxs):
        # shared by all of the kepth
        sort_order = []  # the original order of sent_idx   # siez=t
        for j, sent_idx in enumerate(sent_idxs):
            if sent_idx < 0:
                continue
            if len(self.doc_list) <= sent_idx:
                continue  # lead3 not enough
            sort_order.append(int(sent_idx))
            pred_score = self.prediction_score[j, :, :]  # max_comp, 2
            sp_meta = self.span_meta[sent_idx]
            word_sent: List[str] = self.doc_list[sent_idx]
            self.full_sents.append(word_sent)
            # Show all of the compression spans
            stat_compression = {}
            for comp_idx, comp_meta in enumerate(sp_meta):
                p = pred_score[comp_idx][1]
                node_type, sel_idx, rouge, ratio = comp_meta
                if node_type != "BASELINE":
                    selected_words = [x for idx, x in enumerate(word_sent) if idx in sel_idx]
                    selected_words_str = "_".join(selected_words)
                    stat_compression["{}".format(selected_words_str)] = {
                        "prob": float("{0:.2f}".format(p)),  # float("{0:.2f}".format())
                        "type": node_type,
                        "rouge": float("{0:.2f}".format(rouge)),
                        "ratio": float("{0:.2f}".format(ratio)),
                        "sel_idx": sel_idx,
                        "len": len(sel_idx)
                    }
            stat_compression_order = OrderedDict(
                sorted(stat_compression.items(), key=lambda item: item[1]["prob"], reverse=True))  # Python 3
            self.compressions.append(stat_compression_order)
        self.sort_order = sort_order
        self.removal = [[set() for _ in range(len(sort_order))] for _ in
                        range(len(self.keep_threshold))]  # thresholds,t
        self.consider = [[[] for _ in range(len(sort_order))] for _ in range(len(self.keep_threshold))]


def s(x):
    print(x.size())


import json
import logging
import numpy


def two_dim_index_select(inp, index):
    """
    retrieve [0,a] [1,b] [2,c]
    :param inp: [batch, max_sent_num, ....]
    :param index: len=batch [a,b,c,..]
    :return:
    """
    batch_size = inp.size()[0]
    max_sent_num = inp.size()[1]
    if len(inp.size()) == 2:
        comb_first_two_dim_inp = inp.view(batch_size * max_sent_num)
    elif len(inp.size()) == 3:
        left_dim = inp.size()[2]
        comb_first_two_dim_inp = inp.view(batch_size * max_sent_num, left_dim)
    elif len(inp.size()) == 4:
        left_dim = inp.size()[2]
        sec_left_dim = inp.size()[3]
        comb_first_two_dim_inp = inp.view(batch_size * max_sent_num, left_dim, sec_left_dim)
    else:
        raise NotImplementedError
    # print(index)
    local_index = torch.zeros_like(index)  # TODO device issue
    for idx, i in enumerate(index):
        local_index[idx] = idx * max_sent_num + i
    # print(index.size())
    # print(comb_first_two_dim_inp.size())
    # print(index)
    selected = torch.index_select(comb_first_two_dim_inp, dim=0, index=local_index)
    return selected


from collections import OrderedDict, Counter


class CompressDecoder(torch.nn.Module):
    def __init__(self,
                 context_dim,
                 dec_state_dim,
                 enc_hid_dim,
                 text_field_embedder,
                 aggressive_compression: int = -1,
                 keep_threshold: float = 0.5,
                 abs_board_file="/home/cc/exComp/board.txt",
                 gather='mean',
                 dropout=0.5,
                 dropout_emb=0.2,
                 valid_tmp_path='/scratch/cluster/jcxu/exComp',
                 serilization_name: str = "",
                 vocab=None,
                 elmo: bool = False,
                 elmo_weight: str = "elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"):
        super().__init__()
        self.use_elmo = elmo
        self.serilization_name = serilization_name
        if elmo:
            from allennlp.modules.elmo import Elmo, batch_to_ids
            from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
            self.vocab = vocab

            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
            weight_file = elmo_weight
            self.elmo = Elmo(options_file, weight_file, 1, dropout=dropout_emb)
            # print(self.elmo.get_output_dim())
            # self.word_emb_dim = text_field_embedder.get_output_dim()
            # self._context_layer = PytorchSeq2SeqWrapper(
            #     torch.nn.LSTM(self.word_emb_dim + self.elmo.get_output_dim(), self.word_emb_dim,
            #                   batch_first=True, bidirectional=True))
            self.word_emb_dim = self.elmo.get_output_dim()
        else:
            self._text_field_embedder = text_field_embedder
            self.word_emb_dim = text_field_embedder.get_output_dim()

        self.XEloss = torch.nn.CrossEntropyLoss(reduction='none')
        self.device = get_device()

        # self.rouge_metrics_compression = RougeStrEvaluation(name='cp', path_to_valid=valid_tmp_path,
        #                                                     writting_address=valid_tmp_path,
        #                                                     serilization_name=serilization_name)
        # self.rouge_metrics_compression_best_possible = RougeStrEvaluation(name='cp_ub', path_to_valid=valid_tmp_path,
        #                                                                   writting_address=valid_tmp_path,
        #                                                                   serilization_name=serilization_name)
        self.enc = EncCompression(inp_dim=self.word_emb_dim, hid_dim=enc_hid_dim, gather=gather)  # TODO dropout

        self.aggressive_compression = aggressive_compression
        self.relu = torch.nn.ReLU()

        self.attn = NewAttention(enc_dim=self.enc.get_output_dim(),
                                 dec_dim=self.enc.get_output_dim_unit() * 2 + dec_state_dim)

        self.concat_size = self.enc.get_output_dim() + self.enc.get_output_dim_unit() * 2 + dec_state_dim
        self.valid_tmp_path = valid_tmp_path
        if self.aggressive_compression < 0:
            self.XELoss = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
            # self.nn_lin = torch.nn.Linear(self.concat_size, self.concat_size)
            # self.nn_lin2 = torch.nn.Linear(self.concat_size, 2)

            self.ff = FeedForward(input_dim=self.concat_size, num_layers=3,
                                  hidden_dims=[self.concat_size, self.concat_size, 2],
                                  activations=[torch.nn.Tanh(), torch.nn.Tanh(), lambda x: x],
                                  dropout=dropout
                                  )
            # Keep thresold

            # self.keep_thres = list(np.arange(start=0.2, stop=0.6, step=0.075))
            self.keep_thres = [0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
            self.rouge_metrics_compression_dict = OrderedDict()
            for thres in self.keep_thres:
                self.rouge_metrics_compression_dict["{}".format(thres)] = RougeStrEvaluation(name='cp_{}'.format(thres),
                                                                                             path_to_valid=valid_tmp_path,
                                                                                             writting_address=valid_tmp_path,
                                                                                             serilization_name=serilization_name)

    def encode_sent_and_span_paral(self, text,  # batch, max_sent, max_word
                                   text_msk,  # batch, max_sent, max_word
                                   span,  # batch, max_sent_num, max_span_num, max_word
                                   sent_idx  # batch size
                                   ):
        this_text = two_dim_index_select(text['tokens'], sent_idx)  # batch, max_word
        from allennlp.modules.elmo import batch_to_ids
        if self.use_elmo:
            this_text_list: List = this_text.tolist()
            text_str_list = []
            for sample in this_text_list:
                s = [self.vocab.get_token_from_index(x) for x in sample]
                text_str_list.append(s)
            character_ids = batch_to_ids(text_str_list).to(self.device)
            this_context = self.elmo(character_ids)
            # print(this_context['elmo_representations'][0].size())
            this_context = this_context['elmo_representations'][0]
        else:
            this_text = {'tokens': this_text}
            this_context = self._text_field_embedder(this_text)

        num_doc, max_word, inp_dim = this_context.size()
        batch_size = sent_idx.size()[0]
        assert batch_size == num_doc

        # text is the original text of the selected sentence.
        # this_context = two_dim_index_select(context, sent_idx)  # batch, max_word, hdim
        this_context_mask = two_dim_index_select(text_msk, sent_idx)  # batch, max_word
        this_span = two_dim_index_select(span, sent_idx)  # batch , nspan, max_word

        concat_rep_of_compression, \
        span_msk, original_sent_rep = self.enc.forward(word_emb=this_context,
                                                       word_emb_msk=this_context_mask,
                                                       span=this_span)
        return concat_rep_of_compression, span_msk, original_sent_rep

    def encode_sent_and_span(self, text, text_msk, span, batch_idx, sent_idx):
        context = self._text_field_embedder(text)
        num_doc, max_sent, max_word, inp_dim = context.size()
        num_doc_, max_sent_, nspan = span.size()[0:-1]
        assert num_doc == num_doc_
        assert max_sent == max_sent_
        this_context = context[batch_idx, sent_idx, :, :].unsqueeze(0)
        this_span = span[batch_idx, sent_idx, :, :].unsqueeze(0)
        this_context_mask = text_msk[batch_idx, sent_idx, :].unsqueeze(0)
        flattened_enc, attn_dist, \
        spans_rep, span_msk, score \
            = self.enc.forward(word_emb=this_context,
                               word_emb_msk=this_context_mask,
                               span=this_span)
        return flattened_enc, spans_rep, span_msk
        # 1, hid*2      1, span num, hid        1, span num

    def indep_compression_judger(self, reps):
        # t, batch_size_, max_span_num,self.concat_size
        timestep, batch_size, max_span_num, dim = reps.size()
        score = self.ff.forward(reps)
        # lin_out = self.nn_lin(reps)
        # activated = torch.sigmoid(lin_out)
        # score = self.nn_lin2(activated)
        if random.random() < 0.005:
            print("score: {}".format(score[0]))
        return score

    def get_out_dim(self):
        return self.concat_size

    def forward_parallel(self, sent_decoder_states,  # t, batch, hdim
                         sent_decoder_outputs_logit,  # t, batch
                         document_rep,  # batch, hdim
                         text,  # batch, max_sent, max_word
                         text_msk,  # batch, max_sent, max_word
                         span):  # batch, max_sent_num, max_span_num, max_word
        # Encode compression options given sent emission.
        # output scores, attn dist, ...
        t, batch_size, hdim = sent_decoder_states.size()
        t_, batch_size_ = sent_decoder_outputs_logit.size()  # invalid bits are -1
        batch, max_sent, max_span_num, max_word = span.size()
        # assert t == t_
        t = min(t, t_)
        assert batch_size == batch == batch_size_
        if self.aggressive_compression > 0:
            all_attn_dist = torch.zeros((t, batch_size, max_span_num)).to(self.device)
            all_scores = torch.ones((t, batch_size, max_span_num)).to(self.device) * -100
        else:
            all_attn_dist = None
            all_scores = None
        all_reps = torch.zeros((t, batch_size_, max_span_num, self.concat_size), device=self.device)
        for timestep in range(t):
            dec_state = sent_decoder_states[timestep]  # batch, dim
            logit = sent_decoder_outputs_logit[timestep]  # batch

            # valid_mask = (logit > 0)
            positive_logit = self.relu(logit.float()).long()  # turn -1 to 0

            span_t, span_msk_t, sent_t = self.encode_sent_and_span_paral(text=text,
                                                                         text_msk=text_msk,
                                                                         span=span,
                                                                         sent_idx=positive_logit)
            # sent_t : batch, sent_dim
            # span_t: batch, span_num, span_dim
            # span_msk_t: batch, span_num [[1,1,1,0,0,0],

            concated_rep_high_level = torch.cat([dec_state, document_rep, sent_t], dim=1)
            # batch, DIM
            if self.aggressive_compression > 0:
                attn_dist, score = self.attn.forward_one_step(enc_state=span_t,
                                                              dec_state=concated_rep_high_level,
                                                              enc_mask=span_msk_t.float())
            # attn_dist: batch, span num
            # score:    batch, span num

            # concated_rep: batch, dim ==> batch, 1, dim ==> batch, max_span_num, dim
            expanded_concated_rep = concated_rep_high_level.unsqueeze(1).expand((batch, max_span_num, -1))
            all_reps[timestep, :, :, :] = torch.cat([expanded_concated_rep, span_t], dim=2)
            if self.aggressive_compression > 0:
                all_attn_dist[timestep, :, :] = attn_dist
                all_scores[timestep, :, :] = score

        return all_attn_dist, all_scores, all_reps

    def comp_loss_inf_deletion(self,
                               decoder_outputs_logit,  # gold label!!!!
                               # span_seq_label,  # batch, max sent num
                               span_rouge,  # batch, max sent num, max compression num
                               scores,
                               comp_rouge_ratio,
                               loss_thres=1
                               ):
        """

        :param decoder_outputs_logit:
        :param span_rouge: [batch, max_sent, max_compression]
        :param scores: [timestep, batch, max_compression, 2]
        :param comp_rouge_ratio: [batch_size, max_sent, max_compression]
        :return:
        """
        tim, bat = decoder_outputs_logit.size()
        time, batch, max_span, _ = scores.size()
        batch_, sent_len, max_sp = span_rouge.size()
        assert batch_ == batch == bat
        assert time == tim
        assert max_sp == max_span
        goal_rouge_label = torch.ones((tim, batch, max_span), device=self.device, dtype=torch.long,
                                      ) * (-1)
        weights = torch.ones((tim, batch, max_span), device=self.device, dtype=torch.float)
        decoder_outputs_logit_mask = (decoder_outputs_logit >= 0).unsqueeze(2).expand(
            (time, batch, max_span)).float().view(-1)
        decoder_outputs_logit = torch.nn.functional.relu(decoder_outputs_logit).long()
        z = torch.zeros((1), device=self.device)
        for tt in range(tim):
            decoder_outputs_logit_t = decoder_outputs_logit[tt]
            out = two_dim_index_select(inp=comp_rouge_ratio, index=decoder_outputs_logit_t)
            label = torch.gt(out, loss_thres).long()

            mini_mask = torch.gt(out, 0.01).float()

            # baseline_mask = 1 - torch.lt(torch.abs(out - 0.99), 0.01).float()  # baseline will be 0

            # weight = torch.max(input=-out + 0.5, other=z) + 1
            # weights[tt] = mini_mask * baseline_mask
            weights[tt] = mini_mask
            goal_rouge_label[tt] = label
        probs = scores.view(-1, 2)
        goal_rouge_label = goal_rouge_label.view(-1)
        weights = weights.view(-1)
        loss = self.XELoss(input=probs, target=goal_rouge_label)
        loss = loss * decoder_outputs_logit_mask * weights
        return torch.mean(loss)

    def comp_loss(self, decoder_outputs_logit,  # gold label!!!!
                  scores,
                  span_seq_label,  # batch, max sent num
                  span_rouge,  # batch, max sent num, max compression num
                  comp_rouge_ratio
                  ):
        t, batch = decoder_outputs_logit.size()
        t_, batch_, comp_num = scores.size()
        b, max_sent = span_seq_label.size()
        # b_, max_sen, max_comp_, _ = span.size()
        _b, max_sent_, max_comp = span_rouge.size()
        assert batch == batch_ == b == _b
        assert max_sent_ == max_sent
        assert comp_num == max_comp
        span_seq_label = span_seq_label.long()
        total_loss = torch.zeros((t, b)).to(self.device)
        # print(decoder_outputs_logit)
        # print(span_seq_label)
        for timestep in range(t):

            # this is the sent idx
            for batch_idx in range(b):
                logit = decoder_outputs_logit[timestep][batch_idx]
                # print(logit)
                # decoder_outputs_logit should be the gold label for sentence emission.
                # if it's 0 or -1, then we skip supervision.
                if logit < 0:
                    continue
                ref_rouge_score = comp_rouge_ratio[batch_idx][logit]
                num_of_compression = ref_rouge_score.size()[0]

                _supervision_label_msk = (ref_rouge_score > 0.98).float()
                label = torch.from_numpy(np.arange(num_of_compression)).to(self.device).long()
                score_t = scores[timestep][batch_idx].unsqueeze(0)  # comp num
                score_t = score_t.expand(num_of_compression, -1)
                # label = span_seq_label[batch_idx][logit].unsqueeze(0)

                loss = self.XEloss(score_t, label)
                # print(loss)
                loss = _supervision_label_msk * loss
                total_loss[timestep][batch_idx] = torch.sum(loss)
                # sent_msk_t = two_dim_index_select(sent_mask, logit)

        return torch.mean(total_loss)

    def _dec_compression_one_step(self, predict_compression,
                                  sp_meta,
                                  word_sent: List[str], keep_threshold: List[float],
                                  context: List[List[str]] = None):

        full_set_len = set(range(len(word_sent)))
        # max_comp, _ = predict_compression.size

        preds = [full_set_len.copy() for _ in range(len(keep_threshold))]

        # Show all of the compression spans
        stat_compression = {}
        for comp_idx, comp_meta in enumerate(sp_meta):
            p = predict_compression[comp_idx][1]
            node_type, sel_idx, rouge, ratio = comp_meta
            if node_type != "BASELINE":
                selected_words = [x for idx, x in enumerate(word_sent) if idx in sel_idx]
                selected_words_str = "_".join(selected_words)
                stat_compression["{}".format(selected_words_str)] = {
                    "prob": float("{0:.2f}".format(p)),  # float("{0:.2f}".format())
                    "type": node_type,
                    "rouge": float("{0:.2f}".format(rouge)),
                    "ratio": float("{0:.2f}".format(ratio)),
                    "sel_idx": sel_idx,
                    "len": len(sel_idx)
                }
        stat_compression_order = OrderedDict(
            sorted(stat_compression.items(), key=lambda item: item[1]["prob"], reverse=True))  # Python 3
        for idx, _keep_thres in enumerate(keep_threshold):
            history: List[str] = context[idx]
            his_set = set((" ".join(history)).split(" "))
            for key, value in stat_compression_order.items():
                p = value['prob']
                sel_idx = value['sel_idx']
                sel_txt = set([word_sent[x] for x in sel_idx])
                if sel_txt - his_set == set():
                    # print("Save big!")
                    # print("Context: {}\tCandidate: {}".format(his_set, sel_txt))
                    preds[idx] = preds[idx] - set(value['sel_idx'])
                    continue
                if p > _keep_thres:
                    preds[idx] = preds[idx] - set(value['sel_idx'])

        preds = [list(x) for x in preds]
        for pred in preds:
            pred.sort()
        # Visual output
        visual_outputs: List[str] = []
        words_for_evaluation: List[str] = []
        meta_keep_ratio_word = []

        for idx, compression in enumerate(preds):
            output = [word_sent[jdx] if (jdx in compression) else '_' + word_sent[jdx] + '_' for jdx in
                      range(len(word_sent))]
            visual_outputs.append(" ".join(output))

            words = [word_sent[x] for x in compression]
            meta_keep_ratio_word.append(float(len(words) / len(word_sent)))
            # meta_kepp_ratio_span.append(1 - float(len(survery['type'][idx]) / len(sp_meta)))
            words = " ".join(words)
            words = easy_post_processing(words)
            # print(words)
            words_for_evaluation.append(words)
        d: List[List] = []
        for kep_th, vis, words_eva, keep_word_ratio in zip(keep_threshold, visual_outputs, words_for_evaluation,
                                                           meta_keep_ratio_word):
            d.append([kep_th, vis, words_eva, keep_word_ratio])
        return stat_compression_order, d

    def decode_inf_deletion(self,
                            sent_decoder_outputs_logit,  # time, batch
                            span_prob,  # time, batch, max_comp, 2
                            metadata: List,
                            span_meta: List,
                            span_rouge,  # batch, sent, max_comp
                            keep_threshold: List[float]
                            ):
        batch_size, max_sent_num, max_comp_num = span_rouge.size()
        t, batsz, max_comp, _ = span_prob.size()
        span_score = torch.nn.functional.softmax(span_prob, dim=3).cpu().numpy()
        timestep, batch = sent_decoder_outputs_logit.size()
        sent_decoder_outputs_logit = sent_decoder_outputs_logit.cpu().data

        for idx, m in enumerate(metadata):
            abs_s = [" ".join(s) for s in m["abs_list"]]
            comp_exe = CompExecutor(span_meta=span_meta[idx],
                                    sent_idxs=sent_decoder_outputs_logit[:, idx],
                                    prediction_score=span_score[:, idx, :, :],
                                    abs_str=abs_s,
                                    name=m['name'],
                                    doc_list=m["doc_list"],
                                    keep_threshold=keep_threshold,
                                    part=m['name'], ser_dir=self.valid_tmp_path,
                                    ser_fname=self.serilization_name
                                    )
            # processed_words, del_record, \
            # compressions, full_sents, \
            bag_pred_eval = comp_exe.run()
            full_sents: List[List[str]] = comp_exe.full_sents
            # assemble full sents
            full_sents = [" ".join(x) for x in full_sents]

            # visual to console
            for idx in range(len(keep_threshold)):
                self.rouge_metrics_compression_dict["{}".format(keep_threshold[idx])](pred=bag_pred_eval[idx],
                                                                                      ref=[abs_s], origin=full_sents
                                                                                      )

            # wt stat to disk
            # serilization_name
            # wt to rouge eval group
