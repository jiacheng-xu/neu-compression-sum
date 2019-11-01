from neusum.service.basic_service import convert_list_to_paragraph
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator, util, RegularizerApplicator
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.span_extractors import EndpointSpanExtractor, BidirectionalEndpointSpanExtractor, \
    SelfAttentiveSpanExtractor
from neusum.evaluation.rouge_with_pythonrouge import RougeStrEvaluation
from typing import Any, Dict, List, Optional, Tuple
from overrides import overrides
import torch.nn.functional as F
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
import random
from allennlp.modules.elmo import Elmo, batch_to_ids
from neusum.nn_modules.seq2idx_decoder import InputFeedRNNDecoder
from neusum.service.common_imports import *
from neusum.service.basic_service import log_predict_example
from allennlp.common.params import Params
from neusum.service.basic_service import batch_extraction_from_dict
from neusum.nn_modules.sent_dec import SentRNNDecoder
from neusum.nn_modules.compression_decoder import CompressDecoder
from neusum.service.basic_service import flip_first_two_dim, print_tensor
from neusum.nn_modules.enc.enc_doc import EncDoc
from neusum.service.shared_asset import get_device
import multiprocessing
from neusum.service.basic_service import para_get_metric


@Model.register('Seq2IdxSum')
class Seq2IdxSum(Model):
    def __init__(self,

                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,

                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,


                 word_embedding_dim: int = 200,
                 hidden_dim: int = 200,
                 dropout_emb: float = 0.5,
                 min_dec_step: int = 2,
                 max_decoding_steps=3,
                 fix_edu_num=-1,
                 dropout: float = 0.5,
                 alpha: float = 0.5,
                 span_encoder_type='self_attentive',
                 use_elmo: bool = True,
                 attn_type: str = 'general',
                 schedule_ratio_from_ground_truth: float = 0.8,
                 pretrain_embedding_file=None,
                 nenc_lay: int = 2,
                 mult_orac_sampling: bool = False,
                 word_token_indexers=None,
                 compression: bool = True,
                 dbg: bool = False,
                 dec_avd_trigram_rep: bool = True,
                 aggressive_compression: int = -1,
                 compress_leadn: int = -1,
                 subsentence: bool = False,
                 gather='mean',
                 keep_threshold: float = 0.5,
                 abs_board_file: str = "/home/cc/exComp/board.txt",
                 abs_dir_root: str = "/scratch/cluster/jcxu",
                 serilization_name: str = "",
                 ) -> None:

        super(Seq2IdxSum, self).__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder

        elmo_weight = os.path.join(abs_dir_root,
                                   "elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5")
        # if not os.path.isfile(elmo_weight):
        #     import subprocess
        #     x = "wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5 -P {}".format(abs_dir_root)
        #     subprocess.run(x.split(" "))

        self.device = get_device()
        self.vocab = vocab
        self.dbg = dbg
        self.loss_thres = keep_threshold
        self.compression = compression
        self.comp_leadn = compress_leadn
        # Just encode the whole document without looking at compression options
        self.enc_doc = EncDoc(inp_dim=word_embedding_dim, hid_dim=hidden_dim,
                              vocab=vocab, dropout=dropout, dropout_emb=dropout_emb,
                              pretrain_embedding_file=pretrain_embedding_file,
                              gather=gather
                              )

        self.sent_dec = SentRNNDecoder(rnn_type='lstm',
                                       dec_hidden_size=self.enc_doc.get_output_dim(),
                                       dec_input_size=self.enc_doc.get_output_dim(),
                                       dropout=dropout,
                                       fixed_dec_step=fix_edu_num,
                                       max_dec_steps=max_decoding_steps,
                                       min_dec_steps=min_dec_step,
                                       schedule_ratio_from_ground_truth=schedule_ratio_from_ground_truth,
                                       dec_avd_trigram_rep=dec_avd_trigram_rep,
                                       mult_orac_sample_one=mult_orac_sampling,
                                       abs_board_file=abs_board_file,
                                       valid_tmp_path=abs_dir_root,
                                       serilization_name=serilization_name
                                       )
        if compression:
            self.compression_dec = CompressDecoder(context_dim=hidden_dim * 2,
                                                   dec_state_dim=hidden_dim * 2,
                                                   enc_hid_dim=hidden_dim,
                                                   text_field_embedder=self.enc_doc._text_field_embedder,
                                                   aggressive_compression=aggressive_compression,
                                                   keep_threshold=keep_threshold,
                                                   abs_board_file=abs_board_file,
                                                   gather=gather,
                                                   dropout=dropout,
                                                   dropout_emb=dropout_emb,
                                                   valid_tmp_path=abs_dir_root,
                                                   serilization_name=serilization_name,
                                                   vocab=vocab,
                                                   elmo=use_elmo,
                                                   elmo_weight=elmo_weight)
            self.aggressive_compression = aggressive_compression

        self.use_elmo = use_elmo
        if use_elmo:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
            # print(self.elmo.get_output_dim())
            self._context_layer = PytorchSeq2SeqWrapper(
                torch.nn.LSTM(word_embedding_dim + self.elmo.get_output_dim(), hidden_dim,
                              batch_first=True, bidirectional=True))
        else:

            self._context_layer = PytorchSeq2SeqWrapper(
                torch.nn.LSTM(word_embedding_dim, hidden_dim,
                              batch_first=True, bidirectional=True))

        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=word_embedding_dim)
        if pretrain_embedding_file is not None:
            logger = logging.getLogger()
            logger.info("Loading word embedding: {}".format(pretrain_embedding_file))
            token_embedding.from_params(vocab=vocab,
                                        params=Params({"pretrained_file": pretrain_embedding_file,
                                                       "embedding_dim": word_embedding_dim})
                                        )
        self._text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        # if span_encoder_type == 'self_attentive':
        #     self._span_encoder = SelfAttentiveSpanExtractor(
        #         self._context_layer.get_output_dim()
        #     )
        # else:
        #     raise NotImplementedError

        self._dropout = torch.nn.Dropout(p=dropout)
        self._max_decoding_steps = max_decoding_steps
        self._fix_edu_num = fix_edu_num
        if compression:
            pass
            # self.rouge_metrics_compression = self.compression_dec.rouge_metrics_compression
            # self.rouge_metrics_compression_upper_bound = self.compression_dec.rouge_metrics_compression_best_possible
        self.rouge_metrics_sent = self.sent_dec.rouge_metrics_sent
        self.mult_orac_sampling = mult_orac_sampling
        self.alpha = alpha
        initializer(self)
        if regularizer is not None:
            regularizer(self)
        self.counter = 0  # used for controlling compression and extraction

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],

                sent_label,
                sent_rouge,

                comp_rouge,
                comp_msk,
                comp_meta,
                comp_rouge_ratio,
                comp_seq_label,

                metadata
                ):
        """

        :param text: input words. [batch, max_sent, max_word]  0 for padding bit
        :param sent_label: [batchsize, n_oracles, max_decoding_step] [1,3,6,-1,-1] sorted descendingly by rouge
        :param sent_rouge: [batchsize, n_oracles]
        :param comp_rouge: [batchsize, max_sent, max_num_compression] padding with 0
        :param comp_msk: [batchsize, max_sent, max_num_compression, max_word] deletion mask of compression.
        :param comp_meta: ScatterableList
        :param comp_rouge_ratio: [batchsize, max_sent, max_num_compression] after-compression-rouge / baseline rouge
        :param comp_seq_label: [batchsize, max_sent] index of best compression. padded with -1.
        :param metadata: ScatterableList
        :return:
        """

        batch, sent_num, max_word_num = text['tokens'].size()
        batch_, nora, _max_dec = sent_label.size()
        batchsz, ora_ = sent_rouge.size()
        batchsize, max_sent, max_compre = comp_rouge.size()
        batchsize_, _max_sent, _max_compre, _max_word_num = comp_msk.size()
        bz, max_s = comp_seq_label.size()
        assert batchsize == bz == batchsize_ == batch_ == batch == batchsz
        assert sent_num == max_sent == max_s
        assert _max_word_num == max_word_num
        text_mask = util.get_text_field_mask(text, num_wrapping_dims=1).float()
        sent_blstm_output, document_rep = self.enc_doc.forward(inp=text, context_msk=text_mask)
        # sent_blstm_output: [batch, sent_num, hdim*2]
        # document_rep: [batch, hdim*2]
        sent_mask = text_mask[:, :, 0]

        if self.training:
            decoder_outputs_logit, decoder_outputs_prob, [decoder_states_h, decoder_states_c] = \
                self.sent_dec.forward(context=sent_blstm_output,
                                      context_mask=sent_mask,  # batch size, enc sent num; [1,0]
                                      last_state=document_rep,  # batch size, dim;
                                      tgt=sent_label)
        else:
            decoder_outputs_logit, decoder_outputs_prob, [decoder_states_h, decoder_states_c] = \
                self.sent_dec.forward(context=sent_blstm_output,
                                      context_mask=sent_mask,  # batch size, enc sent num; [1,0]
                                      last_state=document_rep,  # batch size, dim;
                                      tgt=None)
        # Compute sent loss
        decoder_outputs_prob = flip_first_two_dim(decoder_outputs_prob)
        if not self.training:
            sent_label = sent_label[:, :, :self.sent_dec.max_dec_steps]
        sent_loss, ori_loss = self.sent_dec.comp_loss(decoder_outputs_prob=decoder_outputs_prob,
                                                      oracles=sent_label,
                                                      rouge=sent_rouge)

        # comp subsentence model
        # refine_subsent_selection() default is root        --subsentence
        sent_emission = self.refine_sent_selection(batchsz, self.comp_leadn, decoder_outputs_logit, sent_label,
                                                   decoder_outputs_prob, metadata)
        # run compression module
        if self.compression:
            # sent_label or decoder_outputs_logit: batch, t

            sent_emission = sent_emission.detach()
            ####
            #  sent_emission: t, batch_sz. [4, 13, 0, -1, -1]...
            ####

            assert sent_emission.size()[1] == batch_

            all_attn_dist, all_scores, all_reps = self.compression_dec.forward_parallel(
                sent_decoder_states=decoder_states_h,
                sent_decoder_outputs_logit=sent_emission,
                document_rep=document_rep,
                text=text,
                text_msk=text_mask,
                span=comp_msk)
            # all_reps: t, batch_size_, max_span_num, self.concat_size

            if self.aggressive_compression > 0:
                compression_loss = self.compression_dec.comp_loss(sent_emission, all_scores,
                                                                  comp_seq_label, comp_rouge,
                                                                  comp_rouge_ratio)
            elif self.aggressive_compression < 0:
                # Independent Classifier
                span_score = self.compression_dec.indep_compression_judger(all_reps)
                # span_prob: t, batch, max_span_num, 2
                compression_loss = self.compression_dec.comp_loss_inf_deletion(
                    sent_emission, comp_rouge, span_score, comp_rouge_ratio, self.loss_thres)
            else:
                raise NotImplementedError

        else:
            compression_loss = 0
        # Decoding:
        if (self.dbg is True) or (self.training is False):

            if self.compression:
                if self.aggressive_compression > 0:
                    self.compression_dec.decode(decoder_outputs_logit=sent_emission,
                                                span_score=all_scores,
                                                metadata=metadata,
                                                span_meta=comp_meta, span_seq_label=comp_seq_label,
                                                span_rouge=comp_rouge,
                                                compress_num=self.aggressive_compression
                                                )
                elif self.aggressive_compression < 0:
                    # for thres in self.compression_dec.keep_thres:
                    span_score = span_score.detach()
                    self.compression_dec.decode_inf_deletion(sent_decoder_outputs_logit=sent_emission,
                                                             span_prob=span_score,
                                                             metadata=metadata,
                                                             span_meta=comp_meta,
                                                             span_rouge=comp_rouge,
                                                             keep_threshold=self.compression_dec.keep_thres)
                else:
                    raise NotImplementedError

        if self.compression:
            if random.random() < 0.002:
                print("Comp loss: {}".format(compression_loss))
            if self.comp_leadn > 0:
                return {"loss": compression_loss}
            else:
                # print("sent: {}\tcomp: {}".format(sent_loss, compression_loss))
                return {"loss": sent_loss + self.alpha * compression_loss, "sent_loss": sent_loss,
                        "compression_loss": compression_loss}
        else:
            return {"loss": sent_loss, "sent_loss": ori_loss}

    def get_metrics(self, reset: bool = False, note="") -> Dict[str, float]:
        # ROUGE
        _rouge_met_sent = self.rouge_metrics_sent.get_metric(reset=reset, note=note)
        if self.compression:
            # _rouge_met_compression_ub = self.rouge_metrics_compression_upper_bound.get_metric(reset, note=note)
            if self.aggressive_compression < 0:
                dic = self.compression_dec.rouge_metrics_compression_dict
                new_dict = para_get_metric(dic, reset, note)
                return {**new_dict, **_rouge_met_sent}
            else:
                pass
                # _rouge_met_compression = self.rouge_metrics_compression.get_metric(reset=reset, note=note)
        else:
            return _rouge_met_sent

    def decode(self, output_dict: Dict[str, torch.Tensor],
               max_decoding_steps: int = 6,
               fix_edu_num: int = -1,
               min_step_decoding=2) -> Dict[str, torch.Tensor]:
        """

        :param output_dict: ["decoder_outputs_logit", "decoder_outputs_prob"
            "spans", "loss", "label",
            "metadata"["doc_list", "abs_list"] ]
        :param max_decoding_steps:
        :return:
        """
        assert output_dict["loss"] is not None
        meta = output_dict["metadata"]
        output_logit = output_dict["decoder_outputs_logit"][:max_decoding_steps, :]
        output_logit = output_logit.cpu().numpy()

        output_prob = output_dict["decoder_outputs_prob"]

        span_info = output_dict["spans"]
        label = output_dict["label"]

        batch_size = len(meta)
        for idx, m in enumerate(meta):
            _label = label[idx]  # label: batch, step
            logit = output_logit[:, idx]
            prob = output_prob[:, idx, :]  # step, src_len
            sp = span_info[idx]
            name = m["name"]
            formal_doc = m['doc_list']
            formal_abs = m['abs_list']
            abs_s = convert_list_to_paragraph(formal_abs)

            _pred = []

            if fix_edu_num:
                prob = prob[:fix_edu_num, :]
                prob[:, 0] = -1000
                max_idx = torch.argmax(prob, dim=1)
                # predict exactly fix_edu_num of stuff
                # if 0 or unreachable, use prob
                logit = max_idx.cpu().numpy()
                for l in logit:
                    try:
                        start_idx = int(sp[l][0].item())
                        end_idx = int(sp[l][1].item())
                        words = formal_doc[start_idx:end_idx + 1]  # list
                        _pred.append(' '.join(words).replace('@@SS@@', ''))
                    except IndexError:
                        logging.error("----Out of range-----")
            else:
                # reach minimum requirement (2) of edu num and follow the prediction
                max_idx = torch.argmax(prob, dim=1)
                logit = max_idx.cpu().numpy()

                prob[:, 0] = -1000
                backup_max_idx = torch.argmax(prob, dim=1)
                backup_logit = backup_max_idx.cpu().numpy()

                for t, l in enumerate(logit):
                    if t < min_step_decoding and abs(l) < 0.01:
                        l = backup_logit[t]
                    elif abs(l) < 0.01:
                        break
                    try:
                        start_idx = int(sp[l][0].item())
                        end_idx = int(sp[l][1].item())
                        words = formal_doc[start_idx:end_idx + 1]  # list
                        _pred.append(' '.join(words).replace('@@SS@@', ''))
                    except IndexError:
                        logging.error("----Out of range-----")

            if random.random() < 0.1:
                log_predict_example(name=name, pred_label=logit,
                                    gold_label=_label, pred_abs=_pred, gold_abs=abs_s)
            self.rouge_metrics(pred=_pred, ref=[abs_s])
        return output_dict

    def refine_sent_selection(self, batchsz, comp_leadn,
                              decoder_outputs_logit, sent_label, decoder_outputs_prob,
                              metadata):

        if comp_leadn > 0:
            part = metadata[0]['part']
            if part == 'cnn':
                comp_leadn -= 1

            lead3 = torch.ones_like(decoder_outputs_logit, dtype=torch.long, device=self.device) * -1
            assert decoder_outputs_logit.size()[1] == batchsz
            _t = decoder_outputs_logit.size()[0]

            for i in range(comp_leadn):
                if _t > i and comp_leadn >= i:
                    lead3[i, :] = i

            sent_emission = lead3
        else:
            rand_num = random.random()
            if self.training and (rand_num < 0.9):
                # use ground truth
                sent_emission = sent_label[:, 0, :]
                sent_emission = flip_first_two_dim(sent_emission.long())
            else:
                sent_decoded = self.sent_dec.decode(decoder_outputs_prob, metadata, sent_label)
                sent_emission = sent_decoded
                # print(sent_emission.size()[0])
                # print(decoder_outputs_logit.size()[0])
                # assert sent_emission.size()[0] == decoder_outputs_logit.size()[0]
        return sent_emission
