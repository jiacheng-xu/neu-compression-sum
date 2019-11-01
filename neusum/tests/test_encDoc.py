from unittest import TestCase
import torch
from neusum.nn_modules.enc import EncDoc
import numpy as np

from neusum.nn_modules.sent_dec import SentRNNDecoder


def s(x):
    print(x.size())


from neusum.nn_modules.compression_decoder import CompressDecoder


class TestEncDoc(TestCase):
    def test_forward(self):
        inp_dim = 11
        hid_dim = 13
        batch = 3
        max_sent = 7
        max_t = 19
        model = EncDoc(inp_dim, hid_dim, compression=True, vocab=None)

        context = torch.ones((batch, max_sent, max_t, inp_dim))
        context.uniform_()

        context_msk = torch.ones((batch, max_sent, max_t))  # dtype?
        context_msk[batch - 2, 1, 5:] = 0
        context_msk[batch - 1, 0, 2:] = 0
        context_msk[batch - 3, 1, 4:] = 0
        print(context_msk)
        span = torch.ones((batch, max_sent, 4, 2))
        span[:, 0, 1, 0] = 5
        span[:, 0, 1, 1] = 17
        span[:, 1, 2, 0] = 2
        span[:, 1, 2, 1] = 11
        span[batch - 1, 1, :] = -1
        # span[1, 3, 0] = 5
        # span[1, 3, 1] = 9
        print("Span: {}".format(span))
        print(span.size())

        sent_level_context_msk = context_msk[:, :, 0]
        sent_blstm_output, document_rep, attn_dist_of_phrases, \
        reorg_spans_rep, reorg_span_msk = model.forward(context=context,
                                                        context_msk=context_msk, spans=span)

        sent_dec_1 = SentRNNDecoder(device=torch.device("cpu"), dec_hidden_size=hid_dim * 2,
                                    dec_input_size=hid_dim * 2, fixed_dec_step=3)

        # sent_dec_2 = SentRNNDecoder(device=torch.device("cpu"),dec_hidden_size=23,
        #                dec_input_size=hid_dim*2,fixed_dec_step=-1)

        noracles = 2
        lis = list(range(1, 4)) + [0]
        tgt_np = np.asarray(lis * batch * noracles)
        tgt = torch.from_numpy(tgt_np).view(batch, noracles, 4)
        print(tgt)

        decoder_outputs_logit, decoder_outputs_prob, [decoder_states_h, decoder_states_c] = \
            sent_dec_1.forward(context=sent_blstm_output,
                               context_mask=sent_level_context_msk,
                               last_state=document_rep, tgt=tgt)
        s(tgt)
        s(decoder_outputs_logit)
        print(decoder_outputs_logit)
        s(decoder_outputs_prob)
        print("-" * 40)
        print("Compression")
        compression = CompressDecoder(context_dim=hid_dim * 2, doc_dim=hid_dim * 2,
                                      sent_dim=hid_dim * 2, dec_state_dim=hid_dim * 2)
        compression.forward(decoder_states=decoder_states_h,
                            decoder_outputs_logit=decoder_outputs_logit,
                            document_rep=document_rep,
                            sent_rep=sent_blstm_output,
                            spans_rep=reorg_spans_rep,
                            span_msk=reorg_span_msk)
