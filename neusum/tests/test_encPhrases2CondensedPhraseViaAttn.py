from unittest import TestCase
from unittest import TestCase
import torch
from neusum.nn_modules.enc import EncWord2PhrasesViaMapping, EncPhrases2CondensedPhraseViaAttn


class TestEncPhrases2CondensedPhraseViaAttn(TestCase):
    def test_forward(self):
        dim = 11
        hid_dim = 13
        module = EncWord2PhrasesViaMapping(inp_dim=dim, hidden_dim=hid_dim, nenc_lay=2, dropout=0.1)
        batch = 3
        t = 19

        context = torch.ones((batch, t, dim))
        context.uniform_()
        context_msk = torch.ones((batch, t))  # dtype?
        context_msk[batch - 2, 5:] = 0
        context_msk[batch - 1, 15:] = 0
        print(context_msk)
        span = torch.ones((batch, 4, 2))
        span[:, 1, 0] = 5
        span[:, 1, 1] = 17
        span[:, 2, 0] = 2
        span[:, 2, 1] = 11
        span[:, 3, :] = -1
        span[1, 3, 0] = 5
        span[1, 3, 1] = 9
        print("Span: {}".format(span))
        spans_rep, span_msk = module.forward(context=context, context_msk=context_msk, span=span)
        doc_rep = torch.ones((batch, hid_dim))
        doc_rep.uniform_()
        print(spans_rep.size())
        mod = EncPhrases2CondensedPhraseViaAttn(context_dim=hid_dim * 2, sent_dim=hid_dim)
        mod.forward(spans_rep, span_msk, doc_rep)
