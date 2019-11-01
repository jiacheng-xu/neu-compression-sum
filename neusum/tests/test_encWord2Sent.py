from unittest import TestCase
import torch
from neusum.nn_modules.enc.enc_doc import EncWord2Sent


class TestEncWord2Sent(TestCase):
    def test_forward(self):
        dim = 11
        hid_dim = 13
        module = EncWord2Sent(device=torch.device('cpu'), inp_dim=dim, hidden_dim=hid_dim, nenc_lay=2, dropout=0.1)
        batch = 4
        t = 19

        context = torch.ones((batch, t, dim))
        context.uniform_()
        context_msk = torch.ones((batch, t))  # dtype?
        context_msk[batch - 2, 5:] = 0
        context_msk[batch - 1, 15:] = 0
        context_msk[batch - 3, :] = 0
        print(context_msk)
        print("output dim: ".format(module.get_output_dim()))
        out = module.forward(context, context_msk=context_msk)
