from unittest import TestCase
from neusum.nn_modules.enc.transformer import attention,EncoderLayer,Encoder,attention
import torch

class TestEncoderLayer(TestCase):
    def test_forward(self):
        # test attention module it self
        inp_dim = 11
        hid_dim = 13
        batch = 3
        max_sent = 7
        max_t = 19

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
        attention()
        EncoderLayer()