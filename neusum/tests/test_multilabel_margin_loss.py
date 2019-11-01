from unittest import TestCase
from neusum.service.basic_service import multilabel_margin_loss
import torch


class TestMultilabel_margin_loss(TestCase):
    def test_multilabel_margin_loss(self):
        inp = torch.zeros((2, 3))
        inp[0, 0] = 0.2
        inp[0, 1] = 0.5
        inp[0, 2] = 0.3
        inp[1, 1] = 1
        print(inp)
        tgt = torch.ones((2, 3), dtype=torch.long)
        # tgt[0,1] =1

        loss = multilabel_margin_loss(inp, tgt)
        print(loss)
