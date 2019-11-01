from unittest import TestCase

from neusum.nn_modules.enc.transformer import MultiHeadedAttention
class TestMultiHeadedAttention(TestCase):
    def test_forward(self):
        MultiHeadedAttention(h=4,d_model=100)

