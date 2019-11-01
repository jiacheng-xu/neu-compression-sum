

import abc
import torch.nn as nn
class EncoderBase(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_output_dim(self):
        pass