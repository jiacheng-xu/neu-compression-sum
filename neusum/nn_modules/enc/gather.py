# given a sequence, produce the representation of the sequence
import torch


def select_gather(name):
    if name == 'sum':
        return masked_sum
    elif name == 'mean':
        return masked_mean
    elif name == 'attn':
        raise NotImplementedError
    else:
        raise ValueError("sum or mean or attn expected.")


def masked_sum(inp, dim, msk=None):
    """

    :param inp: Float [*0,*1, ..., *n, hid_dim]
    :param dim: int: i
    :param msk: {0,1} [*0,*1, ..., *n,#]
    :return: [*0,*1, ..*i-1, *i+1, .., *n,, hid_dim]
    """
    if msk is not None:
        if (msk.size()[-1] != inp.size()[-1]) or (msk.size() != inp.size()):
            msk = msk.unsqueeze(-1)
        # print("dinner")
        # print(inp.size())
        # print(msk.size())
        inp = inp * msk
    result = torch.sum(inp, dim)
    # print(result.size())
    return result


def masked_mean(inp, dim, msk=None):
    # inp: [ *, dimension]
    # msk: [ *] in [0,1]
    if msk is not None:
        sum_of_mask = torch.sum(msk, dim=dim, keepdim=True) + 1
        if (msk.size()[-1] != inp.size()[-1]) or (msk.size() != inp.size()):
            msk = msk.unsqueeze(-1)
        inp = inp * msk
    if msk is not None:
        result = torch.sum(inp, dim)
        result = result / sum_of_mask
    else:
        result = torch.mean(inp, dim)
    # print(result.size())
    return result


import allennlp
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder


class GatherCNN(torch.nn.Module):

    def __init__(self, input_dim, num_filters, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.cnn_module = CnnEncoder(
            embedding_dim=input_dim, num_filters=num_filters, output_dim=output_dim
        )

    def forward(self, tokens,
                msk=None):
        """

        :param tokens: batch_size, num_tokens, embedding_dim
        :param msk:
        :return:
        """
        output = self.cnn_module.forward(tokens=tokens, mask=msk)
        return output

    def get_output_dim(self):
        return self.output_dim


def cnn(tokens, msk, embedding_dim, num_filters, output_dim):
    cnn_module = CnnEncoder(
        embedding_dim=embedding_dim, num_filters=num_filters, output_dim=output_dim
    )
    output = cnn_module.forward(tokens=tokens, mask=msk)
    print(output.size())
    print(output)


if __name__ == '__main__':
    inp_dim = 11
    hid_dim = 13
    batch = 3
    max_sent = 20
    max_t = 19
    context = torch.ones((batch, max_sent, inp_dim))
    context.uniform_()

    context_msk = torch.ones((batch, max_sent))  # dtype?
    context_msk[batch - 2, 1:4] = 0
    context_msk[batch - 1, 2:] = 0
    context_msk[batch - 3, 4:] = 0
    print(context.size())
    print(context_msk.size())
    print(context_msk)
    # masked_mean(inp=context, dim=1, msk=context_msk)
    cnn(tokens=context, msk=context_msk, embedding_dim=inp_dim, num_filters=5, output_dim=hid_dim)
