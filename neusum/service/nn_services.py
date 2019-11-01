from neusum.service.common_imports import *


def ptr_network_index_select(device, context, select_idx):
    # context: batch, seq, dim
    # selec_idx: batch
    batch, seq, dim = context.size()
    batch_ = select_idx.size()[0]
    assert batch == batch_
    mask = torch.from_numpy(np.linspace(start=0, stop=batch * seq, num=batch, endpoint=False, dtype=np.int)).to(device)
    select_idx = select_idx + mask

    flatten_context = context.view(batch * seq, dim)
    # print(flatten_context.size())
    output = torch.index_select(flatten_context, 0, select_idx)
    # print("output",output.size())
    return output
