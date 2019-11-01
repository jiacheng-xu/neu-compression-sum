import json
#
# data = {
#     'seg': [1, 4, 5],
#     'rouge': [0.3, 0.4, 0.1],
# }
#
# json_str = json.dumps(data)
#
# x = '{"2": {"nlabel": 2, "data": {"0.23002": {"label": [15, 59], "R1": 0.31579, "R2": 0.11111, "RL": 0.26316, "R": 0.23002, "nlabel": 2}, "0.22359666666666667": {"label": [24, 44], "R1": 0.35555, "R2": 0.09302, "RL": 0.22222, "R": 0.22359666666666667, "nlabel": 2}, "0.20025666666666667": {"label": [15, 44], "R1": 0.32432, "R2": 0.11429, "RL": 0.16216, "R": 0.20025666666666667, "nlabel": 2}, "0.20697666666666667": {"label": [44, 59], "R1": 0.30769, "R2": 0.10811, "RL": 0.20513, "R": 0.20697666666666667, "nlabel": 2}}, "best": {"label": [15, 59], "R1": 0.31579, "R2": 0.11111, "RL": 0.26316, "R": 0.23002, "nlabel": 2}}, "3": {"nlabel": 3, "data": {"0.24459": {"label": [15, 44, 59], "R1": 0.36363, "R2": 0.14286, "RL": 0.22728, "R": 0.24459, "nlabel": 3}, "0.3083333333333333": {"label": [15, 56, 59], "R1": 0.44, "R2": 0.125, "RL": 0.36, "R": 0.3083333333333333, "nlabel": 3}, "0.25795000000000007": {"label": [24, 44, 59], "R1": 0.38462, "R2": 0.12, "RL": 0.26923, "R": 0.25795000000000007, "nlabel": 3}, "0.26303666666666664": {"label": [15, 24, 59], "R1": 0.35294, "R2": 0.12245, "RL": 0.31372, "R": 0.26303666666666664, "nlabel": 3}}, "best": {"label": [15, 56, 59], "R1": 0.44, "R2": 0.125, "RL": 0.36, "R": 0.3083333333333333, "nlabel": 3}}, "4": {"nlabel": 4, "data": {"0.31128666666666666": {"label": [15, 44, 56, 59], "R1": 0.46428, "R2": 0.14815, "RL": 0.32143, "R": 0.31128666666666666, "nlabel": 4}, "0.2721766666666667": {"label": [15, 56, 59, 89], "R1": 0.375, "R2": 0.12903, "RL": 0.3125, "R": 0.2721766666666667, "nlabel": 4}, "0.27070666666666665": {"label": [15, 24, 44, 59], "R1": 0.38597, "R2": 0.14545, "RL": 0.2807, "R": 0.27070666666666665, "nlabel": 4}, "0.27513333333333334": {"label": [15, 56, 59, 87], "R1": 0.39286, "R2": 0.11111, "RL": 0.32143, "R": 0.27513333333333334, "nlabel": 4}}, "best": {"label": [15, 44, 56, 59], "R1": 0.46428, "R2": 0.14815, "RL": 0.32143, "R": 0.31128666666666666, "nlabel": 4}}, "5": {"nlabel": 5, "data": {"0.29664": {"label": [15, 44, 56, 59, 89], "R1": 0.4, "R2": 0.14706, "RL": 0.34286, "R": 0.29664, "nlabel": 5}, "0.2810033333333333": {"label": [15, 44, 56, 59, 87], "R1": 0.41935, "R2": 0.13334, "RL": 0.29032, "R": 0.2810033333333333, "nlabel": 5}, "0.2487366666666667": {"label": [15, 56, 59, 87, 89], "R1": 0.34286, "R2": 0.11764, "RL": 0.28571, "R": 0.2487366666666667, "nlabel": 5}, "0.25488333333333335": {"label": [15, 24, 44, 59, 89], "R1": 0.33803, "R2": 0.14493, "RL": 0.28169, "R": 0.25488333333333335, "nlabel": 5}}, "best": {"label": [15, 44, 56, 59, 89], "R1": 0.4, "R2": 0.14706, "RL": 0.34286, "R": 0.29664, "nlabel": 5}}}'
# print(x)
# data = json.loads(x)
# print(data)
# import pprint
#
# pprint.pprint(x)

# rouge: batchsize, n_of_oracles
# tgt: batchsize, n_of_oracles, dec_time_step
import torch
import numpy as np
#
# batch_sz = 3
# n_ora = 4
# dec_step = 5
# rouge = np.arange(start=0, stop=batch_sz * n_ora)
# rouge = torch.from_numpy(np.reshape(rouge * 0.1, [batch_sz, n_ora]))
# tgt = np.arange(start=0, stop=batch_sz * n_ora * dec_step)
# tgt = torch.from_numpy(np.reshape(tgt, [batch_sz, n_ora, dec_step]))
# print(rouge.size())
# print(rouge)
# print(torch.__version__)
# from torch.utils.data import sampler
# sample = list(sampler.RandomSampler(range(16)))

import numpy as np

#
# lists = [[0]*n_ora for _ in range(batch_sz)]
# import random
# print(lists)
#
# for idx, _ in enumerate(lists):
#     lists[idx][random.randint(0,n_ora-1)] = 1
# print(lists)
# torch_rand_idx = torch.ByteTensor(lists).unsqueeze(2)
# print(tgt)
# x = torch.masked_select(tgt,torch_rand_idx).view(batch_sz,dec_step)
# print(x.size())
# print(x)
# exit()
"""
IGNORE_IDX = -1
print("PyTorch version: {}".format(torch.__version__))
xe = torch.nn.CrossEntropyLoss(reduction='none')
input = torch.randn(3, 5, requires_grad=True)
print("Input: {}".format(input))

#### test 1:
target = torch.empty(3, dtype=torch.long).random_(5)
target[2] = -4  # should be an error
print("Target: {}".format(target))
loss = xe(input, target)
print("Loss before reduction: {}".format(loss))
output = torch.mean(loss)
print("Loss after reduction: {}".format(output))
output.backward()
print('-' * 20)

#### test 2:
target = torch.empty(3, dtype=torch.long).random_(5)
target[2] = IGNORE_IDX  # should be an expected case
print("Target: {}".format(target))
loss = xe(input, target)
print("Loss before reduction: {}".format(loss))
output = torch.mean(loss)
print("Loss after reduction: {}".format(output))
output.backward()

### test 3: run test 2 again
#### test 2:
target = torch.empty(3, dtype=torch.long).random_(5)
target[2] = IGNORE_IDX  # should be an expected case
print("Target: {}".format(target))
loss = xe(input, target)
print("Loss before reduction: {}".format(loss))
output = torch.mean(loss)
print("Loss after reduction: {}".format(output))
output.backward()

"""
"""
time=3
batch =2
max_comp = 5
decoder_outputs_logit = torch.ones((time, batch),dtype=torch.long)
comp_rouge_ratio = torch.ones((batch,max_comp))
comp_rouge_ratio.uniform_(0.7,1.2)

decoder_outputs_logit[0,1] = 3
decoder_outputs_logit[0,0] = 2
decoder_outputs_logit[2,0] = -1
from neusum.nn_modules.compression_decoder import two_dim_index_select
print(decoder_outputs_logit)
print(comp_rouge_ratio)
thres=0.9
for t in range(time):
    decoder_outputs_logit_t = decoder_outputs_logit[t]
    out = two_dim_index_select(inp=comp_rouge_ratio,index=decoder_outputs_logit_t)
    print(out)
    label = torch.gt(out, thres).long()
    print(label)
"""

# path = "/backup3/jcxu/data/cnndm_vocab/tokens.txt"
# with open(path, 'r') as fd:
#     lines = fd.read().splitlines()
# lines = lines[1:60]
# print(lines)

import torch

test = torch.ones([2, 4], dtype=torch.float32)
test[0][1] = 0.99
test[1][2] = 0.99
test[0][2] = 0.98
test[1][3] = 0.99999
print(test)
x = torch.eq(test, 0.99)
print(x)
