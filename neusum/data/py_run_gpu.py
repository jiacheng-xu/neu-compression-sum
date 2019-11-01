import os, subprocess
import sys
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

args = sys.argv[1:]
# ta, cnt % 8, path_model
ta = args[0]
cudaid = int(args[1])
path_model = args[2]
arc = load_archive(archive_file=path_model, cuda_device=cudaid)
predictor = Predictor.from_archive(archive=arc)

with open(ta, 'r') as fd:
    lines = fd.read().splitlines()
import json
import random

for l in lines:
    rd, wt = l.split("\t")
    with open(rd, 'r') as fd:
        doc = fd.read().splitlines()
    output = []
    for d in doc:
        out = predictor.predict(
            sentence=d
        )
        output.append(out['trees'])
    x = "\n".join(output)
    if random.random() < 0.001:
        print(x)
    with open(wt, 'w') as fd:
        fd.write(x)
