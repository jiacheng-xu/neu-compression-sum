# root = '/home/cc/final-cnn/merge'
import allennlp
import torch
import json
import os
from typing import List, Dict

import time

LB_SENT_NUM = 20


# y = [x["tokens"], x["pos_tags"], x["hierplane_tree"]["root"], x["hierplane_tree"]["text"]]


def read_files(path, fname):
    f_doc = os.path.join(path, fname + '.doc')
    f_doc_tree = os.path.join(path, fname + '.doc.tree')
    if os.path.isfile(f_doc_tree):
        return None
    with open(f_doc, 'r') as fd:
        lines = fd.read().splitlines()
    return {"f_doc": f_doc, "lines": lines}


class ConstPredictor():

    def __init__(self, cudaid: int = 2, use_cuda: bool = True):
        self.init_predtor(cudaid, use_cuda)

    def init_predtor(self, cudaid: int = 2, use_cuda=True):
        print("AllenNLP", allennlp.__version__)
        from allennlp.predictors.predictor import Predictor

        device = torch.device("cuda:{}".format(cudaid) if use_cuda else "cpu")
        print("Running on {}".format(device))
        predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
        predictor._model.to(device)
        self.predictor = predictor
        print("Finish loading")

    def const_parse_batch(self, sentences: List[str]):
        sent_list_dict = [{"sentence": x} for x in sentences]
        output = self.predictor.predict_batch_json(sent_list_dict)

        output = [{"text": o["hierplane_tree"]["text"],
                   "tokens": o["tokens"],
                   "pos_tags": o["pos_tags"],
                   "tree": o["hierplane_tree"]["root"]}
                  for o in output]
        # json_output = [json.dumps(o) for o in output]

        return output

    def load_database_and_parse(self, database: List[Dict]):
        lines_buff = []
        name_buff = []
        len_buff = []
        start = time.time()
        for kdx, d in enumerate(database):
            # runtime check:
            if os.path.isfile(d["f_doc"] + '.tree'):
                continue
            name_buff.append(d["f_doc"])
            lines_buff += d["lines"]
            len_buff.append(len(d["lines"]))
            if sum(len_buff) > LB_SENT_NUM:
                print("Parsing & Writing!")
                print("Idx: {} Time: {}".format(kdx, time.time() - start))
                output = self.const_parse_batch(lines_buff)
                cur = 0
                for j, l in enumerate(len_buff):
                    parse = output[cur:cur + l]
                    parse = json.dumps(parse)
                    with open(name_buff[j] + '.tree', 'w') as fd:
                        fd.write(parse)
                    cur += l
                lines_buff = []
                name_buff = []
                len_buff = []
        print("Final batch")
        if len_buff != []:
            output = self.const_parse_batch(lines_buff)
            cur = 0
            for j, l in enumerate(len_buff):
                parse = output[cur:cur + l]
                parse = json.dumps(parse)
                with open(name_buff[j] + '.tree', 'w') as fd:
                    fd.write(parse)
                cur += l
            # {"f_doc": f_doc, "lines": lines}
        # concated_string = self.const_parse_batch(lines)
        # with open(f_doc + '.tree', 'w') as fd:
        #     fd.write(concated_string)


import sys

if __name__ == '__main__':
    cudaid = int(sys.argv[1])
    pred_model = ConstPredictor(cudaid=cudaid, use_cuda=True)
    path_to_parse = '/backup2/jcxu/data/seperatecnn'
    files = os.listdir(path_to_parse)
    doc_file_names = [x[:-4] for x in files if x.endswith('.doc')]
    from random import shuffle

    shuffle(doc_file_names)
    start = time.time()
    data_base = []
    for idx, f in enumerate(doc_file_names):
        if idx % 200 == 0:
            print("Idx: {} Time: {}".format(idx, time.time() - start))
        data_base.append(read_files(path_to_parse, f))
        # pred_model.highlevel_cost_parse_and_write(path_to_parse, f)
    data_base = [x for x in data_base if x != None
                 ]
    pred_model.load_database_and_parse(data_base)
