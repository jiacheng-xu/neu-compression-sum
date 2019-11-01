import os

os.chdir("/scratch/cluster/jcxu/data/2merge-nyt")
import json


def read_one_file(name):
    with open(name, "r") as fd:
        lines = fd.read().splitlines()
    lines = [json.loads(l) for l in lines]
    lines = [l['name'] for l in lines]
    x = json.dumps(lines)
    with open("nyt-train.txt","w") as wfd:
        wfd.write(x)


read_one_file("train.txt")
