from nltk.tokenize import sent_tokenize
import os
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

from typing import List
import subprocess

sentence = [
               "New Delhi 's government district is still under a lockdown . New Delhi (CNN) -- Police continued to block off a key government district in India Tuesday in an effort to stop protesters angered about the gang rape of a woman .",
               "All of them repeated the same chant : `` There is no God but Allah , and a martyr is loved by Allah . ''",
               "Rabbis Mendel Epstein , 69 ; Jay Goldstein , 60 ; and Binyamin Stimler , 39 , were found guilty on one count of conspiracy to commit kidnapping in New Jersey federal court .",
               "Duvalier 's lawyer says `` all that money '' has been given for Haitian relief .",
               "Her son , David Lilienstein , told CBC News that his mother died in Toronto on Wednesday night from a rare form of cancer first diagnosed last October .",
               "America 's top technology companies have approval ratings that most politicians can only dream of , according to a new poll .",
               "Social Security numbers , phone numbers or other personally identifying information are securely stored.",
               "He 's just the most positive human being in the world , '' Daly said .",
               "The statement struck at the heart by critics , who said the new law would allow businesses to discriminate"
           ] * 1000


def sent_tok(dir, name):
    with open(os.path.join(dir, name + '.doc'), 'r') as fd:
        doc_list = fd.read()
    doc_list = sent_tokenize(doc_list)
    doc_list = [x for x in doc_list if (len(x) <= 500 and len(x) > 20)]
    with open(os.path.join(dir, name + '.doc'), 'w') as fd:
        fd.write("\n".join(doc_list))

    with open(os.path.join(dir, name + '.abs'), 'r') as fd:
        abss = fd.read()
    abs_list = sent_tokenize(abss)
    with open(os.path.join(dir, name + '.abs'), 'w') as fd:
        fd.write("\n".join(abs_list))


import multiprocessing

predictor = None


def prepare_toked_file(read_path):
    file_names = [x.split(".")[0] for x in os.listdir(read_path) if x.endswith(".doc")]
    l = len(file_names)
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    pool.starmap(sent_tok, zip([read_path] * l, file_names))
    pool.close()
    pool.join()
    print("finish sentence tokenization!")
    # sent_tok(read_path, file_names[0])


def pred_batch(sentences):
    rt_bag = []
    for s in sentences:
        out = predictor.predict(
            sentence=s
        )
        rt_bag.append(out['tree'])
    return rt_bag


def divide_a_list_into_n_pieces(inp_list, n):
    l = len(inp_list)


def wt_todo_to_disk(path_todo, path_read, path_wt, files, batch_num: int) -> List[str]:
    l = len(files)
    rd_files = [os.path.join(path_read, x) for x in files]
    wt_files = [os.path.join(path_wt, x + '.json') for x in files]
    assert len(rd_files) == len(wt_files)
    unit_size = int(l / batch_num)
    bag = []
    for idx in range(batch_num):
        if idx == l - 1:
            fr = rd_files[idx * unit_size:]
            fw = wt_files[idx * unit_size:]
        else:
            fr = rd_files[idx * unit_size:(idx + 1) * unit_size]
            fw = wt_files[idx * unit_size:(idx + 1) * unit_size]
        fline = "\n".join([r + '\t' + w for r, w in zip(fr, fw)])
        with open(os.path.join(path_todo, "task-{}".format(idx)), 'w') as fd:
            fd.write(fline)
        print("Finish writing {}".format(os.path.join(path_todo, "task-{}".format(idx))))
        bag.append(os.path.join(path_todo, "task-{}".format(idx)))
    return bag


def run_on_server_w_gpus(path_root, path_bf, path_af, path_model, path_exComp):
    # compare bf after
    file_gap = compare_bf_and_af(path_bf, path_af)
    task_address = wt_todo_to_disk(path_todo=path_root, path_read=path_bf, path_wt=path_af, files=file_gap,
                                   batch_num=1)
    cnt = 0
    for ta in task_address:
        cmd = (
            "python {}/neusum/data/py_run_gpu.py {} {} {}".format(path_exComp,
                                                                  ta, cnt % 8, path_model))
        cnt += 1
        print(cmd)
        # command = cmd.split(' ')
        # subprocess.call(command)


def run_on_one_gpu(path_model, input: List = None, cuda_device_id: int = 0):
    arc = load_archive(
        archive_file=path_model,
        cuda_device=cuda_device_id)
    predictor = Predictor.from_archive(archive=arc)
    bag = []
    for s in input:
        out = predictor.predict(
            sentence=s
        )
        print(out["trees"])

        bag.append(out['trees'])
    return bag


import random


def compare_bf_and_af(path_bf, path_af) -> List:
    files_bf = os.listdir(path_bf)
    files_af = os.listdir(path_af)
    files_af_match = [".".join(x.split(".")[:2]) for x in files_af]
    file_gap = list(set(files_bf) - set(files_af_match))
    random.shuffle(file_gap)
    return file_gap


import shutil


def clear_dir(_path):
    if os.path.isdir(_path):
        shutil.rmtree(_path)
    os.mkdir(_path)


import sys

if __name__ == '__main__':
    # run stage 1 first
    print("data name     servername: titan or eve or cc")
    data_name = sys.argv[1]
    server_name = sys.argv[2]
    if server_name =='titan':
        path_root = '/scratch/cluster/jcxu/data/'
        path_exc = '/scratch/cluster/jcxu/exComp'
    elif server_name == 'eve':
        path_root = '/backup3/jcxu/data/'
        path_exc = '/backup3/jcxu/exComp'
    else:
        path_root = '/home/cc/'
        path_exc = '/home/cc/exComp'
    # data_name = 'dm'
    if data_name == 'dm':
        full_dataname = 'dailymail'
    else:
        full_dataname = data_name

    path_data_bf = path_root + 'snlp-{}-parse'.format(data_name)
    path_data_af = os.path.join(path_root, 'allen-output-{}'.format(data_name))
    done = False
    if done:
        prepare_toked_file(path_data_bf)

    # run allen
    path_model = os.path.join(path_root, "elmo-constituency-parser-2018.03.14.tar.gz")
    if not os.path.isdir(path_data_af):
        os.mkdir(path_data_af)
    run_on_server_w_gpus(path_root=path_root,
                         path_bf=path_data_bf,
                         path_af=path_data_af,
                         path_model=path_model,
                         path_exComp=path_exc)

    # run several times
