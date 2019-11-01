from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

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
]*1000
import time

from typing import List

def run_on_one_gpu(root, input:List, cuda_device_id):
    arc = load_archive(
        archive_file=os.path.join(root, "elmo-constituency-parser-2018.03.14.tar.gz"),
        cuda_device=cuda_device_id)
    predictor = Predictor.from_archive(archive=arc)
    predictor.predict(
        sentence=sentence
    )

import shutil

def clear_dir(_path):
    if os.path.isdir(_path):
        shutil.rmtree(_path)
    os.mkdir(_path)


import os



if __name__ == '__main__':
    data_name = "dm"
    root = "/scratch/cluster/jcxu/data"
    eve_path = "/backup3/jcxu/data"

    original_file_path = root + "sent_{}.txt".format(data_name)
    what_we_have_now = os.path.join(root, "allen-output-{}".format(data_name))
    if not os.path.isdir(what_we_have_now):
        os.mkdir(what_we_have_now)
